from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from longrun_agent.config import AppConfig, apply_env_overrides, expand_env_vars, load_config
from longrun_agent.context_probes.schemas import ProbeMode, ProbePosition


class ProbeGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_context_limit: int | None = Field(default=None, gt=0)
    token_length_tolerance: float | None = Field(default=None, gt=0, lt=1)
    filler_turn_tokens_min: int = Field(default=128, ge=1)
    filler_turn_tokens_max: int = Field(default=256, ge=1)


class ProbeEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    samples: int = Field(default=20, ge=1)
    lengths: list[int] = Field(default_factory=lambda: [2048, 4096, 8192, 12288])
    positions: list[ProbePosition] = Field(default_factory=lambda: ["beginning", "middle", "end"])
    position_ratios: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    modes: list[ProbeMode] = Field(default_factory=lambda: ["full_history", "recent_window", "deterministic_prune", "structured_reset"])
    mode_configs: dict[ProbeMode, Path]
    generation: ProbeGenerationConfig = Field(default_factory=ProbeGenerationConfig)
    constraints_per_case: list[int] = Field(default_factory=lambda: [2, 3, 5])
    token_length_tolerance: float = Field(default=0.08, gt=0, lt=1)
    max_answer_protocol_retries: int = Field(default=2, ge=0)
    output_root: Path = Path(".runs/context_evals")
    save_prompts: bool = True
    save_predictions: bool = True
    structured_reset_compactor: Literal["configured", "deterministic"] = "configured"
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_context(cls, data):
        if not isinstance(data, dict) or "context" not in data:
            return data
        migrated = dict(data)
        legacy = migrated.pop("context") or {}
        generation = dict(migrated.get("generation") or {})
        if "model_context_limit" in legacy:
            generation.setdefault("target_context_limit", legacy["model_context_limit"])
        if "token_length_tolerance" in legacy:
            generation.setdefault("token_length_tolerance", legacy["token_length_tolerance"])
        migrated["generation"] = generation
        warnings.warn(
            "ProbeEvalConfig top-level 'context' is deprecated and is treated as generation-only metadata; "
            "runtime context is loaded only from mode_configs.",
            DeprecationWarning,
            stacklevel=2,
        )
        return migrated

    @model_validator(mode="after")
    def validate_requested_modes(self) -> ProbeEvalConfig:
        missing = [mode for mode in self.modes if mode not in self.mode_configs]
        if missing:
            raise ValueError(f"missing mode_configs for requested modes: {missing}")
        if not self.lengths:
            raise ValueError("lengths must not be empty")
        if not self.positions:
            raise ValueError("positions must not be empty")
        if not self.position_ratios:
            raise ValueError("position_ratios must not be empty")
        return self

    def resolve_paths(self, base: Path) -> ProbeEvalConfig:
        data = self.model_dump()
        data["mode_configs"] = {mode: path if path.is_absolute() else (base / path).resolve() for mode, path in self.mode_configs.items()}
        output_root = self.output_root if self.output_root.is_absolute() else (base / self.output_root).resolve()
        data["output_root"] = output_root
        data["config_path"] = self.config_path
        return ProbeEvalConfig.model_validate(data)


def load_probe_eval_config(path: Path) -> ProbeEvalConfig:
    config_path = path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = ProbeEvalConfig.model_validate({**raw, "config_path": config_path})
    config = config.resolve_paths(config_path.parent)
    validate_mode_configs(config)
    return config


def validate_mode_configs(config: ProbeEvalConfig) -> None:
    model_fingerprint: dict | None = None
    non_context_fingerprint: dict | None = None
    for mode in config.modes:
        path = config.mode_configs[mode]
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        context = raw.get("context") or {}
        actual_mode = context.get("mode")
        if actual_mode != mode:
            raise ValueError(f"{path} context.mode={actual_mode!r} does not match mode_configs key {mode!r}")
        model = raw.get("model") or {}
        non_context = {key: value for key, value in raw.items() if key != "context"}
        if model_fingerprint is None:
            model_fingerprint = model
            non_context_fingerprint = non_context
            continue
        if model != model_fingerprint:
            raise ValueError("mode configs must use identical model settings")
        if non_context != non_context_fingerprint:
            raise ValueError("mode configs must match outside context-specific settings")


def load_mode_app_config(
    config: ProbeEvalConfig,
    mode: ProbeMode,
    *,
    force_fake_provider: bool = False,
    force_deterministic_compactor: bool = False,
) -> AppConfig:
    path = config.mode_configs[mode]
    if not force_fake_provider and not force_deterministic_compactor:
        return load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    expanded = apply_env_overrides(expand_env_vars(raw))
    if force_fake_provider:
        model = expanded.setdefault("model", {})
        model["provider"] = "fake"
        model["model_name"] = ""
    if force_deterministic_compactor:
        handoff = expanded.setdefault("context", {}).setdefault("structured_handoff", {})
        handoff["use_model"] = False
        handoff["fallback_deterministic"] = True
    return AppConfig.model_validate({**expanded, "config_dir": path.parent})
