from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from longrun_agent.exceptions import ConfigurationError

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env_vars(value: Any) -> Any:
    """Expand ${VAR} strings recursively using the current environment."""

    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    model_name: str = ""
    base_url: str | None = None
    api_key_env: str = "MODEL_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = Field(default=4096, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    max_api_retries: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_model_name(self) -> ModelConfig:
        if self.provider != "fake" and not self.model_name:
            raise ValueError("model.model_name is required unless provider is fake")
        return self


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=30, ge=1)
    max_consecutive_errors: int = Field(default=5, ge=1)
    max_session_seconds: int = Field(default=180, ge=10)
    terminal_grace_turns: int = Field(default=0, ge=0, le=3)
    protocol_retries_per_step: int = Field(default=2, ge=0, le=5)


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path


class ReadFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_lines: int = Field(default=200, ge=1)
    max_chars: int = Field(default=30000, ge=100)


class WriteFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=200000, ge=1)
    save_diff: bool = True


class BashConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(default=120, ge=1)
    max_output_chars: int = Field(default=20000, ge=100)
    shell: bool = False


class ContextTokenCounterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["approximate", "tiktoken"] = "approximate"
    encoding: str = "cl100k_base"
    chars_per_token: float = Field(default=4.0, gt=0)
    per_message_overhead: int = Field(default=4, ge=0)
    per_tool_overhead: int = Field(default=32, ge=0)


class ContextPruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clear_stale_reads: bool = True
    supersede_repeated_bash: bool = True
    invalidate_tests_after_write: bool = True
    deduplicate_reminders: bool = True
    compact_old_tool_results: bool = True
    compact_output_max_chars: int = Field(default=1200, ge=100)


class StructuredHandoffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    use_model: bool = True
    max_protocol_retries: int = Field(default=2, ge=1)
    fallback_deterministic: bool = True
    max_confirmed_progress: int = Field(default=20, ge=1)
    max_hypotheses: int = Field(default=5, ge=0)
    max_next_actions: int = Field(default=5, ge=1)


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["full_history", "recent_window", "deterministic_prune", "structured_reset"] = "structured_reset"
    model_context_limit: int = Field(default=32768, gt=0)
    reserve_output_tokens: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=1024, ge=0)
    trigger_ratio: float = 0.72
    hard_stop_ratio: float = 0.90
    recent_full_turns: int = Field(default=6, ge=1)
    repeat_task_anchor_at_end: bool = True
    token_counter: ContextTokenCounterConfig = Field(default_factory=ContextTokenCounterConfig)
    pruning: ContextPruningConfig = Field(default_factory=ContextPruningConfig)
    structured_handoff: StructuredHandoffConfig = Field(default_factory=StructuredHandoffConfig)

    @model_validator(mode="after")
    def validate_budget(self) -> ContextConfig:
        if self.reserve_output_tokens + self.safety_margin_tokens >= self.model_context_limit:
            raise ValueError("context reserve_output_tokens + safety_margin_tokens must be < model_context_limit")
        if not 0 < self.trigger_ratio < self.hard_stop_ratio < 1:
            raise ValueError("context requires 0 < trigger_ratio < hard_stop_ratio < 1")
        return self


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    read_file: ReadFileConfig = Field(default_factory=ReadFileConfig)
    write_file: WriteFileConfig = Field(default_factory=WriteFileConfig)
    bash: BashConfig = Field(default_factory=BashConfig)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_flat_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        read_file = dict(migrated.get("read_file") or {})
        bash = dict(migrated.get("bash") or {})
        if "read_max_lines" in migrated:
            read_file.setdefault("max_lines", migrated["read_max_lines"])
        if "max_output_chars" in migrated:
            bash.setdefault("max_output_chars", migrated["max_output_chars"])
        if "bash_timeout_seconds" in migrated:
            bash.setdefault("timeout_seconds", migrated["bash_timeout_seconds"])
        if read_file:
            migrated["read_file"] = read_file
        if bash:
            migrated["bash"] = bash
        return migrated


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_root: Path = Path(".runs")
    save_prompts: bool = True
    save_full_tool_outputs: bool = True


class InitialPlanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["model", "file"] = "model"
    plan_file: Path | None = None
    min_tasks: int = Field(default=2, ge=1)
    max_tasks: int = Field(default=8, ge=1)
    max_protocol_retries: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> InitialPlanConfig:
        if self.min_tasks > self.max_tasks:
            raise ValueError("planning.initial_plan.min_tasks must be <= max_tasks")
        return self


class PlanningExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_project_sessions: int = Field(default=30, ge=1)
    attempts_before_decomposition: int = Field(default=1, ge=1)
    max_project_seconds: int = Field(default=540, ge=30)
    max_sessions_per_task: int = Field(default=2, ge=1)
    max_no_progress_sessions: int = Field(default=1, ge=0)
    final_verification_command: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    final_verification_timeout_seconds: int = Field(default=90, ge=1)


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: int = Field(default=3, ge=1)
    min_children: int = Field(default=2, ge=1)
    max_children: int = Field(default=5, ge=1)
    max_protocol_retries: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> DecompositionConfig:
        if self.min_children > self.max_children:
            raise ValueError("planning.decomposition.min_children must be <= max_children")
        return self


class BoundedSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    candidate_count: int = Field(default=3, ge=2, le=5)
    max_protocol_retries: int = Field(default=2, ge=1)


class PlanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["disabled", "static", "adaptive", "adaptive_search"] = "disabled"
    initial_plan: InitialPlanConfig = Field(default_factory=InitialPlanConfig)
    execution: PlanningExecutionConfig = Field(default_factory=PlanningExecutionConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    bounded_search: BoundedSearchConfig = Field(default_factory=BoundedSearchConfig)


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path(".runs/projects")
    atomic_write: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    workspace: WorkspaceConfig
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    config_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def resolve_paths(self) -> AppConfig:
        base = self.config_dir
        if not self.workspace.root.is_absolute():
            self.workspace.root = (base / self.workspace.root).resolve()
        if not self.telemetry.run_root.is_absolute():
            self.telemetry.run_root = (base / self.telemetry.run_root).resolve()
        if not self.state.root.is_absolute():
            self.state.root = (base / self.state.root).resolve()
        if self.planning.initial_plan.plan_file is not None and not self.planning.initial_plan.plan_file.is_absolute():
            self.planning.initial_plan.plan_file = (base / self.planning.initial_plan.plan_file).resolve()
        return self

    def sanitized(self) -> dict[str, Any]:
        return sanitize_config(self.model_dump(mode="json"))


def sanitize_config(data: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(data)
    if "model" in sanitized:
        sanitized["model"] = dict(sanitized["model"])
        sanitized["model"]["api_key_env"] = sanitized["model"].get("api_key_env", "")
    return sanitized


def apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    model = data.setdefault("model", {})
    agent = data.setdefault("agent", {})
    workspace = data.setdefault("workspace", {})
    overrides = {
        ("model", "model_name"): "LONGRUN_MODEL",
        ("model", "base_url"): "LONGRUN_API_BASE_URL",
        ("agent", "max_steps"): "LONGRUN_MAX_STEPS",
        ("workspace", "root"): "LONGRUN_WORKSPACE",
    }
    for (section, key), env_name in overrides.items():
        if env_name not in os.environ:
            continue
        target = {"model": model, "agent": agent, "workspace": workspace}[section]
        current = target.get(key)
        raw = os.environ[env_name]
        target[key] = int(raw) if isinstance(current, int) else raw
    return data


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = apply_env_overrides(expand_env_vars(raw))
    try:
        return AppConfig.model_validate({**expanded, "config_dir": config_path.parent})
    except Exception as exc:
        raise ConfigurationError(f"invalid config {config_path}: {exc}") from exc


def get_api_key(config: ModelConfig) -> str:
    value = os.environ.get(config.api_key_env, "")
    if config.provider != "fake" and not value:
        raise ConfigurationError(f"missing API key environment variable: {config.api_key_env}")
    return value


# Compatibility names for the previous implementation.
ApiConfig = ModelConfig
LoggingConfig = TelemetryConfig
