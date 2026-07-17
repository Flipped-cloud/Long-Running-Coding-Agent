from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from longrun_agent.context_probes.config import ProbeEvalConfig, load_probe_eval_config
from longrun_agent.context_probes.executor import ProbeExecutor, ProviderFactory
from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.context_probes.reporting import write_dry_run_report, write_run_report
from longrun_agent.context_probes.schemas import ProbeMode, ProbeName

ALL_PROBES: list[ProbeName] = ["position", "multi_constraint", "state_tracking", "aggregation"]


def run_probe(
    *,
    config_path: Path,
    probe: str,
    lengths: list[int] | None = None,
    samples: int | None = None,
    seed: int | None = None,
    modes: list[str] | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    fake_provider_script: Path | None = None,
    provider_factory: ProviderFactory | None = None,
) -> dict[str, Any]:
    config = load_probe_eval_config(config_path)
    if lengths is not None:
        config.lengths = lengths
    if samples is not None:
        config.samples = samples
    if seed is not None:
        config.seed = seed
    if modes is not None:
        config.modes = [_as_mode(mode) for mode in modes]
    if output_dir is not None:
        config.output_root = output_dir.resolve()
    probes = ALL_PROBES if probe == "all" else [_as_probe(probe)]
    run_id = f"context-eval-{uuid4().hex[:12]}"
    started_at = datetime.now(UTC).isoformat()
    all_cases = []
    all_results = []
    executor = ProbeExecutor(config, provider_factory=provider_factory, fake_provider_script=fake_provider_script)
    for selected_probe in probes:
        cases = generate_cases(
            probe=selected_probe,
            lengths=config.lengths,
            samples=config.samples,
            seed=config.seed,
            positions=config.positions,
            position_ratios=config.position_ratios,
            constraints_per_case=config.constraints_per_case,
            token_length_tolerance=config.token_length_tolerance,
        )
        all_cases.extend(cases)
        if dry_run:
            continue
        for case in cases:
            for mode in config.modes:
                all_results.append(executor.run_case(case, mode))
    model = _model_name(config)
    if dry_run:
        summary = write_dry_run_report(
            output_dir=config.output_root,
            run_id=run_id,
            model=model,
            seed=config.seed,
            cases=all_cases,
            modes=config.modes,
            lengths=config.lengths,
            started_at=started_at,
        )
    else:
        summary = write_run_report(
            output_dir=config.output_root,
            run_id=run_id,
            model=model,
            seed=config.seed,
            cases=all_cases,
            results=all_results,
            modes=config.modes,
            lengths=config.lengths,
            started_at=started_at,
        )
    return {
        "run_id": run_id,
        "probe": probe,
        "lengths": config.lengths,
        "samples": config.samples,
        "modes": config.modes,
        "case_count": len(all_cases),
        "prediction_count": len(all_results),
        "output_dir": str(config.output_root),
        "summary": summary,
    }


def _as_probe(value: str) -> ProbeName:
    if value not in ALL_PROBES:
        raise ValueError(f"unknown probe: {value}")
    return value  # type: ignore[return-value]


def _as_mode(value: str) -> ProbeMode:
    valid = {"full_history", "recent_window", "deterministic_prune", "structured_reset"}
    if value not in valid:
        raise ValueError(f"unknown context mode: {value}")
    return value  # type: ignore[return-value]


def _model_name(config: ProbeEvalConfig) -> str:
    import yaml

    from longrun_agent.config import expand_env_vars

    names: set[str] = set()
    for mode in config.modes:
        raw = yaml.safe_load(config.mode_configs[mode].read_text(encoding="utf-8")) or {}
        expanded = expand_env_vars(raw)
        model = expanded.get("model") or {}
        name = str(model.get("model_name") or "fake")
        names.add(name)
    if len(names) != 1:
        raise ValueError(f"mode configs must resolve to the same model name: {sorted(names)}")
    return names.pop()


__all__ = ["ProbeMode", "ProbeName", "run_probe"]
