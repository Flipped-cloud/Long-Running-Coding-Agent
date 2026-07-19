from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from longrun_agent.evals.experience_learning.evaluator import evaluate_mode_repeat
from longrun_agent.evals.experience_learning.executor import run_case
from longrun_agent.evals.experience_learning.generator import load_experience_config
from longrun_agent.evals.experience_learning.schemas import ExperienceLearningReport
from longrun_agent.exceptions import ConfigurationError
from longrun_agent.knowledge.store import KnowledgeStore


def run_experience_learning(
    config_path: Path,
    *,
    backend: str | None = None,
    mode: str | None = None,
    repeat: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_experience_config(config_path)
    selected_backend = backend or config.backend
    if selected_backend == "api" and config.model is None:
        raise ConfigurationError("experience learning backend='api' requires top-level model configuration")
    selected_modes = [mode] if mode else config.modes
    repeats = [repeat] if repeat is not None else list(range(1, config.repeats + 1))
    if dry_run:
        model = config.model
        payload = {
            "backend": selected_backend,
            "model_provider": model.provider if model is not None else "fake",
            "model_name": model.model_name if model is not None else "experience-learning-fake",
            "base_url_configured": bool(model and model.base_url),
            "api_key_configured": bool(model and os.environ.get(model.api_key_env)),
            "modes": selected_modes,
            "repeats": repeats,
            "cases": [case.case_id for case in config.cases],
            "output_root": str(config.output_root),
        }
        return payload
    all_case_results = []
    summaries = []
    for item_mode in selected_modes:
        for item_repeat in repeats:
            repeat_root = config.output_root / item_mode / f"repeat_{item_repeat:02d}"
            if repeat_root.exists():
                shutil.rmtree(repeat_root)
            repeat_root.mkdir(parents=True, exist_ok=True)
            _prepare_repeat_artifacts(repeat_root)
            case_results = []
            snapshot_skill_ids: list[str] = []
            for case in config.cases:
                frozen = case.knowledge_phase in {"frozen_transfer", "frozen_negative_control"}
                if frozen and not (repeat_root / "knowledge_snapshot.json").exists():
                    raise ConfigurationError(f"frozen phase {case.knowledge_phase} requires bootstrap snapshot")
                result = run_case(
                    case=case,
                    mode=item_mode,
                    repeat=item_repeat,
                    repeat_root=repeat_root,
                    verification=config.verification,
                    backend=selected_backend,
                    fail_fast_on_knowledge_error=config.fail_fast_on_knowledge_error,
                    model=config.model,
                    record_mutation_policy="frozen_records" if frozen else "read_write",
                    snapshot_skill_ids=snapshot_skill_ids,
                )
                case_results.append(result)
                all_case_results.append(result)
                if result.knowledge_error and config.fail_fast_on_knowledge_error:
                    raise RuntimeError(f"knowledge_error encountered in {item_mode}/repeat_{item_repeat:02d}/{case.case_id}")
                if case.knowledge_phase == "bootstrap_consolidation":
                    snapshot = KnowledgeStore(repeat_root / "knowledge").write_snapshot(repeat_root / "knowledge_snapshot.json")
                    snapshot_skill_ids = list(snapshot["skill_ids"])
            summary = evaluate_mode_repeat(
                mode=item_mode,
                repeat=item_repeat,
                case_results=case_results,
                knowledge_root=repeat_root / "knowledge",
                workspace_root=config.cases[0].repository,
            )
            summaries.append(summary)
            (repeat_root / "summary.json").write_text(
                json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
            )
    report = ExperienceLearningReport(seed=config.seed, results=summaries, case_results=all_case_results)
    report_payload = report.model_dump(mode="json")
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "report.json").write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return json.loads((config.output_root / "report.json").read_text(encoding="utf-8"))


def _prepare_repeat_artifacts(repeat_root: Path) -> None:
    for directory in [
        repeat_root / "cases",
        repeat_root / "workspaces",
        repeat_root / "knowledge" / "memories",
        repeat_root / "knowledge" / "skills",
        repeat_root / "projects",
        repeat_root / "telemetry",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for name in ["events.jsonl", "retrievals.jsonl", "uses.jsonl"]:
        path = repeat_root / "knowledge" / name
        path.touch(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("evals/experience_learning/config.yaml"))
    parser.add_argument("--backend", choices=["fake", "api"], default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experience_learning(args.config, backend=args.backend, mode=args.mode, repeat=args.repeat, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
