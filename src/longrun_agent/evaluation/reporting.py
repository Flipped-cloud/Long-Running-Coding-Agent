from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longrun_agent.evaluation.aggregation import aggregate_results
from longrun_agent.evaluation.schema import TrialResult


def write_evaluation_report(evaluation_dir: Path, results: list[TrialResult]) -> dict[str, Any]:
    aggregate = aggregate_results(results)
    payload = {
        "evaluation_id": results[0].descriptor.evaluation_id if results else evaluation_dir.name,
        "trial_count": len(results),
        "completed_count": sum(item.outcome is not None for item in results),
        "error_count": sum(item.error is not None for item in results),
        "aggregate": aggregate,
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def read_trial_results(path: Path) -> list[TrialResult]:
    if not path.exists():
        return []
    results = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(TrialResult.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid evaluation JSONL at {path}:{line_number}: {exc}") from exc
    return results
