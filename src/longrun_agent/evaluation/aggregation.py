from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from longrun_agent.evaluation.metrics import trial_metrics
from longrun_agent.evaluation.schema import TrialResult, TrialStatus, latest_trial_results


def aggregate_results(results: list[TrialResult]) -> dict[str, Any]:
    results = latest_trial_results(results)
    completed = [item for item in results if item.descriptor.status == TrialStatus.COMPLETED and item.outcome is not None]
    groups: dict[str, dict[str, list[TrialResult]]] = {
        "task": defaultdict(list),
        "config": defaultdict(list),
        "mode": defaultdict(list),
        "trial": defaultdict(list),
        "overall": defaultdict(list),
    }
    for item in completed:
        groups["task"][item.descriptor.case_id].append(item)
        groups["config"][item.descriptor.config_id].append(item)
        groups["mode"][str(item.metadata.get("mode") or item.descriptor.config_id)].append(item)
        groups["trial"][item.descriptor.trial_id].append(item)
        groups["overall"]["all"].append(item)
    return {group_name: {key: _aggregate_group(items) for key, items in sorted(group.items())} for group_name, group in groups.items()}


def _aggregate_group(items: list[TrialResult]) -> dict[str, Any]:
    rows = [trial_metrics(item.outcome) for item in items if item.outcome is not None]

    numeric_keys = sorted(
        {key for row in rows for key, value in row.items() if value is not None and isinstance(value, (int, float, bool))}
    )

    metrics: dict[str, dict[str, float | int]] = {}

    for key in numeric_keys:
        values = [float(value) for row in rows if (value := row.get(key)) is not None and isinstance(value, (int, float, bool))]

        metrics[key] = {
            **distribution(values),
            "observed_count": len(values),
            "missing_count": len(rows) - len(values),
        }

    outcomes = [item.outcome for item in items if item.outcome is not None]
    attributions = [item.attribution for item in items if item.attribution is not None]
    runtime_verdicts = [outcome.runtime_verification_verdict for outcome in outcomes]
    oracle_verdicts = [outcome.oracle_verification_verdict for outcome in outcomes]
    disagreement_count = sum(
        runtime == "verified" and oracle != "verified" for runtime, oracle in zip(runtime_verdicts, oracle_verdicts, strict=True)
    )

    return {
        "count": len(items),
        "metrics": metrics,
        "success_at_k": success_at_k([bool(outcome.full_resolution) for outcome in outcomes]),
        "termination_reason_distribution": dict(Counter(outcome.termination_reason.value for outcome in outcomes)),
        "failure_layer_distribution": dict(Counter(item.primary_layer.value for item in attributions)),
        "failure_code_distribution": dict(Counter(item.primary_code for item in attributions)),
        "false_completion_failure_distribution": dict(
            Counter(item.primary_code for item in attributions if item.primary_code == "COMPLETION_PREMATURE_REQUEST")
        ),
        "reviewed_failure_count": sum(item.reviewer_label is not None for item in attributions),
        "runtime_verification_verdict_distribution": dict(Counter(value for value in runtime_verdicts if value is not None)),
        "oracle_verification_verdict_distribution": dict(Counter(oracle_verdicts)),
        "runtime_oracle_disagreement_count": disagreement_count,
        "runtime_oracle_disagreement_rate": disagreement_count / len(outcomes) if outcomes else 0.0,
        "no_runtime_verifier_count": sum(value is None for value in runtime_verdicts),
        "oracle_verifier_seconds": sum(outcome.oracle_verifier_seconds for outcome in outcomes),
        "oracle_total_requirements": sum(outcome.oracle_total_requirements for outcome in outcomes),
    }


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "standard_deviation": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def success_at_k(successes: list[bool], k: int | None = None) -> float:
    n = len(successes)
    if n == 0:
        return 0.0
    k = min(k or n, n)
    failures = n - sum(successes)
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n, k)
