from __future__ import annotations

from longrun_agent.context_probes.schemas import ProbeCase, ProbeCaseResult, ProbePrediction, ProbeScore


def evaluate_prediction(case: ProbeCase, prediction: ProbePrediction) -> ProbeCaseResult:
    errors: list[str] = []
    if prediction.provider_error:
        errors.append(f"provider_error: {prediction.provider_error}")
    if prediction.protocol_error:
        errors.append(f"protocol_error: {prediction.protocol_error}")
    if not prediction.protocol_valid or prediction.predicted is None:
        errors.append("protocol_valid=false")
        return ProbeCaseResult(case=case, prediction=prediction, exact_match=False, metrics=_zero_metrics(case), errors=errors)
    if case.probe == "position":
        metrics = _constraint_metrics(case.expected.get("constraints", []), prediction.predicted.get("constraints", []))
        exact = metrics["constraint_recall"] == 1.0 and metrics["constraint_precision"] == 1.0
    elif case.probe == "multi_constraint":
        metrics = _multi_metrics(case.expected.get("constraints", []), prediction.predicted.get("constraints", []))
        exact = metrics["exact_set_match"] == 1.0
    elif case.probe == "state_tracking":
        metrics = _state_metrics(case.expected, prediction.predicted)
        exact = metrics["exact_state_match"] == 1.0
    elif case.probe == "aggregation":
        metrics = _aggregation_metrics(case.expected, prediction.predicted)
        exact = metrics["aggregation_exact_match"] == 1.0
    else:
        raise ValueError(f"unknown probe: {case.probe}")
    return ProbeCaseResult(case=case, prediction=prediction, exact_match=exact, metrics=metrics, errors=errors)


def evaluate_oracle_for_test_only(case: ProbeCase) -> ProbeCaseResult:
    """Test-only oracle evaluator. Production runner must never call this helper."""

    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted=dict(case.expected),
        protocol_valid=True,
    )
    return evaluate_prediction(case, prediction)


def score_results(results: list[ProbeCaseResult]) -> ProbeScore:
    if not results:
        return ProbeScore(total=0, exact=0, recall=0.0, precision=0.0)
    exact = sum(1 for result in results if result.exact_match)
    recall_values = [value for result in results for key, value in result.metrics.items() if key in {"constraint_recall", "micro_recall"}]
    precision_values = [
        value for result in results for key, value in result.metrics.items() if key in {"constraint_precision", "micro_precision"}
    ]
    stale = [result.metrics.get("stale_result_rejection", 0.0) for result in results if result.case.probe == "aggregation"]
    return ProbeScore(
        total=len(results),
        exact=exact,
        recall=sum(recall_values) / len(recall_values) if recall_values else exact / len(results),
        precision=sum(precision_values) / len(precision_values) if precision_values else exact / len(results),
        stale_rejection=sum(stale) / len(stale) if stale else 0.0,
    )


def _constraint_metrics(expected_items, predicted_items) -> dict[str, float]:
    expected = _normalized_set(expected_items)
    predicted = _normalized_set(predicted_items)
    correct = len(expected & predicted)
    omitted = len(expected - predicted)
    hallucinated = len(predicted - expected)
    return {
        "exact_match": 1.0 if expected == predicted else 0.0,
        "constraint_recall": correct / len(expected) if expected else 1.0,
        "constraint_precision": correct / len(predicted) if predicted else (1.0 if not expected else 0.0),
        "hallucinated_constraint_count": float(hallucinated),
        "omitted_constraint_count": float(omitted),
    }


def _multi_metrics(expected_items, predicted_items) -> dict[str, float]:
    base = _constraint_metrics(expected_items, predicted_items)
    recall = base["constraint_recall"]
    precision = base["constraint_precision"]
    return {
        "exact_set_match": base["exact_match"],
        "micro_recall": recall,
        "micro_precision": precision,
        "micro_f1": (2 * recall * precision / (recall + precision)) if recall + precision else 0.0,
        "omitted_constraints": base["omitted_constraint_count"],
        "hallucinated_constraints": base["hallucinated_constraint_count"],
    }


def _state_metrics(expected: dict, predicted: dict) -> dict[str, float]:
    keys = ["plan_version", "file_hash", "active_task", "verification_state", "latest_valid_test"]
    field_scores = {key: 1.0 if predicted.get(key) == expected.get(key) else 0.0 for key in keys}
    stale_label = expected.get("stale_test_label")
    stale_used = 1.0 if stale_label and predicted.get("latest_valid_test") == stale_label else 0.0
    return {
        "exact_state_match": 1.0 if all(value == 1.0 for value in field_scores.values()) else 0.0,
        "plan_version_accuracy": field_scores["plan_version"],
        "file_hash_accuracy": field_scores["file_hash"],
        "active_task_accuracy": field_scores["active_task"],
        "test_state_accuracy": min(field_scores["verification_state"], field_scores["latest_valid_test"]),
        "stale_state_usage_rate": stale_used,
    }


def _aggregation_metrics(expected: dict, predicted: dict) -> dict[str, float]:
    current_test = 1.0 if predicted.get("current_test") == expected.get("current_test") else 0.0
    expected_failures = _normalized_set(expected.get("unresolved_failures", []))
    predicted_failures = _normalized_set(predicted.get("unresolved_failures", []))
    expected_files = _normalized_set(expected.get("modified_files", []))
    predicted_files = _normalized_set(predicted.get("modified_files", []))
    failure_recall = len(expected_failures & predicted_failures) / len(expected_failures) if expected_failures else 1.0
    file_recall = len(expected_files & predicted_files) / len(expected_files) if expected_files else 1.0
    stale_rejection = 1.0 if predicted.get("stale_results_rejected") == expected.get("stale_results_rejected") else 0.0
    epoch = 1.0 if predicted.get("code_epoch") == expected.get("code_epoch") else 0.0
    exact = current_test == failure_recall == stale_rejection == file_recall == epoch == 1.0
    return {
        "current_test_exact": current_test,
        "current_failure_recall": failure_recall,
        "stale_result_rejection": stale_rejection,
        "modified_file_recall": file_recall,
        "code_epoch_accuracy": epoch,
        "aggregation_exact_match": 1.0 if exact else 0.0,
    }


def _zero_metrics(case: ProbeCase) -> dict[str, float]:
    if case.probe == "position":
        return {
            "exact_match": 0.0,
            "constraint_recall": 0.0,
            "constraint_precision": 0.0,
            "hallucinated_constraint_count": 0.0,
            "omitted_constraint_count": float(len(case.expected.get("constraints", []))),
        }
    if case.probe == "multi_constraint":
        return {
            "exact_set_match": 0.0,
            "micro_recall": 0.0,
            "micro_precision": 0.0,
            "micro_f1": 0.0,
            "omitted_constraints": float(len(case.expected.get("constraints", []))),
            "hallucinated_constraints": 0.0,
        }
    if case.probe == "state_tracking":
        return {
            "exact_state_match": 0.0,
            "plan_version_accuracy": 0.0,
            "file_hash_accuracy": 0.0,
            "active_task_accuracy": 0.0,
            "test_state_accuracy": 0.0,
            "stale_state_usage_rate": 0.0,
        }
    return {
        "current_test_exact": 0.0,
        "current_failure_recall": 0.0,
        "stale_result_rejection": 0.0,
        "modified_file_recall": 0.0,
        "code_epoch_accuracy": 0.0,
        "aggregation_exact_match": 0.0,
    }


def _normalized_set(values) -> set[str]:
    return {str(value).strip() for value in values or [] if str(value).strip()}
