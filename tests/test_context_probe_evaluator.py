from longrun_agent.context_probes.evaluator import evaluate_prediction
from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.context_probes.schemas import ProbePrediction


def test_wrong_constraint_prediction_gets_no_oracle_credit():
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=8)[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted={"constraints": ["wrong-constraint"]},
        protocol_valid=True,
    )

    result = evaluate_prediction(case, prediction)

    assert not result.exact_match
    assert result.metrics["constraint_recall"] == 0
    assert result.metrics["constraint_precision"] == 0


def test_partial_constraints_score_partial_recall_and_hallucination():
    case = generate_cases(probe="multi_constraint", lengths=[2048], samples=1, seed=9, constraints_per_case=[3])[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted={"constraints": [case.expected["constraints"][0], "hallucinated"]},
        protocol_valid=True,
    )

    result = evaluate_prediction(case, prediction)

    assert result.metrics["micro_recall"] == 1 / 3
    assert result.metrics["micro_precision"] == 1 / 2
    assert result.metrics["hallucinated_constraints"] == 1


def test_state_stale_usage_is_scored():
    case = generate_cases(probe="state_tracking", lengths=[2048], samples=1, seed=10)[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted={
            "plan_version": case.expected["plan_version"],
            "file_hash": case.expected["file_hash"],
            "active_task": case.expected["active_task"],
            "verification_state": "passed",
            "latest_valid_test": case.expected["stale_test_label"],
        },
        protocol_valid=True,
    )

    result = evaluate_prediction(case, prediction)

    assert result.metrics["exact_state_match"] == 0
    assert result.metrics["stale_state_usage_rate"] == 1


def test_provider_and_protocol_errors_are_failures():
    case = generate_cases(probe="aggregation", lengths=[2048], samples=1, seed=11)[0]
    prediction = ProbePrediction(case_id=case.case_id, probe=case.probe, mode="full_history", provider_error="boom")

    result = evaluate_prediction(case, prediction)

    assert not result.exact_match
    assert result.errors
