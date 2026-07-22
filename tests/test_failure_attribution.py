from __future__ import annotations

from longrun_agent.evaluation.attribution import FailureAttributor
from longrun_agent.evaluation.failure_taxonomy import FAILURE_LAYERS, FailureCode, termination_reason_from_status
from longrun_agent.evaluation.schema import FailureLayer, TerminationReason, TrajectoryFeatures


def test_completion_false_positive_and_infrastructure_rules() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.COMPLETED,
        features=TrajectoryFeatures(
            first_completion_request="e1",
            first_failed_required_check="e2",
            first_observable_symptom="e2",
            first_causal_divergence="e1",
        ),
        events=[],
    )
    assert attribution.primary_code == FailureCode.COMPLETION_PREMATURE_REQUEST
    assert FailureCode.VERIFICATION_FALSE_POSITIVE in attribution.secondary_codes
    assert attribution.evidence_event_ids == ["e2", "e1"]

    infra = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.ENVIRONMENT_ERROR,
        features=TrajectoryFeatures(),
        events=[{"event_type": "verification_infrastructure_error"}],
    )
    assert infra.primary_code == FailureCode.VERIFICATION_INFRA_FAILURE

    mismatch = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.VERIFICATION_INCONCLUSIVE,
        features=TrajectoryFeatures(),
        events=[{"event_type": "verification_baseline_mismatch"}],
    )
    assert mismatch.primary_code == FailureCode.VERIFICATION_BASELINE_MISMATCH


def test_agentbench_style_termination_mapping() -> None:
    assert termination_reason_from_status("context_budget_exhausted") == TerminationReason.CONTEXT_LIMIT
    assert termination_reason_from_status("provider_error") == TerminationReason.PROVIDER_ERROR
    assert termination_reason_from_status("session_limit_reached") == TerminationReason.SESSION_LIMIT
    assert termination_reason_from_status("max_steps_reached") == TerminationReason.TASK_LIMIT
    assert termination_reason_from_status("step_limit") == TerminationReason.TASK_LIMIT
    assert termination_reason_from_status("max_session_seconds") == TerminationReason.TIME_LIMIT
    assert termination_reason_from_status("max_project_sessions") == TerminationReason.SESSION_LIMIT


def test_generated_test_budget_failure_has_specific_completion_control_code() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.TASK_LIMIT,
        features=TrajectoryFeatures(),
        events=[
            {
                "event_type": "generated_test_requirement_unmet",
                "payload": {
                    "registered_candidates": 0,
                    "valid_candidates": 0,
                    "registration_attempts": 0,
                },
            }
        ],
    )

    assert attribution.primary_code == FailureCode.GENERATED_TEST_NOT_ATTEMPTED
    assert attribution.primary_layer == FailureLayer.COMPLETION_CONTROL


def test_invalid_generated_candidate_has_verification_failure_code() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.TASK_LIMIT,
        features=TrajectoryFeatures(),
        events=[
            {
                "event_type": "generated_test_requirement_unmet",
                "payload": {
                    "registered_candidates": 1,
                    "valid_candidates": 0,
                    "registration_attempts": 1,
                },
            }
        ],
    )

    assert attribution.primary_code == FailureCode.GENERATED_TEST_INVALID_CANDIDATE
    assert attribution.primary_layer == FailureLayer.VERIFICATION


def test_completion_confirmed_by_later_formal_verification_is_not_false_positive() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.COMPLETED,
        features=TrajectoryFeatures(first_completion_request="completion"),
        events=[
            {"event_type": "verification_report_created", "verdict": "verified"},
            {"event_type": "oracle_verification_finished", "oracle_verdict": "verified", "integrity_passed": True},
        ],
    )
    assert attribution is None


def test_oracle_failure_drives_false_completion_and_regression_attribution() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.COMPLETED,
        features=TrajectoryFeatures(first_completion_request="completion"),
        events=[
            {
                "event_type": "oracle_verification_finished",
                "oracle_verdict": "reopened",
                "f2p_rate": 1.0,
                "p2p_rate": 0.0,
                "integrity_passed": True,
            }
        ],
    )

    assert attribution.primary_code == FailureCode.COMPLETION_PREMATURE_REQUEST
    assert FailureCode.VERIFICATION_FALSE_POSITIVE in attribution.secondary_codes
    assert FailureCode.IMPLEMENTATION_REGRESSION in attribution.secondary_codes


def test_tool_argument_failure_codes_use_tool_aci_layer() -> None:
    assert FAILURE_LAYERS[FailureCode.TOOL_ARGUMENT_NORMALIZED] == FailureLayer.TOOL_ACI
    assert FAILURE_LAYERS[FailureCode.TOOL_INVALID_ARGUMENT] == FailureLayer.TOOL_ACI
    assert FAILURE_LAYERS[FailureCode.TOOL_INTERNAL_ERROR] == FailureLayer.TOOL_ACI


def test_runtime_verified_oracle_reopened_has_explicit_disagreement_code() -> None:
    attribution = FailureAttributor().attribute(
        case_id="case",
        trial_id="trial",
        termination_reason=TerminationReason.COMPLETED,
        features=TrajectoryFeatures(),
        events=[
            {"event_type": "verification_report_created", "verdict": "verified"},
            {"event_type": "oracle_verification_finished", "oracle_verdict": "reopened", "integrity_passed": True},
        ],
    )
    assert attribution.primary_code == FailureCode.VERIFICATION_FALSE_POSITIVE
    assert FailureCode.RUNTIME_ORACLE_DISAGREEMENT in attribution.secondary_codes


def test_integrity_and_budget_failures_have_explicit_codes() -> None:
    integrity = FailureAttributor().attribute(
        case_id="case",
        trial_id="integrity",
        termination_reason=TerminationReason.COMPLETED,
        features=TrajectoryFeatures(),
        events=[{"event_type": "oracle_verification_finished", "oracle_verdict": "reopened", "integrity_passed": False}],
    )
    time_limit = FailureAttributor().attribute(
        case_id="case",
        trial_id="time",
        termination_reason=TerminationReason.TIME_LIMIT,
        features=TrajectoryFeatures(),
        events=[],
    )
    step_limit = FailureAttributor().attribute(
        case_id="case",
        trial_id="step",
        termination_reason=TerminationReason.TASK_LIMIT,
        features=TrajectoryFeatures(first_completion_request="completion"),
        events=[],
    )
    assert integrity.primary_code == FailureCode.VERIFICATION_TEST_TAMPERING
    assert time_limit.primary_code == FailureCode.EXECUTION_TIME_LIMIT
    assert step_limit.primary_code == FailureCode.EXECUTION_STEP_LIMIT
