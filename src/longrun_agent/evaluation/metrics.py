from __future__ import annotations

from longrun_agent.evaluation.progress import progress_metrics
from longrun_agent.evaluation.schema import EvaluationOutcome


def trial_metrics(outcome: EvaluationOutcome) -> dict[str, float | int | bool]:
    oracle_verified = outcome.oracle_verification_verdict == "verified"
    verified_completion_count = max(0, outcome.completion_requests - outcome.false_completion_requests) if oracle_verified else 0
    completion_precision = verified_completion_count / outcome.completion_requests if outcome.completion_requests else 0.0
    false_completion_count = max(outcome.false_completion_requests, outcome.completion_requests - verified_completion_count)
    progress = progress_metrics(outcome.progress_snapshots)
    verified_requirements = outcome.oracle_verified_requirements
    total_requirements = outcome.oracle_total_requirements
    runtime_oracle_disagreement = outcome.runtime_verification_verdict == "verified" and outcome.oracle_verification_verdict != "verified"
    return {
        "verified": oracle_verified,
        "full_resolution": outcome.full_resolution,
        "partial_resolution": outcome.partial_resolution,
        "f2p_rate": outcome.f2p_rate,
        "p2p_rate": outcome.p2p_rate,
        "integrity_passed": outcome.integrity_passed,
        "completion_request_count": outcome.completion_requests,
        "verified_completion_count": verified_completion_count,
        "false_completion_count": false_completion_count,
        "completion_precision": completion_precision,
        "reopen_recovery_rate": outcome.task_verified_count / outcome.task_reopened_count if outcome.task_reopened_count else 0.0,
        "test_candidates": outcome.test_candidates,
        "well_formed_rate": outcome.well_formed_test_candidates / outcome.test_candidates if outcome.test_candidates else 0.0,
        "f2p_test_rate": outcome.f2p_tests / outcome.test_candidates if outcome.test_candidates else 0.0,
        "p2p_irrelevant_rate": outcome.p2p_irrelevant_tests / outcome.test_candidates if outcome.test_candidates else 0.0,
        "harmful_test_rate": outcome.harmful_tests / outcome.test_candidates if outcome.test_candidates else 0.0,
        **progress,
        "verified_requirements": verified_requirements,
        "total_requirements": total_requirements,
        "wall_clock": outcome.wall_clock_seconds,
        "tokens": outcome.input_tokens + outcome.output_tokens + outcome.compactor_tokens,
        "tool_calls": outcome.tool_calls,
        "sessions": outcome.sessions,
        "verification_seconds": outcome.verifier_seconds,
        "oracle_verifier_seconds": outcome.oracle_verifier_seconds,
        "runtime_oracle_disagreement": runtime_oracle_disagreement,
        "no_runtime_verifier": outcome.runtime_verification_verdict is None,
        "cost_per_verified_requirement": (
            (outcome.input_tokens + outcome.output_tokens + outcome.compactor_tokens) / verified_requirements
            if verified_requirements
            else 0.0
        ),
    }
