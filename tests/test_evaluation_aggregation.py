from __future__ import annotations

from longrun_agent.evaluation.aggregation import aggregate_results
from longrun_agent.evaluation.schema import (
    EvaluationOutcome,
    ProgressSnapshot,
    TerminationReason,
    TrialDescriptor,
    TrialResult,
    TrialStatus,
)


def _trial(
    *,
    trial_id: str,
    progress: float,
    with_first_progress: bool,
) -> TrialResult:
    snapshots = [
        ProgressSnapshot(
            timestamp="2026-01-01T00:00:00+00:00",
            score=0.0,
            source_report_id=f"{trial_id}-r1",
        )
    ]

    if with_first_progress:
        snapshots.append(
            ProgressSnapshot(
                timestamp="2026-01-01T00:00:10+00:00",
                score=progress,
                source_report_id=f"{trial_id}-r2",
            )
        )

    descriptor = TrialDescriptor(
        evaluation_id="eval-test",
        case_id="case-a",
        config_id="config-a",
        trial_id=trial_id,
        trial_number=1,
        seed=0,
        trial_dir=f"/tmp/{trial_id}",
        status=TrialStatus.COMPLETED,
    )

    outcome = EvaluationOutcome(
        evaluation_id="eval-test",
        case_id="case-a",
        config_id="config-a",
        trial_id=trial_id,
        seed=0,
        project_id=f"project-{trial_id}",
        project_status="verified",
        verification_verdict="verified",
        runtime_verification_verdict="verified",
        runtime_verification_report_id=f"{trial_id}-runtime",
        oracle_verification_verdict="verified",
        oracle_verification_report_id=f"{trial_id}-oracle",
        oracle_total_requirements=2,
        oracle_verified_requirements=2,
        termination_reason=TerminationReason.COMPLETED,
        full_resolution=True,
        f2p_rate=1.0,
        p2p_rate=1.0,
        integrity_passed=True,
        progress_snapshots=snapshots,
    )

    return TrialResult(
        descriptor=descriptor,
        outcome=outcome,
    )


def test_aggregate_optional_numeric_metric_ignores_none() -> None:
    results = [
        _trial(
            trial_id="trial-with-progress",
            progress=1.0,
            with_first_progress=True,
        ),
        _trial(
            trial_id="trial-without-progress",
            progress=0.0,
            with_first_progress=False,
        ),
    ]

    report = aggregate_results(results)

    metric = report["overall"]["all"]["metrics"]["time_to_first_progress"]

    assert metric["mean"] == 10.0
    assert metric["observed_count"] == 1
    assert metric["missing_count"] == 1


def test_aggregate_omits_metric_when_all_values_are_none() -> None:
    results = [
        _trial(
            trial_id="trial-a",
            progress=0.0,
            with_first_progress=False,
        ),
        _trial(
            trial_id="trial-b",
            progress=0.0,
            with_first_progress=False,
        ),
    ]

    report = aggregate_results(results)

    metrics = report["overall"]["all"]["metrics"]

    assert "time_to_first_progress" not in metrics
