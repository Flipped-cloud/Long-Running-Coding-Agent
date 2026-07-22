from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.evaluation.rebuild import rebuild_evaluation_report
from longrun_agent.evaluation.reporting import read_trial_results, write_evaluation_report, write_trial_results_atomic
from longrun_agent.evaluation.schema import (
    EvaluationOutcome,
    TerminationReason,
    TrialDescriptor,
    TrialResult,
    TrialStatus,
)


def _result(root: Path, trial_id: str, *, verified: bool) -> TrialResult:
    trial_dir = root / "trials" / trial_id
    descriptor = TrialDescriptor(
        evaluation_id="evaluation",
        case_id="case",
        config_id="config",
        trial_id=trial_id,
        trial_number=1,
        seed=0,
        trial_dir=trial_dir,
        status=TrialStatus.COMPLETED,
    )
    outcome = EvaluationOutcome(
        evaluation_id="evaluation",
        case_id="case",
        config_id="config",
        trial_id=trial_id,
        seed=0,
        project_id=trial_id,
        project_status="verified" if verified else "active",
        runtime_verification_verdict="verified" if verified else "reopened",
        oracle_verification_verdict="verified" if verified else "reopened",
        oracle_verification_report_id=f"oracle-{trial_id}",
        oracle_total_requirements=2,
        oracle_verified_requirements=2 if verified else 1,
        termination_reason=TerminationReason.UNKNOWN,
        full_resolution=verified,
        f2p_rate=1.0 if verified else 0.5,
        p2p_rate=1.0,
        integrity_passed=True,
        input_tokens=101,
        output_tokens=23,
        wall_clock_seconds=12.5,
        tool_calls=7,
    )
    event_path = trial_dir / "oracle" / "public_events.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        json.dumps(
            {
                "event_type": "oracle_verification_finished",
                "event_id": f"event-{trial_id}",
                "oracle_verdict": outcome.oracle_verification_verdict,
                "integrity_passed": True,
                "required_checks_failed": 0 if verified else 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sessions = trial_dir / "state" / trial_id / "sessions.jsonl"
    sessions.parent.mkdir(parents=True)
    sessions.write_text(
        json.dumps({"run_status": "completed" if verified else "max_steps_reached"}) + "\n",
        encoding="utf-8",
    )
    return TrialResult(descriptor=descriptor, outcome=outcome)


def test_rebuild_is_offline_and_preserves_oracle_and_core_metrics(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "evaluation"
    results = [_result(root, "success", verified=True), _result(root, "failure", verified=False)]
    results[1].descriptor.trial_dir = Path("/mnt/d/stale/failure")
    write_trial_results_atomic(root / "trials.jsonl", results)
    write_evaluation_report(root, results)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline rebuild must not invoke provider or oracle")

    monkeypatch.setattr("longrun_agent.evaluation.oracle.OfflineOracleEvaluator.evaluate", forbidden)
    before = {
        item.descriptor.trial_id: (
            item.outcome.oracle_verification_verdict,
            item.outcome.input_tokens,
            item.outcome.output_tokens,
            item.outcome.wall_clock_seconds,
            item.outcome.tool_calls,
            item.outcome.f2p_rate,
            item.outcome.p2p_rate,
        )
        for item in read_trial_results(root / "trials.jsonl")
    }

    report = rebuild_evaluation_report(root)
    rebuilt = read_trial_results(root / "trials.jsonl")
    after = {
        item.descriptor.trial_id: (
            item.outcome.oracle_verification_verdict,
            item.outcome.input_tokens,
            item.outcome.output_tokens,
            item.outcome.wall_clock_seconds,
            item.outcome.tool_calls,
            item.outcome.f2p_rate,
            item.outcome.p2p_rate,
        )
        for item in rebuilt
    }

    assert after == before
    assert next(item for item in rebuilt if item.descriptor.trial_id == "success").attribution is None
    failure = next(item for item in rebuilt if item.descriptor.trial_id == "failure")
    assert failure.outcome.termination_reason == TerminationReason.TASK_LIMIT
    assert failure.attribution.primary_code == "EXECUTION_STEP_LIMIT"
    assert report["trial_count"] == 2
    backup = Path(report["backup_path"])
    assert (backup / "trials.jsonl").exists()
    assert (backup / "report.json").exists()
