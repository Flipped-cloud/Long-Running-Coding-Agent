from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.evaluation.coordinator import EvaluationCoordinator
from longrun_agent.evaluation.reporting import read_trial_attempts, read_trial_results
from longrun_agent.evaluation.schema import (
    AdapterVerificationResult,
    AgentConfigReference,
    EvaluationManifest,
    EvaluationOutcome,
    EvaluationTaskCase,
    ProgressSnapshot,
    TerminationReason,
    TrialStatus,
)
from longrun_agent.evaluation.tool_calls import summarize_tool_calls


class FakeAdapter:
    def __init__(self):
        self.runs = 0

    def prepare(self, case, descriptor):
        descriptor.trial_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, case, descriptor):
        return None

    def objective(self, case, descriptor):
        return "objective"

    def workspace(self, case, descriptor):
        return descriptor.trial_dir / "workspace"

    def verification_contract(self, case, descriptor):
        return None

    def run_agent(self, case, config_path, seed, descriptor):
        self.runs += 1
        if case.case_id == "error":
            raise RuntimeError("injected trial failure")
        return {"project_id": descriptor.trial_id}

    def verify(self, case, outcome, descriptor):
        return AdapterVerificationResult(
            oracle_report_id="oracle-report",
            oracle_verdict="verified",
            oracle_f2p_rate=1,
            oracle_p2p_rate=1,
            oracle_integrity_passed=True,
            oracle_partial_resolution=False,
            oracle_required_checks_passed=2,
            oracle_required_checks_failed=0,
            oracle_total_requirements=2,
            oracle_verified_requirements=2,
            oracle_verifier_seconds=0.1,
            oracle_contract_id="contract",
            oracle_contract_hash="hash",
            oracle_baseline_fingerprint="baseline",
            oracle_candidate_fingerprint="candidate",
            sanitized_summary="Oracle verified all required categories.",
        )

    def collect_artifacts(self, case, outcome, verification, descriptor):
        return EvaluationOutcome(
            evaluation_id=descriptor.evaluation_id,
            case_id=case.case_id,
            config_id=descriptor.config_id,
            trial_id=descriptor.trial_id,
            seed=descriptor.seed,
            project_id=outcome["project_id"],
            project_status="verified",
            verification_verdict="verified",
            oracle_verification_verdict=verification.oracle_verdict,
            oracle_verification_report_id=verification.oracle_report_id,
            oracle_total_requirements=verification.oracle_total_requirements,
            oracle_verified_requirements=verification.oracle_verified_requirements,
            termination_reason=TerminationReason.COMPLETED,
            full_resolution=True,
            f2p_rate=1,
            p2p_rate=1,
            integrity_passed=True,
            progress_snapshots=[ProgressSnapshot(score=1, source_report_id="report")],
        )

    def cleanup(self, case, descriptor):
        return None


def test_coordinator_isolates_resumes_and_continues_after_trial_error(tmp_path: Path) -> None:
    manifest = EvaluationManifest(
        evaluation_id="eval",
        task_cases=[EvaluationTaskCase(case_id="ok"), EvaluationTaskCase(case_id="error")],
        agent_configs=[AgentConfigReference(config_id="config", path=tmp_path / "config.yaml")],
        output_root=tmp_path,
    )
    adapter = FakeAdapter()
    coordinator = EvaluationCoordinator(manifest, {"local_project": adapter})
    report = coordinator.run()
    assert report["completed_count"] == 1
    assert report["error_count"] == 1
    assert adapter.runs == 2
    assert len({item[2].trial_dir for item in coordinator.expand_trials()}) == 2
    results = read_trial_results(coordinator.results_path)
    completed = next(item for item in results if item.outcome is not None)
    assert completed.metadata["mode"] == "default"
    assert completed.metadata["context_mode"] == "unknown"
    assert completed.metadata["evaluation_semantics_version"] == "v0.5.5-sandbox-attribution"
    events = [json.loads(line) for line in coordinator.events_path.read_text(encoding="utf-8").splitlines()]
    event_types = {event["event_type"] for event in events}
    assert {"progress_snapshot_created", "trajectory_features_extracted"} <= event_types
    assert "failure_attribution_created" not in event_types
    required_fields = {
        "project_id",
        "task_id",
        "session_id",
        "contract_id",
        "contract_hash",
        "report_id",
        "check_id",
        "trial_id",
        "case_id",
        "timestamp",
        "verdict",
        "sanitized_reason",
        "evidence_ids",
        "artifact_paths",
    }
    assert all(required_fields <= event.keys() for event in events)

    coordinator.run()
    assert adapter.runs == 3
    assert len(read_trial_results(coordinator.results_path)) == 2
    assert len(coordinator.results_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(read_trial_attempts(coordinator.attempts_path)) == 3


class RecoveringAdapter(FakeAdapter):
    def run_agent(self, case, config_path, seed, descriptor):
        self.runs += 1
        if self.runs == 1:
            raise RuntimeError("first attempt fails")
        return {"project_id": descriptor.trial_id}


def test_coordinator_retries_error_with_upsert_and_skips_completed_resume(tmp_path: Path) -> None:
    manifest = EvaluationManifest(
        evaluation_id="resume",
        task_cases=[EvaluationTaskCase(case_id="recover")],
        agent_configs=[AgentConfigReference(config_id="config", path=tmp_path / "config.yaml")],
        output_root=tmp_path,
    )
    adapter = RecoveringAdapter()
    coordinator = EvaluationCoordinator(manifest, {"local_project": adapter})

    first = coordinator.run()
    assert first["trial_count"] == 1
    assert first["error_count"] == 1

    second = coordinator.run()
    assert second["trial_count"] == 1
    assert second["completed_count"] == 1
    assert second["error_count"] == 0
    assert adapter.runs == 2
    assert len(coordinator.results_path.read_text(encoding="utf-8").splitlines()) == 1
    assert read_trial_results(coordinator.results_path)[0].descriptor.status == TrialStatus.COMPLETED
    attempts = read_trial_attempts(coordinator.attempts_path)
    assert len(attempts) == 2
    assert attempts[1].retry_reason == "retry_after_error"

    coordinator.run()
    assert adapter.runs == 2
    assert len(coordinator.results_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(read_trial_attempts(coordinator.attempts_path)) == 2


def test_tool_call_summary_uses_only_unique_tool_started_events(tmp_path: Path) -> None:
    events_path = tmp_path / "telemetry" / "run" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    started = {
        "run_id": "run",
        "step": 3,
        "event_type": "tool_started",
        "tool_call_id": "call-1",
        "tool_name": "read_file",
    }
    finished = {
        **started,
        "event_type": "tool_finished",
        "success": False,
        "error_type": "tool_error",
        "duration_seconds": 0.25,
    }
    events_path.write_text(
        "\n".join(json.dumps(item) for item in [started, started, finished]) + "\n",
        encoding="utf-8",
    )

    calls = summarize_tool_calls([events_path])

    assert calls == [
        {
            "run_id": "run",
            "step": 3,
            "tool_call_id": "call-1",
            "tool_name": "read_file",
            "success": False,
            "error_type": "tool_error",
            "duration": 0.25,
        }
    ]
    assert not (tmp_path / "telemetry" / "run" / "prompts").exists()
