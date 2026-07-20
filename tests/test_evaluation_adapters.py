from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.adapters.swebench_export import SWEbenchExportAdapter
from longrun_agent.evaluation.schema import AdapterVerificationResult, EvaluationTaskCase, TrialDescriptor
from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.schema import VerificationReport, VerificationSummary, VerificationVerdict
from longrun_agent.verification.store import VerificationStore


def test_local_adapter_prepares_fixture_and_collects_formal_artifacts(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "TASK.md").write_text("Fix the fixture.", encoding="utf-8")
    (fixture / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (fixture / ".runs").mkdir()
    (fixture / ".runs" / "private.txt").write_text("excluded", encoding="utf-8")
    case = EvaluationTaskCase(case_id="case", fixture=fixture, contract_path=tmp_path / "contract.yaml")
    descriptor = TrialDescriptor(
        evaluation_id="eval",
        case_id="case",
        config_id="config",
        trial_id="trial",
        trial_number=1,
        seed=7,
        trial_dir=tmp_path / "trial",
    )
    adapter = LocalProjectAdapter(lambda _config, _case, _seed: None)
    adapter.prepare(case, descriptor)

    assert adapter.objective(case, descriptor) == "Fix the fixture."
    assert adapter.verification_contract(case, descriptor) == case.contract_path
    assert not (adapter.workspace(case, descriptor) / ".runs").exists()

    state_root = descriptor.trial_dir / "state"
    state_store = ProjectStateStore(state_root, workspace_root=adapter.workspace(case, descriptor))
    task = TaskNode(
        id="task",
        key="task",
        title="task",
        objective="fix",
        acceptance_criteria=["verified"],
        status=TaskStatus.VERIFIED,
        reopen_count=1,
    )
    state_store.create(ProjectState(project_id="project", objective="fix", status=ProjectStatus.VERIFIED, tasks=[task]))
    state_store.append_session(
        "project",
        {
            "input_tokens_total": 10,
            "output_tokens_total": 4,
            "compactor_input_tokens": 2,
            "knowledge_tokens_injected": 3,
            "tool_call_count": 5,
            "context_reset_count": 1,
            "memories_referenced": 1,
            "skills_referenced": 1,
        },
    )
    state_store.events_path("project").write_text(
        json.dumps({"event_type": "task_completion_requested"}) + "\n" + json.dumps({"event_type": "task_reopened"}) + "\n",
        encoding="utf-8",
    )
    report = VerificationReport(
        project_id="project",
        contract_id="contract",
        contract_hash="hash",
        verdict=VerificationVerdict.VERIFIED,
        summary=VerificationSummary(
            resolution_total=1,
            resolution_passed=1,
            f2p_rate=1,
            regression_total=1,
            regression_passed=1,
            p2p_rate=1,
            integrity_passed=True,
        ),
    )
    VerificationStore(state_root, "project", workspace_root=adapter.workspace(case, descriptor)).save_report(report)

    verification = AdapterVerificationResult(
        runtime_report_id=report.report_id,
        runtime_verdict="verified",
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
        oracle_verifier_seconds=0.25,
        oracle_contract_id="oracle-contract",
        oracle_contract_hash="oracle-hash",
        oracle_baseline_fingerprint="baseline",
        oracle_candidate_fingerprint="candidate",
        sanitized_summary="Oracle verified all required categories.",
    )
    outcome = adapter.collect_artifacts(
        case,
        SimpleNamespace(project_id="project", verification_verdict=None),
        verification,
        descriptor,
    )

    assert outcome.full_resolution
    assert outcome.verification_verdict == "verified"
    assert outcome.termination_reason.value == "completed"
    assert outcome.completion_requests == outcome.false_completion_requests == 1
    assert outcome.task_verified_count == outcome.task_reopened_count == 1
    assert (outcome.input_tokens, outcome.output_tokens, outcome.compactor_tokens) == (10, 4, 2)
    assert (outcome.knowledge_tokens, outcome.tool_calls, outcome.context_resets) == (3, 5, 1)
    assert [item.source_report_id for item in outcome.progress_snapshots] == ["oracle-report"]
    assert outcome.runtime_verification_report_id == report.report_id
    assert outcome.oracle_verifier_seconds == 0.25
    assert adapter.cleanup(case, descriptor) is None


def test_swebench_adapter_exports_patch_and_parses_external_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "longrun_agent.evaluation.adapters.swebench_export.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="diff --git a/app.py b/app.py\n"),
    )
    adapter = SWEbenchExportAdapter()
    export_path = tmp_path / "exports" / "prediction.json"

    payload = adapter.export_patch(
        instance_id="instance-1",
        model_name_or_path="fake-model",
        workspace=tmp_path,
        output_path=export_path,
    )

    assert payload["model_patch"].startswith("diff --git")
    assert json.loads(export_path.read_text(encoding="utf-8")) == payload

    external_path = tmp_path / "report.json"
    external_path.write_text(
        json.dumps(
            {
                "instance_id": "instance-1",
                "resolved": True,
                "f2p_rate": 1,
                "p2p_rate": 0.75,
                "integrity_passed": True,
            }
        ),
        encoding="utf-8",
    )
    outcome = adapter.parse_external_report(
        external_path,
        evaluation_id="eval",
        config_id="config",
        trial_id="trial",
        seed=3,
    )

    assert outcome.case_id == "instance-1"
    assert outcome.full_resolution
    assert outcome.verification_verdict == "verified"
    assert outcome.p2p_rate == 0.75
