from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.coordinator import EvaluationCoordinator, _collect_events
from longrun_agent.evaluation.metrics import trial_metrics
from longrun_agent.evaluation.oracle import OfflineOracleEvaluator
from longrun_agent.evaluation.schema import (
    AgentConfigReference,
    EvaluationManifest,
    EvaluationOutcome,
    EvaluationTaskCase,
    TrialDescriptor,
)
from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.contract import load_contract, split_contract
from longrun_agent.verification.schema import (
    VerificationPurpose,
    VerificationReport,
    VerificationSummary,
    VerificationVerdict,
)
from longrun_agent.verification.store import VerificationStore


def test_disabled_condition_receives_oracle_verdict(tmp_path: Path) -> None:
    adapter, case, descriptor, project_id = _adapter_trial(tmp_path, "disabled")
    verification = adapter.verify(case, SimpleNamespace(project_id=project_id), descriptor)
    outcome = adapter.collect_artifacts(case, SimpleNamespace(project_id=project_id), verification, descriptor)

    assert outcome.runtime_verification_verdict is None
    assert outcome.oracle_verification_verdict == "reopened"
    assert outcome.verification_verdict == "reopened"
    assert not outcome.full_resolution
    assert outcome.f2p_rate < 1
    assert outcome.oracle_total_requirements > 0


def test_legacy_condition_uses_oracle_final_metrics(tmp_path: Path) -> None:
    adapter, case, descriptor, project_id = _adapter_trial(tmp_path, "legacy")
    runtime = VerificationReport(
        project_id=project_id,
        contract_id="legacy",
        contract_hash="runtime-hash",
        verdict=VerificationVerdict.VERIFIED,
        summary=VerificationSummary(required_checks_passed=1, integrity_passed=True),
    )
    VerificationStore(
        descriptor.trial_dir / "state",
        project_id,
        workspace_root=adapter.workspace(case, descriptor),
    ).save_report(runtime)
    verification = adapter.verify(case, SimpleNamespace(project_id=project_id), descriptor)
    outcome = adapter.collect_artifacts(case, SimpleNamespace(project_id=project_id), verification, descriptor)
    metrics = trial_metrics(outcome)

    assert outcome.runtime_verification_verdict == "verified"
    assert outcome.oracle_verification_verdict == "reopened"
    assert not outcome.full_resolution
    assert metrics["runtime_oracle_disagreement"] is True
    assert metrics["false_completion_count"] == 1


def test_contract_condition_separates_runtime_and_oracle_reports(tmp_path: Path) -> None:
    adapter, case, descriptor, project_id = _adapter_trial(tmp_path, "contract")
    runtime_store = VerificationStore(
        descriptor.trial_dir / "state",
        project_id,
        workspace_root=adapter.workspace(case, descriptor),
    )
    runtime = VerificationReport(
        project_id=project_id,
        contract_id="runtime",
        contract_hash="runtime-hash",
        verdict=VerificationVerdict.REOPENED,
    )
    runtime_store.save_report(runtime)
    verification = adapter.verify(case, SimpleNamespace(project_id=project_id), descriptor)
    oracle_report = VerificationReport.model_validate_json(Path(verification.oracle_report_private_path or "").read_text(encoding="utf-8"))

    assert runtime.purpose == VerificationPurpose.RUNTIME
    assert oracle_report.purpose == VerificationPurpose.ORACLE
    assert runtime.report_id != oracle_report.report_id
    assert runtime_store.list_reports(purpose=VerificationPurpose.ORACLE) == []
    assert oracle_report.report_id not in {item.report_id for item in runtime_store.list_reports()}


def test_same_final_workspace_has_same_oracle_result_across_modes(tmp_path: Path) -> None:
    rows = []
    for mode in ("disabled", "legacy", "contract", "contract_generated"):
        workspace, case, descriptor = _oracle_trial(tmp_path / mode, mode)
        OfflineOracleEvaluator().prepare_baseline(case=case, descriptor=descriptor, workspace=workspace)
        (workspace / "app.py").write_text("VALUE = 1\nREGRESSION = True\n", encoding="utf-8")
        rows.append(
            OfflineOracleEvaluator().evaluate(
                case=case,
                descriptor=descriptor,
                project_id=f"project-{mode}",
                final_workspace=workspace,
            )
        )

    assert len({row.oracle_contract_hash for row in rows}) == 1
    assert (
        len(
            {
                (
                    row.oracle_f2p_rate,
                    row.oracle_p2p_rate,
                    row.oracle_integrity_passed,
                    row.oracle_verdict,
                    row.oracle_total_requirements,
                )
                for row in rows
            }
        )
        == 1
    )


def test_oracle_baseline_created_before_agent_run(tmp_path: Path) -> None:
    workspace, case, descriptor = _oracle_trial(tmp_path, "baseline")
    evaluator = OfflineOracleEvaluator()
    evaluator.prepare_baseline(case=case, descriptor=descriptor, workspace=workspace)
    baseline = json.loads((descriptor.trial_dir / "oracle" / "baseline.json").read_text(encoding="utf-8"))
    (workspace / "app.py").write_text("VALUE = 1\nREGRESSION = True\n", encoding="utf-8")
    result = evaluator.evaluate(
        case=case,
        descriptor=descriptor,
        project_id="project-baseline",
        final_workspace=workspace,
    )

    assert result.oracle_baseline_fingerprint == baseline["baseline_fingerprint"]
    assert result.oracle_candidate_fingerprint != result.oracle_baseline_fingerprint


def test_oracle_does_not_call_model_provider(tmp_path: Path) -> None:
    adapter, case, descriptor, project_id = _adapter_trial(tmp_path, "no-provider")
    calls = 0

    def provider_factory(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("offline oracle must not create a provider")

    adapter.provider_factory = provider_factory
    adapter.verify(case, SimpleNamespace(project_id=project_id), descriptor)

    assert calls == 0


def test_oracle_does_not_modify_project_state(tmp_path: Path) -> None:
    workspace, case, descriptor = _oracle_trial(tmp_path, "readonly")
    evaluator = OfflineOracleEvaluator()
    evaluator.prepare_baseline(case=case, descriptor=descriptor, workspace=workspace)
    state_root = descriptor.trial_dir / "state"
    state_store = ProjectStateStore(state_root, workspace_root=workspace)
    state_store.create(ProjectState(project_id="project-readonly", objective="fix"))
    before = state_store.state_path("project-readonly").read_bytes()
    evaluator.evaluate(
        case=case,
        descriptor=descriptor,
        project_id="project-readonly",
        final_workspace=workspace,
    )

    assert state_store.state_path("project-readonly").read_bytes() == before


def test_oracle_result_is_not_exposed_to_agent(tmp_path: Path) -> None:
    adapter, case, descriptor, project_id = _adapter_trial(tmp_path, "not-exposed")
    state_path = ProjectStateStore(
        descriptor.trial_dir / "state",
        workspace_root=adapter.workspace(case, descriptor),
    ).state_path(project_id)
    before = state_path.read_bytes()

    adapter.verify(case, SimpleNamespace(project_id=project_id), descriptor)

    assert state_path.read_bytes() == before
    agent_visible = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in (
            descriptor.trial_dir / "workspace",
            descriptor.trial_dir / "state",
            descriptor.trial_dir / "telemetry",
            descriptor.trial_dir / "knowledge",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "verification" not in path.parts
    )
    assert "oracle_verification_finished" not in agent_visible
    assert "Oracle evaluation finished" not in agent_visible


def test_hidden_oracle_events_are_not_collected(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    (trial / "oracle").mkdir(parents=True)
    (trial / "workspace").mkdir()
    marker = "HIDDEN_MARKER_NEVER_PUBLIC"
    (trial / "oracle" / "private_events.jsonl").write_text(
        json.dumps({"event_type": "private", "secret": marker}) + "\n",
        encoding="utf-8",
    )
    (trial / "workspace" / "events.jsonl").write_text(
        json.dumps({"event_type": "workspace", "secret": marker}) + "\n",
        encoding="utf-8",
    )
    (trial / "oracle" / "public_events.jsonl").write_text(
        json.dumps({"event_type": "oracle_verification_finished", "oracle_verdict": "reopened"}) + "\n",
        encoding="utf-8",
    )

    events = _collect_events(trial)

    assert [item["event_type"] for item in events] == ["oracle_verification_finished"]
    assert marker not in json.dumps(events)


def test_generated_test_does_not_replace_oracle_contract(tmp_path: Path) -> None:
    workspace, case, descriptor = _oracle_trial(tmp_path, "generated")
    evaluator = OfflineOracleEvaluator()
    evaluator.prepare_baseline(case=case, descriptor=descriptor, workspace=workspace)
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_generated_candidate.py").write_text("def test_generated():\n    assert True\n", encoding="utf-8")
    result = evaluator.evaluate(
        case=case,
        descriptor=descriptor,
        project_id="project-generated",
        final_workspace=workspace,
    )

    assert result.oracle_verdict != "verified"
    assert result.oracle_f2p_rate < 1


def test_incompatible_evaluation_semantics_resume_rejected(tmp_path: Path) -> None:
    manifest = EvaluationManifest(
        evaluation_id="old",
        task_cases=[EvaluationTaskCase(case_id="case")],
        agent_configs=[AgentConfigReference(config_id="config", path=tmp_path / "config.yaml")],
        output_root=tmp_path,
    )
    coordinator = EvaluationCoordinator(manifest, {})
    coordinator.evaluation_dir.mkdir(parents=True)
    coordinator.results_path.write_text(
        json.dumps({"descriptor": {"status": "completed"}, "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Existing evaluation results use incompatible semantics"):
        coordinator.run()


def test_oracle_time_excluded_from_agent_cost() -> None:
    outcome = EvaluationOutcome(
        evaluation_id="eval",
        case_id="case",
        config_id="config",
        trial_id="trial",
        seed=0,
        project_id="project",
        project_status="candidate_complete",
        verification_verdict="verified",
        oracle_verification_verdict="verified",
        oracle_verification_report_id="oracle",
        oracle_total_requirements=1,
        oracle_verified_requirements=1,
        oracle_verifier_seconds=9,
        verifier_seconds=2,
        wall_clock_seconds=3,
        input_tokens=10,
        output_tokens=5,
    )
    metrics = trial_metrics(outcome)

    assert metrics["verification_seconds"] == 2
    assert metrics["oracle_verifier_seconds"] == 9
    assert metrics["wall_clock"] == 3
    assert metrics["cost_per_verified_requirement"] == 15


def _adapter_trial(
    tmp_path: Path,
    mode: str,
) -> tuple[LocalProjectAdapter, EvaluationTaskCase, TrialDescriptor, str]:
    workspace, case, descriptor = _oracle_trial(tmp_path, mode)
    fixture = tmp_path / "fixture"
    shutil.copytree(workspace, fixture)
    case.fixture = fixture
    adapter = LocalProjectAdapter(lambda _config, _case, _seed: None)
    adapter.prepare(case, descriptor)
    adapter.reset(case, descriptor)
    project_id = f"project-{mode}"
    state_store = ProjectStateStore(descriptor.trial_dir / "state", workspace_root=adapter.workspace(case, descriptor))
    state_store.create(
        ProjectState(
            project_id=project_id,
            objective="fix",
            status=ProjectStatus.CANDIDATE_COMPLETE,
            tasks=[
                TaskNode(
                    id="task",
                    key="task",
                    title="task",
                    objective="fix",
                    acceptance_criteria=["complete"],
                    status=TaskStatus.CANDIDATE_COMPLETE,
                )
            ],
        )
    )
    state_store.events_path(project_id).write_text(
        json.dumps({"event_type": "task_completion_requested"}) + "\n",
        encoding="utf-8",
    )
    return adapter, case, descriptor, project_id


def _oracle_trial(tmp_path: Path, mode: str) -> tuple[Path, EvaluationTaskCase, TrialDescriptor]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "workspace-source"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("Set VALUE to one.", encoding="utf-8")
    (workspace / "app.py").write_text("VALUE = 0\nREGRESSION = True\n", encoding="utf-8")
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "contract_id": "shared-oracle-contract",
                "project_id": "__PROJECT_ID__",
                "source": "fixture",
                "checks": [
                    {
                        "check_id": "resolution",
                        "title": "resolution",
                        "kind": "resolution",
                        "argv": [sys.executable, "-c", "import app; raise SystemExit(0 if app.VALUE == 1 else 1)"],
                    },
                    {
                        "check_id": "regression",
                        "title": "regression",
                        "kind": "regression",
                        "argv": [sys.executable, "-c", "import app; raise SystemExit(0 if app.REGRESSION else 1)"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    descriptor = TrialDescriptor(
        evaluation_id="eval",
        case_id="case",
        config_id=mode,
        trial_id=f"trial-{mode}",
        trial_number=1,
        seed=0,
        trial_dir=tmp_path / "trial",
    )
    case = EvaluationTaskCase(case_id="case", fixture=workspace, task_file=Path("TASK.md"), contract_path=contract)
    _, private = split_contract(load_contract(contract, workspace_root=workspace))
    private_path = descriptor.trial_dir / "oracle" / "private" / "contract.json"
    private_path.parent.mkdir(parents=True)
    private_path.write_text(private.model_dump_json(indent=2), encoding="utf-8")
    return workspace, case, descriptor
