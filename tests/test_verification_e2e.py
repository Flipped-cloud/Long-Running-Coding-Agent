from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from longrun_agent.config import load_config
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.store import VerificationStore


def test_premature_completion_reopens_then_verifies_without_hidden_leak(tmp_path: Path) -> None:
    fixture = Path("examples/verification_bench/partial_fix").resolve()
    workspace = tmp_path / "workspace"
    shutil.copytree(fixture, workspace)
    config_path = _config(tmp_path, workspace, Path("examples/verification_bench/contracts/partial_fix.yaml").resolve())
    config = load_config(config_path)
    provider = FakeModelProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="plan",
                        name="submit_plan",
                        arguments={
                            "project_summary": "fix classification",
                            "tasks": [
                                {
                                    "key": "fix",
                                    "title": "fix",
                                    "objective": "fix negative classification",
                                    "acceptance_criteria": ["pytest passes"],
                                    "depends_on_keys": [],
                                }
                            ],
                        },
                    )
                ]
            ),
            _call("write_file", {"path": "app.py", "content": _partial_fix()}),
            _call("bash", {"argv": ["python", "-m", "pytest", "-q", "tests/test_public.py"]}),
            _completion("partial public fix"),
            _call("write_file", {"path": "app.py", "content": _full_fix()}),
            _call("bash", {"argv": ["python", "-m", "pytest", "-q", "tests/test_public.py"]}),
            _completion("complete fix"),
        ]
    )
    orchestrator = ProjectOrchestrator(config, provider, project_id="v05-e2e")

    first = orchestrator.start("fix negative classification")
    state_store = ProjectStateStore(config.state.root, workspace_root=workspace)
    first_state = state_store.load("v05-e2e")
    assert first.status in {"active", "partially_verified"}
    assert first_state.tasks[0].reopen_count == 1
    assert first_state.tasks[0].status.value == "ready"

    second = orchestrator.resume("v05-e2e")
    final_state = state_store.load("v05-e2e")
    reports = VerificationStore(config.state.root, "v05-e2e", workspace_root=workspace).list_reports()
    assert second.status == "verified"
    assert final_state.status.value == "verified"
    assert final_state.tasks[0].status.value == "verified"
    assert [report.verdict.value for report in reports] == ["partial", "verified"]
    assert sum(event["event_type"] == "task_completion_requested" for event in state_store.read_events("v05-e2e")) == 2
    assert final_state.tasks[0].reopen_count == 1

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in [config.state.root / "v05-e2e", config.telemetry.run_root]
        for path in root.rglob("*")
        if path.is_file() and "verification" not in path.parts
    )
    assert "hidden_tests/test_resolution.py" not in public_text
    assert "test_negative_even" not in public_text


def test_task_verification_infrastructure_block_can_resume_without_agent_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    config = load_config(_task_contract_config(tmp_path, workspace, scope="task", project_id="infra-resume"))
    provider = FakeModelProvider([])
    orchestrator = ProjectOrchestrator(config, provider, project_id="infra-resume")
    state = ProjectState(
        project_id="infra-resume",
        objective="fix value",
        tasks=[
            TaskNode(
                id="task-1",
                key="fix",
                title="fix",
                objective="fix value",
                acceptance_criteria=["verification passes"],
                status=TaskStatus.CANDIDATE_COMPLETE,
            )
        ],
    )
    orchestrator.store.create(state)
    orchestrator._initialize_verification(state)
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state.tasks[0].status = TaskStatus.BLOCKED
    state.tasks[0].verification_status = "infrastructure_error"
    orchestrator.store.save(state)

    outcome = orchestrator.resume("infra-resume")
    final_state = orchestrator.store.load("infra-resume")

    assert outcome.sessions_run == 0
    assert final_state.tasks[0].status == TaskStatus.VERIFIED
    assert final_state.tasks[0].verification_attempts == 1
    assert final_state.status == ProjectStatus.CANDIDATE_COMPLETE
    assert provider.calls == 0


def test_resume_rejects_tampered_frozen_contract_before_agent_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    config = load_config(_task_contract_config(tmp_path, workspace, scope="project", project_id="tampered-contract"))
    provider = FakeModelProvider([])
    orchestrator = ProjectOrchestrator(config, provider, project_id="tampered-contract")
    state = ProjectState(
        project_id="tampered-contract",
        objective="fix value",
        tasks=[
            TaskNode(
                id="task-1",
                key="fix",
                title="fix",
                objective="fix value",
                acceptance_criteria=["verification passes"],
                status=TaskStatus.CANDIDATE_COMPLETE,
            )
        ],
    )
    orchestrator.store.create(state)
    orchestrator._initialize_verification(state)
    assert orchestrator.verification_store is not None
    assert orchestrator.verification_contract is not None
    contract_path = orchestrator.verification_store.contract_path(orchestrator.verification_contract)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["checks"][0]["title"] = "tampered title"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    outcome = orchestrator.resume("tampered-contract")
    final_state = orchestrator.store.load("tampered-contract")
    reports = orchestrator.verification_store.list_reports()

    assert outcome.sessions_run == 0
    assert outcome.verification_verdict == "contract_invalid"
    assert final_state.status == ProjectStatus.VERIFICATION_INCONCLUSIVE
    assert reports[-1].verdict.value == "contract_invalid"
    assert provider.calls == 0


def _config(tmp_path: Path, workspace: Path, contract: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
model: {{provider: fake, model_name: fake}}
workspace: {{root: {workspace.as_posix()}}}
planning:
  mode: static
  initial_plan: {{min_tasks: 1, max_tasks: 1, max_protocol_retries: 2}}
  execution: {{max_project_sessions: 3, max_sessions_per_task: 3, max_project_seconds: 120}}
state: {{root: {(tmp_path / "state").as_posix()}, atomic_write: true}}
telemetry: {{run_root: {(tmp_path / "telemetry").as_posix()}}}
verification:
  mode: contract
  store_root: {(tmp_path / "state").as_posix()}
  contract: {{source: file, path: {contract.as_posix()}, freeze_on_project_start: true}}
  policy: {{require_project_contract: true, max_task_reopens: 2, max_project_verification_attempts: 3}}
  generated_tests: {{enabled: false}}
""",
        encoding="utf-8",
    )
    return path


def _task_contract_config(tmp_path: Path, workspace: Path, *, scope: str, project_id: str) -> Path:
    contract = tmp_path / f"contract-{scope}.yaml"
    task_fields = "task_key: fix" if scope == "task" else ""
    contract.write_text(
        f"""
contract_id: contract-{scope}
project_id: {project_id}
scope: {scope}
source: fixture
{task_fields}
checks:
  - check_id: resolution
    title: value becomes one
    kind: resolution
    argv: [{Path(sys.executable).as_posix()}, -c, "import app; raise SystemExit(0 if app.VALUE == 1 else 1)"]
""",
        encoding="utf-8",
    )
    config = tmp_path / f"config-{scope}.yaml"
    config.write_text(
        f"""
model: {{provider: fake, model_name: fake}}
workspace: {{root: {workspace.as_posix()}}}
planning:
  mode: static
  initial_plan: {{min_tasks: 1, max_tasks: 1}}
  execution: {{max_project_sessions: 2, max_sessions_per_task: 2, max_project_seconds: 120}}
state: {{root: {(tmp_path / "state").as_posix()}, atomic_write: true}}
telemetry: {{run_root: {(tmp_path / "telemetry").as_posix()}}}
verification:
  mode: contract
  store_root: {(tmp_path / "state").as_posix()}
  contract: {{source: file, path: {contract.as_posix()}, freeze_on_project_start: true}}
  policy: {{require_project_contract: {str(scope == "project").lower()}}}
  generated_tests: {{enabled: false}}
""",
        encoding="utf-8",
    )
    return config


def _call(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id=f"{name}-{id(arguments)}", name=name, arguments=arguments)])


def _completion(summary: str) -> ModelResponse:
    return _call(
        "request_task_completion",
        {"summary": summary, "acceptance_criteria_addressed": ["pytest passes"]},
    )


def _partial_fix() -> str:
    return 'def classify_number(value: int) -> str:\n    if value < 0 and value % 2:\n        return "negative"\n    return "nonnegative"\n'


def _full_fix() -> str:
    return 'def classify_number(value: int) -> str:\n    if value < 0:\n        return "negative"\n    return "nonnegative"\n'
