import json
from pathlib import Path

from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectStatus, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config, submit_plan


def test_write_file_and_pytest_are_recorded_as_progress(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    (cfg.workspace.root / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "sample.py", "content": "VALUE = 1\n"}),
                ToolCall(id="b1", name="bash", arguments={"command": "python -m pytest -q"}),
                ToolCall(id="c1", name="request_task_completion", arguments={"summary": "done", "acceptance_criteria_addressed": ["done"]}),
            ]
        ),
    ]
    ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="trace-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("trace-1")
    session = store.read_sessions("trace-1")[0]
    assert "sample.py" in state.task_by_id("trace-1:T1").files_touched
    assert session["changed_files"] == ["sample.py"]
    assert session["successful_test_commands"] == ["python -m pytest -q"]
    assert session["no_progress"] is False


def test_write_file_no_change_is_not_changed_file(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    (cfg.workspace.root / "same.txt").write_text("same", encoding="utf-8")
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "same.txt", "content": "same"}),
                ToolCall(id="c1", name="request_task_completion", arguments={"summary": "done", "acceptance_criteria_addressed": ["done"]}),
            ]
        ),
    ]
    ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="trace-2").start("ship")
    session = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).read_sessions("trace-2")[0]
    assert session["written_files"] == ["same.txt"]
    assert session["changed_files"] == []


def test_handoff_and_no_progress_are_recorded_after_max_steps(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.agent.max_steps = 1
    (cfg.workspace.root / "a.py").write_text("A = 1\n", encoding="utf-8")
    responses = [submit_plan(), ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})])]
    ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="handoff-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("handoff-1")
    session = store.read_sessions("handoff-1")[0]
    assert session["run_status"] == "max_steps_reached"
    assert session["no_progress"] is True
    assert "Completed work:" in session["handoff_summary"]
    assert "Next required action:" in session["handoff_summary"]
    assert "Do not repeat:" in session["handoff_summary"]
    assert state.task_by_id("handoff-1:T1").consecutive_no_progress_sessions == 1
    assert "session_handoff_created" in {event["event_type"] for event in store.read_events("handoff-1")}


def test_per_task_session_limit_fails_project_and_resume_higher_limit_continues(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=3)
    cfg.agent.max_steps = 1
    cfg.planning.execution.max_sessions_per_task = 1
    (cfg.workspace.root / "a.py").write_text("A = 1\n", encoding="utf-8")
    responses = [submit_plan(), ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})])]
    first = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="limit-1").start("ship")
    assert first.status == ProjectStatus.FAILED.value
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("limit-1")
    assert state.task_by_id("limit-1:T1").status == TaskStatus.FAILED

    higher = config(tmp_path, mode="static", max_sessions=3)
    higher.planning.execution.max_sessions_per_task = 2
    resumed = ProjectOrchestrator(higher, FakeModelProvider([completion("c1")])).resume("limit-1")
    assert resumed.status == ProjectStatus.SESSION_LIMIT_REACHED.value
    resumed_state = ProjectStateStore(higher.state.root, workspace_root=higher.workspace.root).load("limit-1")
    assert resumed_state.task_by_id("limit-1:T1").status == TaskStatus.CANDIDATE_COMPLETE


def test_final_verification_success_and_failure(tmp_path: Path):
    success_root = tmp_path / "success"
    success_root.mkdir()
    success_cfg = config(success_root, mode="static", max_sessions=1)
    success_cfg.planning.execution.max_project_sessions = 2
    success_cfg.planning.execution.final_verification_command = ["python", "-c", "raise SystemExit(0)"]
    ProjectOrchestrator(success_cfg, FakeModelProvider([submit_plan(), completion("c1"), completion("c2")]), project_id="verify-ok").start(
        "ship"
    )
    success_store = ProjectStateStore(success_cfg.state.root, workspace_root=success_cfg.workspace.root)
    assert success_store.load("verify-ok").status == ProjectStatus.CANDIDATE_COMPLETE

    fail_root = tmp_path / "fail"
    fail_root.mkdir()
    fail_cfg = config(fail_root, mode="static", max_sessions=2)
    fail_cfg.planning.execution.final_verification_command = ["python", "-c", "raise SystemExit(7)"]
    ProjectOrchestrator(fail_cfg, FakeModelProvider([submit_plan(), completion("c1"), completion("c2")]), project_id="verify-fail").start(
        "ship"
    )
    fail_store = ProjectStateStore(fail_cfg.state.root, workspace_root=fail_cfg.workspace.root)
    assert fail_store.load("verify-fail").status == ProjectStatus.FAILED
    metrics = json.loads(fail_store.metrics_path("verify-fail").read_text(encoding="utf-8"))
    assert metrics["final_verification_exit_code"] == 7
    assert metrics["final_verification_passed"] is False
    assert "exit_code: 7" in fail_store.final_verification_path("verify-fail").read_text(encoding="utf-8")
