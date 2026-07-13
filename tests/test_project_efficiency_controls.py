import json
import time
from pathlib import Path

from longrun_agent.agent.loop import AgentLoop
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.base import ModelProvider
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import FinalAnswer, ModelResponse, RunStatus, ToolCall
from longrun_agent.state.schema import ProjectStatus, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from tests.test_agent_loop import config as loop_config
from tests.test_project_orchestrator import completion, config, submit_plan


def one_task_plan():
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="plan-one",
                name="submit_plan",
                arguments={
                    "project_summary": "one task",
                    "tasks": [
                        {
                            "key": "T1",
                            "title": "T1",
                            "objective": "first task",
                            "acceptance_criteria": ["done"],
                            "depends_on_keys": [],
                        }
                    ],
                },
            )
        ]
    )


def test_project_terminal_completion_stops_without_followup_model_call(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=2)
    cfg.planning.initial_plan.min_tasks = 1
    provider = FakeModelProvider(
        [
            one_task_plan(),
            completion("c1"),
            ModelResponse(tool_calls=[ToolCall(id="unused", name="read_file", arguments={"path": "unused.py"})]),
        ]
    )
    ProjectOrchestrator(cfg, provider, project_id="stop-completion").start("ship")
    assert provider.calls == 2


def test_project_terminal_blocker_stops_without_followup_model_call(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=2)
    provider = FakeModelProvider(
        [
            submit_plan(),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="b1",
                        name="report_blocker",
                        arguments={"reason": "blocked", "attempted_actions": [], "decomposition_recommended": False},
                    )
                ]
            ),
            completion("unused"),
        ]
    )
    outcome = ProjectOrchestrator(cfg, provider, project_id="stop-blocker").start("ship")
    assert provider.calls == 2
    assert outcome.status == ProjectStatus.BLOCKED.value


def test_project_final_answer_without_terminal_requests_control_tool(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    provider = FakeModelProvider([submit_plan(), ModelResponse(final_answer=FinalAnswer(content="done")), completion("c1")])
    ProjectOrchestrator(cfg, provider, project_id="final-repair").start("ship")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("final-repair")
    events = [
        json.loads(line)
        for line in Path(cfg.telemetry.run_root / "final-repair-s1" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "project_session_final_without_signal" in {event["event_type"] for event in events}
    assert state.task_by_id("final-repair:T1").status == TaskStatus.CANDIDATE_COMPLETE


def test_non_project_agent_final_answer_still_completes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = AgentLoop(
        loop_config(repo, tmp_path / ".runs"),
        FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content="done"))]),
        run_id="plain-final",
    ).run(repo, "finish")
    assert result.status == RunStatus.COMPLETED
    assert result.final_answer == "done"


def test_agent_loop_session_deadline_stops_before_model_request(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content="unused"))])
    result = AgentLoop(loop_config(repo, tmp_path / ".runs"), provider, run_id="deadline").run_with_controls(
        repo,
        "deadline",
        deadline_monotonic=time.monotonic() - 1,
    )
    assert result.status == RunStatus.TIME_LIMIT_REACHED
    assert provider.calls == 0


def test_project_wall_clock_deadline_marks_time_limit(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.execution.max_project_seconds = 30
    orchestrator = ProjectOrchestrator(cfg, FakeModelProvider([submit_plan()]), project_id="project-deadline")
    original = time.monotonic
    values = iter([100.0, 200.0, 200.0, 200.0])
    try:
        time.monotonic = lambda: next(values)
        outcome = orchestrator.start("ship")
    finally:
        time.monotonic = original
    assert outcome.status == ProjectStatus.TIME_LIMIT_REACHED.value
    metrics = json.loads((cfg.state.root / "project-deadline" / "project_metrics.json").read_text(encoding="utf-8"))
    assert metrics["time_budget_exhausted"] is True


class CapturingProvider(FakeModelProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.messages = []
        self.tools_seen = []

    def generate(self, messages, tools):
        self.messages.append(messages)
        self.tools_seen.append(tools)
        return super().generate(messages, tools)


class ProtocolThenResponseProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_read_only_streak_inserts_action_required(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 5
    for name in ["a.py", "b.py", "c.py"]:
        (cfg.workspace.root / name).write_text("VALUE = 1\n", encoding="utf-8")
    provider = CapturingProvider(
        [
            one_task_plan(),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})]),
            ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "b.py"})]),
            ModelResponse(tool_calls=[ToolCall(id="r3", name="read_file", arguments={"path": "c.py"})]),
            completion("c1"),
        ]
    )
    ProjectOrchestrator(cfg, provider, project_id="read-streak").start("ship")
    assert any("action_required:" in item.get("content", "") for item in provider.messages[-1] if item["role"] == "user")


def test_repeated_read_file_is_suppressed(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    (cfg.workspace.root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})]),
            ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "a.py"})]),
            completion("c1"),
        ]
    )
    ProjectOrchestrator(cfg, provider, project_id="repeat-read").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    session = store.read_sessions("repeat-read")[0]
    events = store.read_events("repeat-read")
    assert session["suppressed_tool_calls"]
    assert "repeated_tool_call_suppressed" in {event["event_type"] for event in events}


def test_write_file_resets_read_only_streak(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    (cfg.workspace.root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = CapturingProvider(
        [
            one_task_plan(),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})]),
            ModelResponse(tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "VALUE = 2\n"})]),
            ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "a.py"})]),
            completion("c1"),
        ]
    )
    ProjectOrchestrator(cfg, provider, project_id="write-resets-streak").start("ship")
    assert not any(
        "action_required:" in item.get("content", "") for messages in provider.messages for item in messages if item["role"] == "user"
    )


def test_terminal_grace_turn_recovers_completion_after_verified_work(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 2
    cfg.agent.terminal_grace_turns = 1
    provider = CapturingProvider(
        [
            one_task_plan(),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "test_sample.py", "content": "def test_ok():\n    assert True\n"},
                    )
                ]
            ),
            ModelResponse(tool_calls=[ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]})]),
            completion("c1"),
        ]
    )

    outcome = ProjectOrchestrator(cfg, provider, project_id="grace-complete").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    session = store.read_sessions("grace-complete")[0]
    events = {event["event_type"] for event in store.read_events("grace-complete")}
    grace_tool_names = {tool["function"]["name"] for tool in provider.tools_seen[-1]}
    assert outcome.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert session["terminal_signal"] == "completion_request"
    assert session["terminal_grace_turn_count"] == 1
    assert session["terminal_signal_recovered"] is True
    assert session["steps"] == 2
    assert grace_tool_names == {"report_blocker", "request_task_completion"}
    assert "terminal_grace_turn_started" in events
    assert "terminal_signal_recovered" in events
    assert "completion_candidate_created" in events
    assert "completion_candidate_confirmed" in events


def test_terminal_grace_turn_not_started_without_completion_evidence(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 1
    cfg.agent.terminal_grace_turns = 1
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "missing.py"})]),
        ]
    )

    outcome = ProjectOrchestrator(cfg, provider, project_id="no-grace").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    session = store.read_sessions("no-grace")[0]
    assert outcome.status == ProjectStatus.SESSION_LIMIT_REACHED.value
    assert session["terminal_grace_turn_count"] == 0
    assert session["terminal_signal"] is None


def test_tool_arguments_protocol_retry_stays_in_same_task_session(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 3
    cfg.agent.protocol_retries_per_step = 2
    provider = ProtocolThenResponseProvider(
        [
            one_task_plan(),
            ToolArgumentsProtocolError("write_file", "Expecting property name", "{path: bad}"),
            ModelResponse(tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "ok"})]),
            completion("c1"),
        ]
    )

    outcome = ProjectOrchestrator(cfg, provider, project_id="protocol-retry").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("protocol-retry")
    session = store.read_sessions("protocol-retry")[0]
    events = {event["event_type"] for event in store.read_events("protocol-retry")}
    assert outcome.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert state.task_by_id("protocol-retry:T1").attempts == 1
    assert len(state.task_by_id("protocol-retry:T1").session_ids) == 1
    assert session["tool_argument_protocol_retry_count"] == 1
    assert "tool_arguments_protocol_error" in events
    assert "tool_arguments_protocol_retry" in events
    assert "tool_arguments_protocol_recovered" in events


def test_tool_arguments_protocol_retry_limit_returns_protocol_error(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 1
    cfg.agent.protocol_retries_per_step = 1
    provider = ProtocolThenResponseProvider(
        [
            one_task_plan(),
            ToolArgumentsProtocolError("write_file", "bad json 1", "{path: bad}"),
            ToolArgumentsProtocolError("write_file", "bad json 2", "{path: bad}"),
        ]
    )

    ProjectOrchestrator(cfg, provider, project_id="protocol-limit").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    session = store.read_sessions("protocol-limit")[0]
    assert session["run_status"] == RunStatus.PROTOCOL_ERROR.value
    assert session["tool_argument_protocol_retry_count"] == 1


def test_grace_turn_without_tool_call_auto_completes_from_candidate(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 2
    cfg.agent.terminal_grace_turns = 1
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "test_sample.py", "content": "def test_ok():\n    assert True\n"},
                    )
                ]
            ),
            ModelResponse(tool_calls=[ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]})]),
            ModelResponse(),
        ]
    )

    outcome = ProjectOrchestrator(cfg, provider, project_id="auto-complete").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("auto-complete")
    task = state.task_by_id("auto-complete:T1")
    session = store.read_sessions("auto-complete")[0]
    events = {event["event_type"] for event in store.read_events("auto-complete")}
    assert outcome.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert task.status == TaskStatus.CANDIDATE_COMPLETE
    assert task.auto_completion_recovered is True
    assert task.completion_candidate is not None
    assert task.completion_candidate.changed_files == ["test_sample.py"]
    assert session["run_status"] == RunStatus.TERMINAL_SIGNAL_MISSING.value
    assert session["auto_completion_recovered"] is True
    assert session["completion_candidate"]["task_id"] == "auto-complete:T1"
    assert "completion_candidate_created" in events
    assert "auto_completion_recovered" in events


def test_auto_completion_not_created_without_verification_evidence(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 1
    cfg.agent.terminal_grace_turns = 1
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "sample.py", "content": "VALUE = 1\n"},
                    )
                ]
            ),
        ]
    )

    outcome = ProjectOrchestrator(cfg, provider, project_id="no-auto-without-verification").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("no-auto-without-verification")
    task = state.task_by_id("no-auto-without-verification:T1")
    assert outcome.status == ProjectStatus.SESSION_LIMIT_REACHED.value
    assert task.status == TaskStatus.IN_PROGRESS
    assert task.completion_candidate is None
    assert not task.auto_completion_recovered


def test_auto_completion_not_created_when_blocker_reported(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 3
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "test_sample.py", "content": "def test_ok():\n    assert True\n"},
                    )
                ]
            ),
            ModelResponse(tool_calls=[ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]})]),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="block",
                        name="report_blocker",
                        arguments={"reason": "remaining issue", "attempted_actions": ["pytest"], "decomposition_recommended": False},
                    )
                ]
            ),
        ]
    )

    ProjectOrchestrator(cfg, provider, project_id="blocker-no-auto").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    task = store.load("blocker-no-auto").task_by_id("blocker-no-auto:T1")
    events = {event["event_type"] for event in store.read_events("blocker-no-auto")}
    assert task.status == TaskStatus.BLOCKED
    assert task.completion_candidate is None
    assert "auto_completion_recovered" not in events


def test_handoff_is_action_oriented_without_completion_candidate(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan.min_tasks = 1
    cfg.agent.max_steps = 1
    (cfg.workspace.root / "a.py").write_text("A = 1\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            one_task_plan(),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "a.py"})]),
        ]
    )

    ProjectOrchestrator(cfg, provider, project_id="action-handoff").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    session = store.read_sessions("action-handoff")[0]
    handoff = session["handoff_summary"]
    assert session["run_status"] == RunStatus.MAX_STEPS_REACHED.value
    assert "Completed work:" in handoff
    assert "Next required action:" in handoff
    assert "Do not repeat:" in handoff
    assert "Commands:" not in handoff
