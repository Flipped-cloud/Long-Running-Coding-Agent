import json
from pathlib import Path

import pytest

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import AgentConfig, AppConfig, BashConfig, ModelConfig, TelemetryConfig, ToolsConfig, WorkspaceConfig
from longrun_agent.control.channel import ControlSignalType, TaskControlChannel
from longrun_agent.control.tools import control_tools
from longrun_agent.exceptions import ProviderError
from longrun_agent.knowledge.tools import KnowledgeUseChannel, ReportKnowledgeUseTool
from longrun_agent.model.base import ModelProvider
from longrun_agent.model.fake import FakeModelProvider, default_calculator_script
from longrun_agent.orchestration.orchestrator import _ChannelRouter
from longrun_agent.orchestration.session_trace import SessionTrace
from longrun_agent.protocol import ErrorType, FinalAnswer, ModelResponse, RunStatus, ToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.router import ToolRouter


class RaisingProvider(ModelProvider):
    def generate(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        raise ProviderError("temporary API failure")


def config(workspace: Path, run_root: Path, max_steps: int = 10) -> AppConfig:
    return AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        agent=AgentConfig(max_steps=max_steps, max_consecutive_errors=2),
        workspace=WorkspaceConfig(root=workspace),
        tools=ToolsConfig(bash=BashConfig(timeout_seconds=10, max_output_chars=20000, shell=False)),
        telemetry=TelemetryConfig(run_root=run_root, save_prompts=True, save_full_tool_outputs=True),
    )


def make_calculator_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def divide(a: float, b: float) -> float:\n    return a * b\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        """
import pytest
from calculator import divide

def test_divide_positive_numbers():
    assert divide(8, 2) == 4

def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)
""",
        encoding="utf-8",
    )
    return repo


def test_agent_loop_full_fake_provider_trace_repairs_calculator(tmp_path: Path):
    repo = make_calculator_repo(tmp_path)
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(default_calculator_script()), run_id="run1").run(
        repo,
        "Fix the implementation bug in calculator.py so that all tests pass.",
    )
    assert result.status == RunStatus.COMPLETED
    assert result.tool_call_count == 3
    assert "return a / b" in (repo / "calculator.py").read_text(encoding="utf-8")
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert {"read_file", "write_file", "bash"}.issubset({event.get("tool_name") for event in events})
    assert "final_answer" in {event["event_type"] for event in events}
    bash_events = [event for event in events if event.get("tool_name") == "bash"]
    assert bash_events[-1]["exit_code"] == 0


def test_agent_loop_multiple_tool_calls_execute_in_order(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [
        ModelResponse(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "1"}),
                ToolCall(id="w2", name="write_file", arguments={"path": "b.txt", "content": "2"}),
            ]
        ),
        ModelResponse(final_answer=FinalAnswer(content="done")),
    ]
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(responses), run_id="run2").run(repo, "write files")
    assert result.status == RunStatus.COMPLETED
    assert (repo / "a.txt").read_text(encoding="utf-8") == "1"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "2"


def test_agent_loop_stops_at_max_steps(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [
        ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "missing.txt"})]),
        ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "missing.txt"})]),
    ]
    result = AgentLoop(config(repo, tmp_path / ".runs", max_steps=1), FakeModelProvider(responses), run_id="run3").run(repo, "loop")
    assert result.status == RunStatus.MAX_STEPS_REACHED
    assert result.final_answer is None
    assert Path(result.run_json_path).exists()
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event_type"] == "run_finished"
    assert events[-1]["success"] is False


def test_agent_loop_provider_exception_status(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = AgentLoop(config(repo, tmp_path / ".runs"), RaisingProvider(), run_id="run4").run(repo, "task")
    assert result.status == RunStatus.PROVIDER_ERROR
    assert Path(result.run_json_path).exists()
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert "provider_error" in {event["event_type"] for event in events}
    assert events[-1]["event_type"] == "run_finished"
    assert events[-1]["success"] is False


def test_agent_loop_aborts_after_consecutive_empty_responses(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [ModelResponse(), ModelResponse()]
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(responses), run_id="run-empty").run(repo, "task")
    assert result.status == RunStatus.ABORTED
    assert result.final_answer is None
    assert result.consecutive_errors == 2
    assert Path(result.event_log_path).exists()
    assert Path(result.run_json_path).exists()
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events].count("protocol_error") == 2
    assert events[-1]["event_type"] == "run_finished"
    assert events[-1]["success"] is False


def test_agent_loop_aborts_after_consecutive_unknown_tools(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [
        ModelResponse(tool_calls=[ToolCall(id="bad-1", name="missing_tool", arguments={})]),
        ModelResponse(tool_calls=[ToolCall(id="bad-2", name="missing_tool", arguments={})]),
    ]
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(responses), run_id="run-tools").run(repo, "task")
    assert result.status == RunStatus.ABORTED
    assert result.consecutive_errors == 2
    assert result.final_answer is None
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    failed_tool_events = [event for event in events if event["event_type"] == "tool_finished" and event["tool_name"] == "missing_tool"]
    assert len(failed_tool_events) == 2
    assert all(event["success"] is False for event in failed_tool_events)
    assert events[-1]["event_type"] == "run_finished"
    assert events[-1]["success"] is False


def test_fake_provider_response_exhaustion():
    provider = FakeModelProvider([])
    with pytest.raises(ProviderError):
        provider.generate([], [])


def test_terminal_tool_schema_allows_knowledge_decision(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    router = ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()])
    loop = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider([]), router=router)
    names = {schema["function"]["name"] for schema in loop._schemas(terminal_tools_only=True)}
    assert {"request_task_completion", "report_blocker", "report_knowledge_use"} <= names


def test_report_knowledge_use_executes_before_completion_in_same_response(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    channel = TaskControlChannel()
    knowledge = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=[])
    trace = SessionTrace()
    inner = ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()])
    router = _ChannelRouter(inner, channel, trace, knowledge_channel=knowledge)
    responses = [
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="complete",
                    name="request_task_completion",
                    arguments={"summary": "done", "acceptance_criteria_addressed": ["verified"]},
                ),
                ToolCall(
                    id="decision",
                    name="report_knowledge_use",
                    arguments={"memory_ids": ["m1"], "skill_ids": [], "reason": "used memory"},
                ),
            ]
        )
    ]

    result = AgentLoop(config(repo, tmp_path / ".runs", max_steps=1), FakeModelProvider(responses), router=router).run_with_controls(
        repo,
        "finish task",
        stop_condition=lambda: channel.terminal_signal is not None,
        require_external_terminal=True,
    )

    assert result.status == RunStatus.COMPLETED
    assert knowledge.decision_recorded is True
    assert channel.terminal_signal is not None
    assert channel.terminal_signal.type == ControlSignalType.COMPLETION_REQUEST
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    tool_finished = [event["tool_name"] for event in events if event["event_type"] == "tool_finished"]
    assert tool_finished[:2] == ["report_knowledge_use", "request_task_completion"]


def test_knowledge_decision_gate_blocks_verification_until_reported(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    channel = TaskControlChannel()
    knowledge = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=[])
    trace = SessionTrace()
    router = _ChannelRouter(
        ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()]),
        channel,
        trace,
        knowledge_channel=knowledge,
    )
    context = ToolContext(repo)

    blocked = router.execute(ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]}), context)
    assert not blocked.success
    assert blocked.summary == "knowledge_decision_required"
    assert blocked.error_type == ErrorType.POLICY_GATE
    assert router.action_required_message

    reported = router.execute(
        ToolCall(
            id="k1",
            name="report_knowledge_use",
            arguments={"memory_ids": [], "skill_ids": [], "reason": "reviewed but not needed"},
        ),
        context,
    )
    assert reported.success
    assert knowledge.decision_recorded is True


def test_gate_rejected_write_is_not_repeated_action(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    knowledge = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=[])
    trace = SessionTrace()
    router = _ChannelRouter(
        ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()]),
        TaskControlChannel(),
        trace,
        knowledge_channel=knowledge,
    )
    context = ToolContext(repo)
    call = ToolCall(id="w1", name="write_file", arguments={"path": "app.py", "content": "VALUE = 1\n"})

    blocked = router.execute(call, context)

    assert blocked.error_type == ErrorType.POLICY_GATE
    assert trace.repeated_tool_calls == []
    assert trace.suppressed_tool_calls == []
    assert trace.changed_files == []
    assert trace.no_progress(progress_count=0, terminal_signal=None) is False


def test_write_after_knowledge_decision_is_legitimate_retry(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    knowledge = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=[])
    trace = SessionTrace()
    router = _ChannelRouter(
        ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()]),
        TaskControlChannel(),
        trace,
        knowledge_channel=knowledge,
    )
    context = ToolContext(repo)
    write = ToolCall(id="w1", name="write_file", arguments={"path": "app.py", "content": "VALUE = 1\n"})

    blocked = router.execute(write, context)
    reported = router.execute(
        ToolCall(id="k1", name="report_knowledge_use", arguments={"memory_ids": ["m1"], "skill_ids": [], "reason": "used memory"}),
        context,
    )
    retried = router.execute(ToolCall(id="w2", name="write_file", arguments=write.arguments), context)

    assert blocked.error_type == ErrorType.POLICY_GATE
    assert reported.success
    assert retried.success
    assert trace.repeated_tool_calls == []
    assert trace.suppressed_tool_calls == []
    assert trace.changed_files == ["app.py"]


def test_policy_gate_does_not_increment_protocol_or_consecutive_errors(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    knowledge = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=[])
    trace = SessionTrace()
    router = _ChannelRouter(
        ToolRouter([*default_router().tools.values(), *control_tools(), ReportKnowledgeUseTool()]),
        TaskControlChannel(),
        trace,
        knowledge_channel=knowledge,
    )
    responses = [
        ModelResponse(tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "app.py", "content": "VALUE = 1\n"})]),
        ModelResponse(tool_calls=[ToolCall(id="w2", name="write_file", arguments={"path": "app.py", "content": "VALUE = 2\n"})]),
    ]

    result = AgentLoop(config(repo, tmp_path / ".runs", max_steps=2), FakeModelProvider(responses), router=router).run(repo, "task")

    assert result.protocol_error_count == 0
    assert result.recoverable_protocol_error_count == 0
    assert result.fatal_protocol_error_count == 0
    assert result.consecutive_errors == 0
    assert trace.repeated_tool_calls == []
