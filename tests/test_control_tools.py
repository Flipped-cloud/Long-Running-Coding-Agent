from pathlib import Path

from longrun_agent.control.channel import ControlSignalType, TaskControlChannel
from longrun_agent.control.tools import ReportBlockerTool, ReportProgressTool, RequestDecompositionTool, RequestTaskCompletionTool
from longrun_agent.protocol import ToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.router import ToolRouter


def context(tmp_path: Path, channel: TaskControlChannel) -> ToolContext:
    return ToolContext(workspace=tmp_path, artifacts_dir=tmp_path / "artifacts", control_channel=channel)


def test_report_progress_records_multiple_signals(tmp_path: Path):
    channel = TaskControlChannel()
    router = ToolRouter([ReportProgressTool()])
    router.execute(
        ToolCall(id="p1", name="report_progress", arguments={"summary": "read files", "files_touched": ["a.py"]}),
        context(tmp_path, channel),
    )
    router.execute(ToolCall(id="p2", name="report_progress", arguments={"summary": "ran tests"}), context(tmp_path, channel))
    assert [signal.type for signal in channel.signals] == [ControlSignalType.PROGRESS, ControlSignalType.PROGRESS]


def test_terminal_signal_conflict_returns_structured_error(tmp_path: Path):
    channel = TaskControlChannel()
    router = ToolRouter([RequestTaskCompletionTool(), ReportBlockerTool()])
    first = router.execute(
        ToolCall(id="c1", name="request_task_completion", arguments={"summary": "done", "acceptance_criteria_addressed": ["done"]}),
        context(tmp_path, channel),
    )
    second = router.execute(
        ToolCall(id="b1", name="report_blocker", arguments={"reason": "blocked", "attempted_actions": []}),
        context(tmp_path, channel),
    )
    assert first.success
    assert not second.success
    assert channel.terminal_signal.type == ControlSignalType.COMPLETION_REQUEST


def test_request_decomposition_records_terminal_signal(tmp_path: Path):
    channel = TaskControlChannel()
    result = ToolRouter([RequestDecompositionTool()]).execute(
        ToolCall(id="d1", name="request_decomposition", arguments={"reason": "too broad"}),
        context(tmp_path, channel),
    )
    assert result.success
    assert channel.terminal_signal.type == ControlSignalType.DECOMPOSITION_REQUEST


def test_control_tool_without_channel_returns_error(tmp_path: Path):
    result = ToolRouter([ReportProgressTool()]).execute(
        ToolCall(id="p1", name="report_progress", arguments={"summary": "progress"}),
        ToolContext(workspace=tmp_path, artifacts_dir=tmp_path / "artifacts"),
    )
    assert not result.success
    assert "control channel" in result.error_message
