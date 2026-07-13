from __future__ import annotations

from pydantic import BaseModel, Field

from longrun_agent.control.channel import ControlSignal, ControlSignalType, TaskControlChannel
from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext


def _channel(context: ToolContext) -> TaskControlChannel:
    if context.control_channel is None:
        raise ValueError("control channel is not configured")
    return context.control_channel


def _success(call_id: str, tool_name: str, summary: str) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        tool_name=tool_name,
        success=True,
        summary=summary,
        output=f"{summary}. Stop further repository modifications and provide a brief final answer.",
    )


def _failure(call_id: str, tool_name: str, exc: Exception) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        tool_name=tool_name,
        success=False,
        summary=f"{tool_name} failed: {exc}",
        output=str(exc),
        error_type=ErrorType.TOOL,
        error_message=str(exc),
    )


class ReportProgressArgs(BaseModel):
    summary: str = Field(min_length=1)
    files_touched: list[str] = Field(default_factory=list)


class ReportProgressTool(BaseTool):
    name = "report_progress"
    description = "Report progress on the active task without changing task state."
    args_model = ReportProgressArgs

    def execute(self, call_id: str, arguments: ReportProgressArgs, context: ToolContext) -> ToolResult:
        try:
            _channel(context).record(
                ControlSignal(type=ControlSignalType.PROGRESS, summary=arguments.summary, files_touched=arguments.files_touched)
            )
            return _success(call_id, self.name, "progress recorded")
        except ValueError as exc:
            return _failure(call_id, self.name, exc)


class ReportBlockerArgs(BaseModel):
    reason: str = Field(min_length=1)
    attempted_actions: list[str] = Field(default_factory=list)
    decomposition_recommended: bool = False


class ReportBlockerTool(BaseTool):
    name = "report_blocker"
    description = "Report that the active task is blocked."
    args_model = ReportBlockerArgs

    def execute(self, call_id: str, arguments: ReportBlockerArgs, context: ToolContext) -> ToolResult:
        try:
            _channel(context).record(
                ControlSignal(
                    type=ControlSignalType.BLOCKER,
                    reason=arguments.reason,
                    attempted_actions=arguments.attempted_actions,
                    decomposition_recommended=arguments.decomposition_recommended,
                )
            )
            return _success(call_id, self.name, "blocker recorded")
        except ValueError as exc:
            return _failure(call_id, self.name, exc)


class RequestTaskCompletionArgs(BaseModel):
    summary: str = Field(min_length=1)
    acceptance_criteria_addressed: list[str] = Field(default_factory=list)


class RequestTaskCompletionTool(BaseTool):
    name = "request_task_completion"
    description = "Request that the active task become candidate_complete."
    args_model = RequestTaskCompletionArgs

    def execute(self, call_id: str, arguments: RequestTaskCompletionArgs, context: ToolContext) -> ToolResult:
        try:
            _channel(context).record(
                ControlSignal(
                    type=ControlSignalType.COMPLETION_REQUEST,
                    summary=arguments.summary,
                    acceptance_criteria_addressed=arguments.acceptance_criteria_addressed,
                )
            )
            return _success(call_id, self.name, "task completion requested")
        except ValueError as exc:
            return _failure(call_id, self.name, exc)


class RequestDecompositionArgs(BaseModel):
    reason: str = Field(min_length=1)


class RequestDecompositionTool(BaseTool):
    name = "request_decomposition"
    description = "Request decomposition of the active task."
    args_model = RequestDecompositionArgs

    def execute(self, call_id: str, arguments: RequestDecompositionArgs, context: ToolContext) -> ToolResult:
        try:
            _channel(context).record(ControlSignal(type=ControlSignalType.DECOMPOSITION_REQUEST, reason=arguments.reason))
            return _success(call_id, self.name, "task decomposition requested")
        except ValueError as exc:
            return _failure(call_id, self.name, exc)


def control_tools() -> list[BaseTool]:
    return [ReportProgressTool(), ReportBlockerTool(), RequestTaskCompletionTool(), RequestDecompositionTool()]
