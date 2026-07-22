from __future__ import annotations

import time

from pydantic import ValidationError

from longrun_agent.protocol import ErrorType, ToolCall, ToolResult
from longrun_agent.tools.arguments import ArgumentNormalization, ToolArgumentError
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.result_guard import sanitize_agent_visible_tool_result


class ToolRouter:
    """Register tools, validate arguments, and isolate tool failures."""

    def __init__(self, tools: list[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.openai_schema() for tool in self.tools.values()]

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        started = time.monotonic()
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                summary=f"unknown tool: {call.name}",
                error_type=ErrorType.TOOL,
                error_message=f"unknown tool: {call.name}",
                metadata={"duration_seconds": time.monotonic() - started},
            )
        validation_context: dict[str, list[ArgumentNormalization]] = {"argument_normalizations": []}
        try:
            args = tool.args_model.model_validate(call.arguments, context=validation_context)
            result = tool.execute(call.id, args, context)
            result = sanitize_agent_visible_tool_result(result, context.workspace_policy)
            result.metadata.setdefault("duration_seconds", time.monotonic() - started)
            _attach_normalizations(result.metadata, validation_context["argument_normalizations"])
            return result
        except (ValidationError, ToolArgumentError, ValueError, TypeError) as exc:
            message = _safe_argument_error(call.name, exc)
            metadata = {"duration_seconds": time.monotonic() - started, "failure_code": "TOOL_INVALID_ARGUMENT"}
            _attach_normalizations(metadata, validation_context["argument_normalizations"])
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                summary=f"invalid arguments for {call.name}",
                output=message,
                error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
                error_message=message,
                retryable=True,
                metadata=metadata,
            )


def _attach_normalizations(metadata: dict, records: list[ArgumentNormalization]) -> None:
    if records:
        metadata["argument_normalizations"] = [record.model_dump() for record in records]
        metadata["normalization_code"] = "TOOL_ARGUMENT_NORMALIZED"


def _safe_argument_error(tool_name: str, exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_context=True, include_input=False)
        detail = str((errors[0].get("ctx") or {}).get("error") or errors[0].get("msg") or "invalid arguments")
        detail = detail.removeprefix("Value error, ")
    else:
        detail = str(exc)
    return detail if detail.startswith(f"{tool_name} ") else f"{tool_name} {detail}"
