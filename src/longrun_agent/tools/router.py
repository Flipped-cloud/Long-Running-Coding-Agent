from __future__ import annotations

import time

from pydantic import ValidationError

from longrun_agent.protocol import ErrorType, ToolCall, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext


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
        try:
            args = tool.args_model.model_validate(call.arguments)
            result = tool.execute(call.id, args, context)
            result.metadata.setdefault("duration_seconds", time.monotonic() - started)
            return result
        except ValidationError as exc:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                summary=f"invalid arguments for {call.name}",
                output=str(exc),
                error_type=ErrorType.PROTOCOL,
                error_message="invalid tool arguments",
                metadata={"duration_seconds": time.monotonic() - started},
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                summary=f"tool crashed: {type(exc).__name__}",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
                metadata={"duration_seconds": time.monotonic() - started},
            )
