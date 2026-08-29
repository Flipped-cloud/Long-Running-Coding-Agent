from pathlib import Path

from pydantic import BaseModel

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import AgentToolCall, ErrorType
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.read_file import ReadFileTool
from longrun_agent.tools.router import ToolRouter


def test_router_unknown_tool(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, artifacts_dir=tmp_path, config=ToolsConfig())
    result = ToolRouter([]).execute(AgentToolCall(call_id="c1", tool_name="missing", arguments={}), ctx)
    assert not result.success
    assert result.error_type == ErrorType.TOOL


def test_router_argument_validation(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, artifacts_dir=tmp_path, config=ToolsConfig())
    result = ToolRouter([ReadFileTool()]).execute(AgentToolCall(call_id="c1", tool_name="read_file", arguments={}), ctx)
    assert not result.success
    assert result.error_type == ErrorType.INVALID_TOOL_ARGUMENTS
    assert result.retryable is True


class BrokenArgs(BaseModel):
    pass


class BrokenTool(BaseTool):
    name = "broken"
    description = "Raise an internal error."
    args_model = BrokenArgs

    def execute(self, call_id, arguments, context):
        raise RuntimeError("broken implementation")


def test_router_returns_internal_tool_errors_as_observations(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, artifacts_dir=tmp_path, config=ToolsConfig())
    result = ToolRouter([BrokenTool()]).execute(AgentToolCall(call_id="c1", tool_name="broken", arguments={}), ctx)
    assert not result.success
    assert result.error_type == ErrorType.TOOL_INTERNAL
    assert result.error_message == "RuntimeError: broken implementation"
