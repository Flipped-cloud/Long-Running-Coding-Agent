from pathlib import Path

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import AgentToolCall, ErrorType
from longrun_agent.tools.base import ToolContext
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
