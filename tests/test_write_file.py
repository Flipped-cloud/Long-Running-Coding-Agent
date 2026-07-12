from pathlib import Path

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import AgentToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.router import ToolRouter
from longrun_agent.tools.write_file import WriteFileTool


def context(tmp_path: Path) -> ToolContext:
    artifacts = tmp_path / ".runs" / "r1" / "artifacts"
    artifacts.mkdir(parents=True)
    return ToolContext(workspace=tmp_path, artifacts_dir=artifacts, config=ToolsConfig())


def execute(tmp_path: Path, args: dict):
    return ToolRouter([WriteFileTool()]).execute(AgentToolCall(call_id="c1", tool_name="write_file", arguments=args), context(tmp_path))


def test_write_file_create(tmp_path: Path):
    result = execute(tmp_path, {"path": "dir/a.txt", "content": "hello\n"})
    assert result.success
    assert (tmp_path / "dir" / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert result.metadata["status"] == "created"


def test_write_file_update(tmp_path: Path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    result = execute(tmp_path, {"path": "a.txt", "content": "new\n"})
    assert result.success
    assert result.metadata["status"] == "updated"
    assert result.metadata["before_hash"] != result.metadata["after_hash"]


def test_write_file_atomic_metadata(tmp_path: Path):
    result = execute(tmp_path, {"path": "a.txt", "content": ""})
    assert result.success
    assert result.metadata["atomic"] is True
    assert result.metadata["after_line_count"] == 0


def test_write_file_path_escape(tmp_path: Path):
    result = execute(tmp_path, {"path": "../outside.txt", "content": "bad"})
    assert not result.success
