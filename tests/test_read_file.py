from pathlib import Path

import pytest

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import AgentToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.read_file import ReadFileTool
from longrun_agent.tools.router import ToolRouter


def context(tmp_path: Path) -> ToolContext:
    artifacts = tmp_path / ".runs" / "r1" / "artifacts"
    artifacts.mkdir(parents=True)
    return ToolContext(workspace=tmp_path, artifacts_dir=artifacts, config=ToolsConfig())


def execute(tmp_path: Path, args: dict):
    return ToolRouter([ReadFileTool()]).execute(AgentToolCall(call_id="c1", tool_name="read_file", arguments=args), context(tmp_path))


def test_read_file_normal(tmp_path: Path):
    (tmp_path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    result = execute(tmp_path, {"path": "a.txt"})
    assert result.success
    assert "1 | one" in result.output
    assert result.metadata["total_lines"] == 2


def test_read_file_window(tmp_path: Path):
    (tmp_path / "a.txt").write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")
    result = execute(tmp_path, {"path": "a.txt", "start_line": 3, "end_line": 5})
    assert "3 | 3" in result.output
    assert "6 |" not in result.output
    assert result.metadata["has_previous"] is True
    assert result.metadata["has_next"] is True


def test_read_file_path_escape(tmp_path: Path):
    result = execute(tmp_path, {"path": "../outside.txt"})
    assert not result.success


def test_read_file_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    result = execute(tmp_path, {"path": "link.txt"})
    assert not result.success
