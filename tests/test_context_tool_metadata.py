from pathlib import Path

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import ToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.bash import BashTool
from longrun_agent.tools.read_file import ReadFileTool
from longrun_agent.tools.router import ToolRouter
from longrun_agent.tools.write_file import WriteFileTool


def test_read_file_returns_hash_metadata(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    result = ToolRouter([ReadFileTool()]).execute(
        ToolCall(id="r1", name="read_file", arguments={"path": "a.txt"}),
        ToolContext(tmp_path, config=ToolsConfig()),
    )

    assert result.success
    assert result.metadata["content_sha256"]
    assert result.metadata["size_bytes"] == 5
    assert isinstance(result.metadata["modified_time_ns"], int)


def test_write_file_no_change_returns_current_hash_and_code_epoch(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    context = ToolContext(tmp_path, config=ToolsConfig())
    router = ToolRouter([WriteFileTool()])

    result = router.execute(ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "hello"}), context)

    assert result.success
    assert result.metadata["status"] == "no_change"
    assert result.metadata["current_sha256"] == result.metadata["after_sha256"]
    assert result.metadata["code_epoch"] == 0


def test_bash_returns_normalized_command_and_verification_kind(tmp_path: Path):
    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-c", "print('ok')"]}),
        ToolContext(tmp_path, config=ToolsConfig()),
    )

    assert result.success
    assert result.metadata["normalized_command"] == "python -c print('ok')"
    assert "combined_artifact" in result.metadata
