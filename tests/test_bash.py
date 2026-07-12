from pathlib import Path

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import AgentToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.bash import BashTool
from longrun_agent.tools.router import ToolRouter


def context(tmp_path: Path, max_output_chars: int = 12000) -> ToolContext:
    artifacts = tmp_path / ".runs" / "r1" / "artifacts"
    artifacts.mkdir(parents=True)
    return ToolContext(
        workspace=tmp_path, artifacts_dir=artifacts, config=ToolsConfig(max_output_chars=max_output_chars, bash_timeout_seconds=2)
    )


def execute(tmp_path: Path, args: dict, max_output_chars: int = 12000):
    return ToolRouter([BashTool()]).execute(
        AgentToolCall(call_id="c1", tool_name="bash", arguments=args), context(tmp_path, max_output_chars)
    )


def test_bash_normal_command(tmp_path: Path):
    result = execute(tmp_path, {"command": "python -c \"print('ok')\""})
    assert result.success
    assert result.metadata["exit_code"] == 0
    assert "ok" in result.output


def test_bash_nonzero_exit_is_observation(tmp_path: Path):
    result = execute(tmp_path, {"command": 'python -c "import sys; sys.exit(3)"'})
    assert result.success
    assert result.metadata["exit_code"] == 3


def test_bash_timeout(tmp_path: Path):
    result = execute(tmp_path, {"command": 'python -c "import time; time.sleep(5)"', "timeout": 1})
    assert not result.success
    assert result.metadata["timed_out"] is True


def test_bash_output_truncation_creates_artifact(tmp_path: Path):
    result = execute(tmp_path, {"command": "python -c \"print('x'*500)\""}, max_output_chars=100)
    assert result.success
    assert "truncated" in result.output
    assert Path(result.metadata["output_artifact"]).exists()


def test_bash_dangerous_command_rejected(tmp_path: Path):
    result = execute(tmp_path, {"command": "rm -rf /"})
    assert not result.success
    assert "rejected" in result.summary
