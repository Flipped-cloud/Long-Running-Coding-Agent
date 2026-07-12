from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app

runner = CliRunner()


def write_fake_config(path: Path, workspace: Path, run_root: Path) -> None:
    path.write_text(
        f"""
model:
  provider: fake
  model_name: fake
  base_url: null
  api_key_env: MODEL_API_KEY
  temperature: 0.0
  max_output_tokens: 100
  request_timeout_seconds: 30
  max_api_retries: 1
agent:
  max_steps: 10
  max_consecutive_errors: 5
workspace:
  root: {workspace.as_posix()}
tools:
  read_file:
    max_lines: 50
    max_chars: 1000
  write_file:
    max_chars: 2000
    save_diff: true
  bash:
    timeout_seconds: 10
    max_output_chars: 2000
    shell: false
telemetry:
  run_root: {run_root.as_posix()}
  save_prompts: true
  save_full_tool_outputs: true
""",
        encoding="utf-8",
    )


def make_repo(path: Path) -> None:
    path.mkdir()
    (path / "calculator.py").write_text("def divide(a: float, b: float) -> float:\n    return a * b\n", encoding="utf-8")
    tests = path / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        """
import pytest
from calculator import divide

def test_divide_positive_numbers():
    assert divide(8, 2) == 4

def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)
""",
        encoding="utf-8",
    )


def test_cli_tools_lists_schemas(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    write_fake_config(config, repo, tmp_path / ".runs")
    result = runner.invoke(app, ["tools", "--config", str(config)])
    assert result.exit_code == 0
    assert "read_file" in result.stdout
    assert "write_file" in result.stdout
    assert "bash" in result.stdout


def test_cli_fake_provider_run_repairs_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    make_repo(repo)
    config = tmp_path / "fake.yaml"
    write_fake_config(config, repo, tmp_path / ".runs")
    result = runner.invoke(app, ["run", "--config", str(config), "--fake-provider", "--workspace", str(repo)])
    assert result.exit_code == 0
    assert "completed" in result.stdout
    assert "return a / b" in (repo / "calculator.py").read_text(encoding="utf-8")
