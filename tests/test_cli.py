from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from longrun_agent.cli import app

runner = CliRunner()


def write_fake_config(path: Path, workspace: Path, run_root: Path, *, max_consecutive_errors: int = 5) -> None:
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
  max_consecutive_errors: {max_consecutive_errors}
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


def write_script(path: Path, items: list[dict]) -> None:
    import json

    path.write_text(json.dumps(items), encoding="utf-8")


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


def test_cli_completed_exit_code_is_zero_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    script = tmp_path / "script.json"
    write_fake_config(config, repo, tmp_path / ".runs")
    write_script(script, [{"final_answer": "done"}])
    result = runner.invoke(app, ["run", "--config", str(config), "--fake-provider", "--scripted-responses", str(script)])
    assert result.exit_code == 0
    assert "completed" in result.stdout


def test_cli_max_steps_exit_code_is_one(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    script = tmp_path / "script.json"
    write_fake_config(config, repo, tmp_path / ".runs")
    write_script(script, [{"tool_calls": [{"id": "r1", "name": "read_file", "arguments": {"path": "missing.txt"}}]}])
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--fake-provider", "--scripted-responses", str(script), "--max-steps", "1"],
    )
    assert result.exit_code == 1
    assert "max_steps_reached" in result.stdout


def test_cli_protocol_error_exit_code_is_one(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    script = tmp_path / "script.json"
    write_fake_config(config, repo, tmp_path / ".runs", max_consecutive_errors=2)
    write_script(script, [{}, {}])
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--fake-provider", "--scripted-responses", str(script)],
    )
    assert result.exit_code == 1
    assert "aborted" in result.stdout


def test_cli_missing_config_returns_nonzero():
    result = runner.invoke(app, ["run", "--config", "does-not-exist.yaml", "--fake-provider"])
    assert result.exit_code != 0


def test_sandbox_doctor_reports_preflight_without_creating_provider(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    write_fake_config(config, repo, tmp_path / ".runs")
    report = {
        "status": "GO",
        "python_runtime_available": True,
        "outside_workspace_hidden": True,
        "trusted_runtime_read_only": True,
    }
    monkeypatch.setattr(
        "longrun_agent.tools.sandbox.build_subprocess_sandbox",
        lambda _policy: SimpleNamespace(preflight=lambda: report),
    )
    monkeypatch.setattr("longrun_agent.cli._provider", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider called")))

    result = runner.invoke(app, ["sandbox", "doctor", "--config", str(config)])

    assert result.exit_code == 0
    assert '"status": "GO"' in result.stdout


def test_sandbox_doctor_reports_runtime_unavailable(tmp_path: Path, monkeypatch) -> None:
    from longrun_agent.tools.sandbox import EvaluationSandboxRuntimeUnavailable

    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "fake.yaml"
    write_fake_config(config, repo, tmp_path / ".runs")

    def unavailable(_policy):
        raise EvaluationSandboxRuntimeUnavailable("EVALUATION_SANDBOX_RUNTIME_UNAVAILABLE")

    monkeypatch.setattr("longrun_agent.tools.sandbox.build_subprocess_sandbox", unavailable)
    result = runner.invoke(app, ["sandbox", "doctor", "--config", str(config)])

    assert result.exit_code == 2
    assert "EVALUATION_SANDBOX_RUNTIME_UNAVAILABLE" in result.stdout
