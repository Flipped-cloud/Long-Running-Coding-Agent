import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app
from tests.test_project_orchestrator import completion, config, submit_plan

runner = CliRunner()


def write_project_config(path: Path, tmp_path: Path) -> None:
    cfg = config(tmp_path, mode="static", max_sessions=1)
    path.write_text(
        f"""
model:
  provider: fake
  model_name: fake
  base_url: null
  api_key_env: MODEL_API_KEY
  temperature: 0.0
  max_output_tokens: 1024
  request_timeout_seconds: 30
  max_api_retries: 1
agent:
  max_steps: 5
  max_consecutive_errors: 2
workspace:
  root: {cfg.workspace.root.as_posix()}
tools:
  read_file:
    max_lines: 50
    max_chars: 1000
  write_file:
    max_chars: 2000
    save_diff: true
  bash:
    timeout_seconds: 5
    max_output_chars: 2000
    shell: false
telemetry:
  run_root: {(tmp_path / "runs").as_posix()}
  save_prompts: true
  save_full_tool_outputs: true
planning:
  mode: static
  initial_plan:
    min_tasks: 2
    max_tasks: 8
    max_protocol_retries: 2
  execution:
    max_project_sessions: 1
    attempts_before_decomposition: 1
    final_verification_command: []
  decomposition:
    max_depth: 3
    min_children: 2
    max_children: 5
    max_protocol_retries: 2
  bounded_search:
    enabled: false
    candidate_count: 3
    max_protocol_retries: 2
state:
  root: {(tmp_path / "projects").as_posix()}
  atomic_write: true
""",
        encoding="utf-8",
    )


def write_script(path: Path) -> None:
    responses = [submit_plan().model_dump(mode="json"), completion("c1").model_dump(mode="json")]
    items = []
    for response in responses:
        if response.get("final_answer"):
            items.append({"final_answer": response["final_answer"]["content"]})
        else:
            items.append({"tool_calls": response["tool_calls"]})
    path.write_text(json.dumps(items), encoding="utf-8")


def test_project_cli_start_status_and_tree(tmp_path: Path):
    config_path = tmp_path / "planning.yaml"
    script_path = tmp_path / "script.json"
    write_project_config(config_path, tmp_path)
    write_script(script_path)
    result = runner.invoke(
        app,
        [
            "project",
            "start",
            "--config",
            str(config_path),
            "--project-id",
            "cli-project",
            "--task",
            "ship",
            "--scripted-responses",
            str(script_path),
        ],
    )
    assert result.exit_code == 1
    assert "project_id: cli-project" in result.stdout
    status = runner.invoke(app, ["project", "status", "--config", str(config_path), "--project-id", "cli-project"])
    assert status.exit_code == 0
    assert "Project objective: ship" in status.stdout
    tree = runner.invoke(app, ["project", "tree", "--config", str(config_path), "--project-id", "cli-project"])
    assert tree.exit_code == 0
    assert "T1" in tree.stdout
