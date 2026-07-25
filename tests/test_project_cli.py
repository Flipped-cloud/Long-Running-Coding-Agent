import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import _load_project_resume_config, app
from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus
from longrun_agent.state.store import ProjectStateStore
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
    state = ProjectStateStore(tmp_path / "projects").load("cli-project")
    assert state.workspace_root == str((tmp_path / "workspace").resolve())
    status = runner.invoke(app, ["project", "status", "--config", str(config_path), "--project-id", "cli-project"])
    assert status.exit_code == 0
    assert "Project objective: ship" in status.stdout
    tree = runner.invoke(app, ["project", "tree", "--config", str(config_path), "--project-id", "cli-project"])
    assert tree.exit_code == 0
    assert "T1" in tree.stdout


def test_project_resume_recovers_legacy_runtime_paths_from_telemetry(tmp_path: Path):
    config_path = tmp_path / "planning.yaml"
    write_project_config(config_path, tmp_path)
    wrong_workspace = tmp_path / "wrong-workspace"
    wrong_workspace.mkdir()
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f"root: {(tmp_path / 'workspace').as_posix()}",
            f"root: {wrong_workspace.as_posix()}",
            1,
        ),
        encoding="utf-8",
    )

    correct_workspace = tmp_path / "runs" / "workspaces" / "legacy-project"
    correct_workspace.mkdir(parents=True)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text('{"tasks": []}', encoding="utf-8")
    store = ProjectStateStore(tmp_path / "projects")
    store.create(
        ProjectState(
            project_id="legacy-project",
            objective="ship",
            tasks=[
                TaskNode(
                    id="legacy-project:T1",
                    key="T1",
                    title="T1",
                    objective="fix",
                    acceptance_criteria=["done"],
                    status=TaskStatus.FAILED,
                    consecutive_no_progress_sessions=4,
                )
            ],
        )
    )
    events_dir = tmp_path / "runs" / "legacy-project-s1"
    events_dir.mkdir(parents=True)
    (events_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "run_started",
                "payload": {
                    "workspace": str(correct_workspace),
                    "config": {"planning": {"initial_plan": {"plan_file": str(plan_file)}}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = _load_project_resume_config(config_path, "legacy-project")

    migrated = store.load("legacy-project")
    assert config.workspace.root == correct_workspace.resolve()
    assert config.planning.initial_plan.plan_file == plan_file.resolve()
    assert migrated.workspace_root == str(correct_workspace.resolve())
    assert migrated.initial_plan_file == str(plan_file.resolve())
    migrated_task = migrated.task_by_id("legacy-project:T1")
    assert migrated_task.consecutive_no_progress_sessions == 0
    assert "Runtime path recovery" in migrated_task.last_handoff_summary
