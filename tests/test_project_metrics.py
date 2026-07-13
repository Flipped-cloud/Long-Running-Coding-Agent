import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_cli import write_project_config, write_script
from tests.test_project_orchestrator import completion, config, submit_plan


def test_project_metrics_are_derived_from_sessions(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=2)
    responses = [
        submit_plan(),
        ModelResponse(tool_calls=[ToolCall(id="p1", name="report_progress", arguments={"summary": "touched", "files_touched": ["a.py"]})]),
        completion("c1"),
        completion("c2"),
    ]
    ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="metrics-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    sessions = store.read_sessions("metrics-1")
    metrics = json.loads(store.metrics_path("metrics-1").read_text(encoding="utf-8"))
    assert metrics["project_status"] == "candidate_complete"
    assert metrics["project_sessions"] == len(sessions) == 2
    assert metrics["total_tool_calls"] == sum(session["tool_call_count"] for session in sessions)
    assert metrics["total_tokens"] == sum(session["total_tokens"] for session in sessions)
    assert metrics["sessions_without_terminal_signal"] == sum(1 for session in sessions if not session["terminal_signal"])
    assert {"project_id", "task_id", "session_id", "run_id", "task_attempt", "duration_seconds", "files_touched"}.issubset(sessions[0])
    assert {
        "wall_clock_seconds",
        "configured_max_project_seconds",
        "time_budget_exhausted",
        "tasks_failed",
        "no_progress_sessions",
        "repeated_tool_calls",
        "changed_file_count",
        "successful_test_command_count",
        "final_verification_exit_code",
        "final_verification_passed",
        "terminal_grace_turn_count",
        "terminal_signal_recovered_count",
        "unsupported_shell_syntax_count",
        "tool_argument_protocol_retry_count",
        "tasks_completed_after_grace_turn",
        "auto_completion_recovered_count",
    }.issubset(metrics)


def test_project_metrics_cli_outputs_metrics_json(tmp_path: Path):
    runner = CliRunner()
    config_path = tmp_path / "planning.yaml"
    script_path = tmp_path / "script.json"
    write_project_config(config_path, tmp_path)
    write_script(script_path)
    runner.invoke(
        app,
        [
            "project",
            "start",
            "--config",
            str(config_path),
            "--project-id",
            "metrics-cli",
            "--task",
            "ship",
            "--scripted-responses",
            str(script_path),
        ],
    )
    result = runner.invoke(app, ["project", "metrics", "--config", str(config_path), "--project-id", "metrics-cli"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["project_sessions"] == 1
