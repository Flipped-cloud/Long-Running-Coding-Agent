import json
from pathlib import Path

from longrun_agent.cli import _load_scripted_responses
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import config


def test_static_realwork_fake_script_uses_code_tools_and_records_artifacts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "task_service").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "TASK.md").write_text("Do real work", encoding="utf-8")
    (workspace / "README.md").write_text("readme", encoding="utf-8")
    (workspace / "task_service" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "task_service" / "model.py").write_text("old model", encoding="utf-8")
    (workspace / "task_service" / "storage.py").write_text("old storage", encoding="utf-8")
    (workspace / "task_service" / "service.py").write_text("old service", encoding="utf-8")
    (workspace / "task_service" / "cli.py").write_text("old cli", encoding="utf-8")
    (workspace / "tests" / "test_service.py").write_text("def test_old():\n    assert True\n", encoding="utf-8")
    (workspace / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\naddopts = "-p no:cacheprovider"\n', encoding="utf-8"
    )

    cfg = config(tmp_path, mode="static", max_sessions=2)
    cfg.workspace.root = workspace
    script = Path("examples/task_service_repo/scripted_project_static_realwork.json")
    outcome = ProjectOrchestrator(cfg, FakeModelProvider(_load_scripted_responses(script)), project_id="realwork-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    sessions = store.read_sessions("realwork-1")
    events = store.read_events("realwork-1")
    assert outcome.status == "candidate_complete"
    assert "task id must be a simple identifier" in (workspace / "task_service" / "model.py").read_text(encoding="utf-8")
    assert "test_retry_missing_task_has_explicit_error" in (workspace / "tests" / "test_service.py").read_text(encoding="utf-8")
    assert any(session["terminal_signal"] == "completion_request" for session in sessions)
    assert sum(session["tool_call_count"] for session in sessions) >= 5
    assert any(event["event_type"] == "task_progress" for event in events)
    assert json.loads(store.metrics_path("realwork-1").read_text(encoding="utf-8"))["total_tool_calls"] >= 5
