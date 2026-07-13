from pathlib import Path

import pytest

from longrun_agent.exceptions import StateStoreError
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.state.schema import TaskStatus
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import config, final, submit_plan


def test_resume_after_interrupted_in_progress_state_keeps_state_parseable(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    ProjectOrchestrator(cfg, FakeModelProvider([submit_plan(), final()]), project_id="crash-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("crash-1")
    assert state.task_by_id("crash-1:T1").status == TaskStatus.IN_PROGRESS
    assert store.state_path("crash-1").read_text(encoding="utf-8").strip().endswith("}")

    resumed = ProjectOrchestrator(cfg, FakeModelProvider([])).resume("crash-1")
    assert resumed.project_id == "crash-1"
    events = store.read_events("crash-1")
    assert sum(1 for event in events if event["event_type"] == "project_created") == 1
    assert any(event["event_type"] == "project_resumed" for event in events)


def test_corrupt_project_state_reports_clear_error(tmp_path: Path):
    store = ProjectStateStore(tmp_path / "projects")
    project_dir = store.project_dir("bad")
    project_dir.mkdir(parents=True)
    store.state_path("bad").write_text("{bad json", encoding="utf-8")
    with pytest.raises(StateStoreError, match="project state is not readable JSON"):
        store.load("bad")


def test_incomplete_sessions_jsonl_reports_clear_error(tmp_path: Path):
    store = ProjectStateStore(tmp_path / "projects")
    project_dir = store.project_dir("bad-sessions")
    project_dir.mkdir(parents=True)
    store.sessions_path("bad-sessions").write_text('{"ok": true}\n{"broken"', encoding="utf-8")
    with pytest.raises(StateStoreError, match="invalid JSONL"):
        store.read_sessions("bad-sessions")
