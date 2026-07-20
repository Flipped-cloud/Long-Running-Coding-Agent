import os
from pathlib import Path

import pytest

from longrun_agent.exceptions import ConfigurationError
from longrun_agent.state.schema import ProjectState, TaskNode
from longrun_agent.state.store import ProjectStateStore


def state() -> ProjectState:
    return ProjectState(
        project_id="p1",
        objective="ship",
        tasks=[TaskNode(id="t1", key="T1", title="T1", objective="do it", acceptance_criteria=["done"])],
    )


def test_state_store_create_load_and_list(tmp_path: Path):
    store = ProjectStateStore(tmp_path / "projects")
    store.create(state())
    loaded = store.load("p1")
    assert loaded.objective == "ship"
    assert store.exists("p1")
    assert store.list_projects() == ["p1"]


def test_state_store_save_failure_keeps_previous_file(tmp_path: Path, monkeypatch):
    store = ProjectStateStore(tmp_path / "projects")
    current = state()
    store.create(current)
    original = store.state_path("p1").read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("longrun_agent.state.store.os.replace", fail_replace)
    current.objective = "changed"
    with pytest.raises(OSError):
        store.save(current)
    assert store.state_path("p1").read_text(encoding="utf-8") == original


def test_state_store_rejects_root_inside_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ConfigurationError):
        ProjectStateStore(workspace / ".runs" / "projects", workspace_root=workspace)


def test_state_store_retries_transient_permission_error(tmp_path: Path, monkeypatch):
    store = ProjectStateStore(tmp_path / "projects")
    actual_replace = os.replace
    calls = 0

    def transient_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient Windows file lock")
        actual_replace(src, dst)

    monkeypatch.setattr("longrun_agent.state.store.os.replace", transient_replace)

    store.create(state())

    assert calls == 3
    assert store.load("p1").objective == "ship"
