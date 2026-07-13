from pathlib import Path

import pytest
from task_service.model import Task
from task_service.service import TaskService
from task_service.storage import load_tasks, save_tasks


def test_add_rejects_duplicate_id():
    service = TaskService()
    service.add(Task(id="1", title="one"))
    with pytest.raises(ValueError):
        service.add(Task(id="1", title="again"))


def test_json_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "tasks.json"
    save_tasks(path, [Task(id="1", title="one")])
    assert load_tasks(path)[0].title == "one"


def test_json_persistence_nested_directory(tmp_path: Path):
    path = tmp_path / "nested" / "subdir" / "tasks.json"
    save_tasks(path, [Task(id="1", title="one")])
    assert path.exists()
    assert load_tasks(path)[0].title == "one"


def test_retry_counts_attempts():
    service = TaskService([Task(id="1", title="one")])
    assert service.retry("1") == 1


def test_retry_unknown_task():
    service = TaskService([Task(id="1", title="one")])
    with pytest.raises(ValueError, match="unknown task id: missing"):
        service.retry("missing")
