import json

import pytest
from workflow_service.models import Task, TaskStatus
from workflow_service.serialization import task_from_dict, task_to_dict
from workflow_service.storage import JsonTaskStore


def test_task_round_trip_preserves_all_fields():
    original = Task(
        "build",
        "Build",
        status=TaskStatus.FAILED,
        dependencies=("lint",),
        attempts=2,
        max_attempts=4,
        error="boom",
        metadata={"owner": "ops"},
        version=3,
    )
    assert task_from_dict(task_to_dict(original)) == original


def test_task_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown task fields"):
        task_from_dict({"id": "x", "name": "X", "surprise": True})


def test_task_from_dict_rejects_non_object():
    with pytest.raises(ValueError, match="task payload must be an object"):
        task_from_dict(["x"])


def test_result_must_be_json_serializable():
    with pytest.raises(ValueError, match="result must be JSON serializable"):
        task_to_dict(Task("x", "X", result=object()))


def test_store_missing_file_loads_empty(tmp_path):
    assert JsonTaskStore(tmp_path / "tasks.json").load() == {}


def test_store_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "data" / "tasks.json"
    JsonTaskStore(path).save({"x": Task("x", "X")})
    assert path.exists()


def test_store_writes_atomically_without_temp_residue(tmp_path):
    path = tmp_path / "tasks.json"
    store = JsonTaskStore(path)
    store.save({"x": Task("x", "X")})
    store.save({"y": Task("y", "Y")})
    assert set(store.load()) == {"y"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_rejects_duplicate_ids_in_json(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"id": "x", "name": "X"}, {"id": "x", "name": "Other"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        JsonTaskStore(path).load()


def test_import_export_round_trip_is_canonical(tmp_path):
    source = JsonTaskStore(tmp_path / "source.json")
    source.save({"b": Task("b", "B"), "a": Task("a", "A")})
    exported = source.export_json()
    target = JsonTaskStore(tmp_path / "target.json")
    target.import_json(exported)
    assert target.export_json() == exported
