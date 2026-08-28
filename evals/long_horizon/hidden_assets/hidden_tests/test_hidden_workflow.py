"""Private integration oracle injected only into isolated verification copies."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from workflow_service.audit import AuditLog
from workflow_service.graph import ready_task_ids, validate_graph
from workflow_service.metrics import task_metrics
from workflow_service.models import Task, TaskStatus
from workflow_service.retry import deterministic_backoff
from workflow_service.serialization import task_from_dict
from workflow_service.service import WorkflowService
from workflow_service.storage import JsonTaskStore

PRIVATE_ORACLE_MARKER = "LONG_HORIZON_PRIVATE_ORACLE_V1"


def make_service(tmp_path):
    return WorkflowService(JsonTaskStore(tmp_path / "state" / "tasks.json"), AuditLog(tmp_path / "audit" / "events.jsonl"))


def cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "workflow_service.cli", "--db", str(tmp_path / "state" / "tasks.json"), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_model_rejects_boolean_version_and_counter():
    with pytest.raises(ValueError):
        Task("x", "X", version=True)
    with pytest.raises(ValueError):
        Task("x", "X", max_attempts=False)


def test_deserializer_does_not_coerce_dependency_scalar():
    with pytest.raises(ValueError):
        task_from_dict({"id": "x", "name": "X", "dependencies": "parent"})


def test_failed_import_preserves_previous_state(tmp_path):
    store = JsonTaskStore(tmp_path / "nested" / "tasks.json")
    store.save({"good": Task("good", "Good")})
    with pytest.raises(ValueError):
        store.import_json('[{"id":"bad","name":"Bad","dependencies":["missing"]}]')
    assert set(store.load()) == {"good"}


def test_atomic_store_orders_tasks_and_cleans_temporary_files(tmp_path):
    store = JsonTaskStore(tmp_path / "nested" / "tasks.json")
    store.save({"z": Task("z", "Z"), "a": Task("a", "A")})
    assert [item["id"] for item in json.loads(store.export_json())] == ["a", "z"]
    assert not list(store.path.parent.glob(f".{store.path.name}.*.tmp"))


def test_cycle_diagnostic_is_deterministic():
    tasks = {
        "c": Task("c", "C", dependencies=("a",)),
        "a": Task("a", "A", dependencies=("b",)),
        "b": Task("b", "B", dependencies=("c",)),
    }
    with pytest.raises(ValueError, match=r"dependency cycle: a -> b -> c -> a"):
        validate_graph(tasks)


def test_ready_selection_updates_after_success(tmp_path):
    service = make_service(tmp_path)
    service.create("prepare", "Prepare")
    service.create("build", "Build", dependencies=("prepare",))
    assert [task.id for task in service.ready()] == ["prepare"]
    service.run("prepare", lambda _task: "ok")
    assert [task.id for task in service.ready()] == ["build"]


def test_ready_ids_never_mutate_task_status():
    tasks = {"a": Task("a", "A")}
    assert ready_task_ids(tasks) == ["a"]
    assert tasks["a"].status == TaskStatus.PENDING


def test_retry_backoff_has_no_jitter():
    assert [deterministic_backoff(attempt, base_seconds=3, maximum_seconds=20) for attempt in range(1, 6)] == [3, 6, 12, 20, 20]


def test_failure_propagation_preserves_unrelated_branch(tmp_path):
    service = make_service(tmp_path)
    service.create("root", "Root")
    service.create("child", "Child", dependencies=("root",))
    service.create("grandchild", "Grandchild", dependencies=("child",))
    service.create("other", "Other")
    service.run("root", lambda _task: (_ for _ in ()).throw(RuntimeError("source exploded")))
    states = {task.id: task for task in service.list()}
    assert states["child"].cancel_reason == "dependency failed: root"
    assert states["grandchild"].cancel_reason == "dependency failed: root"
    assert states["other"].status == TaskStatus.PENDING


def test_cancel_replay_survives_service_reconstruction(tmp_path):
    service = make_service(tmp_path)
    service.create("a", "A")
    service.cancel("a", "operator")
    rebuilt = make_service(tmp_path)
    rebuilt.cancel("a", "operator")
    assert [event.event_type for event in rebuilt.audit.read()].count("task_cancelled") == 1


def test_create_idempotency_survives_service_reconstruction(tmp_path):
    make_service(tmp_path).create("a", "A", idempotency_key="create-42")
    replay = make_service(tmp_path).create("a", "A", idempotency_key="create-42")
    assert replay.id == "a"
    assert [event.event_type for event in make_service(tmp_path).audit.read()].count("task_created") == 1


def test_audit_sequences_cover_propagated_mutations(tmp_path):
    service = make_service(tmp_path)
    service.create("a", "A")
    service.create("b", "B", dependencies=("a",))
    service.cancel("a", "operator")
    events = service.audit.read()
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [(event.event_type, event.task_id) for event in events[-2:]] == [
        ("task_cancelled", "a"),
        ("task_cancelled", "b"),
    ]


def test_metrics_count_retries_not_first_attempts():
    tasks = {
        "a": Task("a", "A", status=TaskStatus.SUCCEEDED, attempts=1),
        "b": Task("b", "B", status=TaskStatus.FAILED, attempts=4),
    }
    assert task_metrics(tasks)["retry_count"] == 3


def test_cli_dependency_run_and_export(tmp_path):
    assert cli(tmp_path, "create", "--id", "a", "--name", "A").returncode == 0
    assert cli(tmp_path, "create", "--id", "b", "--name", "B", "--depends-on", "a").returncode == 0
    blocked = cli(tmp_path, "run", "--id", "b")
    assert blocked.returncode != 0
    assert "dependencies are not satisfied" in blocked.stderr
    assert cli(tmp_path, "run", "--id", "a").returncode == 0
    assert cli(tmp_path, "run", "--id", "b").returncode == 0
    output = tmp_path / "exports" / "tasks.json"
    assert cli(tmp_path, "export", "--output", str(output)).returncode == 0
    assert [item["id"] for item in json.loads(output.read_text(encoding="utf-8"))] == ["a", "b"]


def test_cli_import_rejects_cycle_without_replacing_state(tmp_path):
    assert cli(tmp_path, "create", "--id", "safe", "--name", "Safe").returncode == 0
    source = tmp_path / "cycle.json"
    source.write_text(
        json.dumps(
            [
                {"id": "a", "name": "A", "dependencies": ["b"]},
                {"id": "b", "name": "B", "dependencies": ["a"]},
            ]
        ),
        encoding="utf-8",
    )
    result = cli(tmp_path, "import", "--input", str(source))
    assert result.returncode != 0
    assert [item["id"] for item in json.loads(cli(tmp_path, "list").stdout)] == ["safe"]
