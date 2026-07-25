import pytest
from workflow_service.audit import AuditLog
from workflow_service.models import TaskStatus
from workflow_service.service import WorkflowService
from workflow_service.storage import JsonTaskStore


@pytest.fixture
def service(tmp_path):
    return WorkflowService(JsonTaskStore(tmp_path / "db" / "tasks.json"), AuditLog(tmp_path / "audit.jsonl"))


def test_create_get_and_list(service):
    service.create("b", "B")
    service.create("a", "A")
    assert service.get("a").name == "A"
    assert [task.id for task in service.list()] == ["a", "b"]


def test_get_unknown_task_is_explicit(service):
    with pytest.raises(ValueError, match="unknown task id: absent"):
        service.get("absent")


def test_create_rejects_duplicate_id(service):
    service.create("a", "A")
    with pytest.raises(ValueError, match="task already exists: a"):
        service.create("a", "Again")


def test_create_idempotency_key_replays_same_result(service):
    first = service.create("a", "A", idempotency_key="request-1")
    second = service.create("a", "A", idempotency_key="request-1")
    assert second == first
    assert len(service.list()) == 1


def test_create_idempotency_key_rejects_changed_request(service):
    service.create("a", "A", idempotency_key="request-1")
    with pytest.raises(ValueError, match="idempotency key conflict"):
        service.create("b", "B", idempotency_key="request-1")


def test_run_requires_satisfied_dependencies(service):
    service.create("prepare", "Prepare")
    service.create("build", "Build", dependencies=("prepare",))
    with pytest.raises(ValueError, match="dependencies are not satisfied"):
        service.run("build", lambda task: "done")


def test_run_success_records_result_and_audit(service):
    service.create("build", "Build")
    completed = service.run("build", lambda task: {"artifact": task.id})
    assert completed.status == TaskStatus.SUCCEEDED
    assert completed.result == {"artifact": "build"}
    assert [event.event_type for event in service.audit.read()] == ["task_created", "task_started", "task_succeeded"]


def test_run_failure_records_error(service):
    service.create("build", "Build")

    def fail(_task):
        raise RuntimeError("compiler failed")

    completed = service.run("build", fail)
    assert completed.status == TaskStatus.FAILED
    assert completed.error == "compiler failed"
    assert completed.attempts == 1


def test_retry_reopens_failed_task(service):
    service.create("build", "Build")
    service.run("build", lambda _task: (_ for _ in ()).throw(RuntimeError("failed")))
    retried = service.retry("build")
    assert retried.status == TaskStatus.PENDING
    assert retried.error is None
    assert service.audit.read()[-1].event_type == "task_retried"


def test_retry_limit_is_enforced(service):
    service.create("build", "Build", max_attempts=1)
    service.run("build", lambda _task: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(ValueError, match="retry limit reached"):
        service.retry("build")


def test_cancel_propagates_to_transitive_dependents(service):
    service.create("a", "A")
    service.create("b", "B", dependencies=("a",))
    service.create("c", "C", dependencies=("b",))
    service.cancel("a", "operator request")
    assert {task.id: task.status for task in service.list()} == {
        "a": TaskStatus.CANCELLED,
        "b": TaskStatus.CANCELLED,
        "c": TaskStatus.CANCELLED,
    }


def test_failure_propagates_to_dependents(service):
    service.create("a", "A")
    service.create("b", "B", dependencies=("a",))
    service.run("a", lambda _task: (_ for _ in ()).throw(RuntimeError("failed")))
    dependent = service.get("b")
    assert dependent.status == TaskStatus.CANCELLED
    assert dependent.cancel_reason == "dependency failed: a"


def test_cancel_is_idempotent_with_same_reason(service):
    service.create("a", "A")
    first = service.cancel("a", "operator request")
    second = service.cancel("a", "operator request")
    assert second == first
    assert [event.event_type for event in service.audit.read()].count("task_cancelled") == 1


def test_terminal_success_cannot_be_cancelled(service):
    service.create("a", "A")
    service.run("a", lambda _task: "done")
    with pytest.raises(ValueError, match="cannot cancel task from status: succeeded"):
        service.cancel("a", "too late")
