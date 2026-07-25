import pytest
from workflow_service.models import Task, TaskStatus


def test_task_defaults_are_stable():
    task = Task("build", "Build package")
    assert task.status == TaskStatus.PENDING
    assert task.attempts == 0
    assert task.max_attempts == 3


def test_task_strips_identity_fields():
    task = Task("  build  ", "  Build package  ")
    assert (task.id, task.name) == ("build", "Build package")


@pytest.mark.parametrize("value", ["", "   ", None, 12])
def test_task_rejects_invalid_id(value):
    with pytest.raises(ValueError, match="task id"):
        Task(value, "name")


def test_task_rejects_boolean_attempts():
    with pytest.raises(ValueError, match="attempts must be an integer"):
        Task("build", "Build", attempts=True)


def test_task_rejects_duplicate_dependencies():
    with pytest.raises(ValueError, match="dependencies must be unique"):
        Task("build", "Build", dependencies=("lint", "lint"))


def test_task_rejects_self_dependency():
    with pytest.raises(ValueError, match="cannot depend on itself"):
        Task("build", "Build", dependencies=("build",))


def test_task_rejects_non_mapping_metadata():
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        Task("build", "Build", metadata=[])


def test_status_values_are_public_strings():
    assert [status.value for status in TaskStatus] == [
        "pending",
        "ready",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    ]
