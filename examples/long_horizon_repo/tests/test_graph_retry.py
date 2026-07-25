import pytest
from workflow_service.graph import dependent_ids, ready_task_ids, validate_graph
from workflow_service.models import Task, TaskStatus
from workflow_service.retry import can_retry, deterministic_backoff


def test_graph_rejects_unknown_dependencies():
    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_graph({"build": Task("build", "Build", dependencies=("missing",))})


def test_graph_rejects_direct_cycle():
    tasks = {
        "a": Task("a", "A", dependencies=("b",)),
        "b": Task("b", "B", dependencies=("a",)),
    }
    with pytest.raises(ValueError, match="dependency cycle: a -> b -> a"):
        validate_graph(tasks)


def test_graph_rejects_long_cycle():
    tasks = {
        "a": Task("a", "A", dependencies=("c",)),
        "b": Task("b", "B", dependencies=("a",)),
        "c": Task("c", "C", dependencies=("b",)),
    }
    with pytest.raises(ValueError, match="dependency cycle"):
        validate_graph(tasks)


def test_ready_tasks_are_dependency_aware_and_sorted():
    tasks = {
        "z": Task("z", "Z"),
        "a": Task("a", "A"),
        "build": Task("build", "Build", dependencies=("a",)),
    }
    assert ready_task_ids(tasks) == ["a", "z"]
    tasks["a"].status = TaskStatus.SUCCEEDED
    assert ready_task_ids(tasks) == ["build", "z"]


def test_terminal_tasks_are_not_ready():
    tasks = {
        status.value: Task(status.value, status.value, status=status)
        for status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)
    }
    assert ready_task_ids(tasks) == []


def test_dependents_are_sorted():
    tasks = {
        "root": Task("root", "Root"),
        "z": Task("z", "Z", dependencies=("root",)),
        "a": Task("a", "A", dependencies=("root",)),
    }
    assert dependent_ids(tasks, "root") == ["a", "z"]


@pytest.mark.parametrize(("attempt", "expected"), [(1, 2), (2, 4), (3, 8), (6, 60)])
def test_deterministic_backoff_is_exponential_and_capped(attempt, expected):
    assert deterministic_backoff(attempt) == expected


@pytest.mark.parametrize("attempt", [0, -1, True])
def test_deterministic_backoff_rejects_invalid_attempt(attempt):
    with pytest.raises(ValueError, match="attempt must be a positive integer"):
        deterministic_backoff(attempt)


def test_retry_boundary_is_strict():
    assert can_retry(2, 3)
    assert not can_retry(3, 3)
