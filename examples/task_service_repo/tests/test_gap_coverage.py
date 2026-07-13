import pytest
from task_service.model import Task, validate_task


def test_validate_task_rejects_bool_attempts():
    with pytest.raises(ValueError) as exc_info:
        validate_task(Task(id="1", title="one", attempts=True))
    assert (
        str(exc_info.value)
        == "invalid task field 'attempts': expected an int, got bool (bool is not a valid attempt count, got True). Use 0 for no attempts, not True/False."
    )


def test_validate_task_rejects_negative_attempts():
    with pytest.raises(ValueError) as exc_info:
        validate_task(Task(id="1", title="one", attempts=-1))
    assert str(exc_info.value) == "invalid task field 'attempts': must be a non-negative integer, got -1"


def test_validate_task_rejects_non_task_input():
    with pytest.raises(ValueError) as exc_info:
        validate_task({"id": "1", "title": "one"})
    assert str(exc_info.value) == "invalid task: expected a Task instance, got dict"
