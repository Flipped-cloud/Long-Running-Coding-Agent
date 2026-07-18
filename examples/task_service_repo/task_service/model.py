from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    title: str
    completed: bool = False
    attempts: int = 0


def validate_task(task: Task) -> None:
    if not isinstance(task, Task):
        raise ValueError(f"invalid task: expected a Task instance, got {type(task).__name__}")
    if isinstance(task.attempts, bool):
        raise ValueError(
            "invalid task field 'attempts': expected an int, got bool "
            "(bool is not a valid attempt count, got True). "
            "Use 0 for no attempts, not True/False."
        )
    if task.attempts < 0:
        raise ValueError(f"invalid task field 'attempts': must be a non-negative integer, got {task.attempts}")
    if not task.id:
        raise ValueError("task id is required")
    if not task.title:
        raise ValueError("task title is required")
