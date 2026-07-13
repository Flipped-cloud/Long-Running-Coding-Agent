from __future__ import annotations

from task_service.model import Task, validate_task


class TaskService:
    def __init__(self, tasks: list[Task] | None = None):
        self.tasks = tasks or []

    def add(self, task: Task) -> None:
        validate_task(task)
        if any(existing.id == task.id for existing in self.tasks):
            raise ValueError("duplicate task id")
        self.tasks.append(task)

    def complete(self, task_id: str) -> bool:
        for task in self.tasks:
            if task.id == task_id:
                task.completed = True
                return True
        return False

    def retry(self, task_id: str) -> int:
        for task in self.tasks:
            if task.id == task_id:
                task.attempts += 1
                return task.attempts
        raise ValueError(f"unknown task id: {task_id}")
