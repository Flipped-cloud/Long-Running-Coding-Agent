from __future__ import annotations

from workflow_service.models import Task, TaskStatus


def validate_graph(tasks: dict[str, Task]) -> None:
    for task in tasks.values():
        missing = [dependency for dependency in task.dependencies if dependency not in tasks]
        if missing:
            raise ValueError(f"task {task.id} has unknown dependencies: {', '.join(missing)}")


def ready_task_ids(tasks: dict[str, Task]) -> list[str]:
    ready = []
    for task in tasks.values():
        if task.status != TaskStatus.PENDING:
            continue
        if all(tasks[dependency].status == TaskStatus.SUCCEEDED for dependency in task.dependencies):
            ready.append(task.id)
    return ready


def dependent_ids(tasks: dict[str, Task], task_id: str) -> list[str]:
    return [task.id for task in tasks.values() if task_id in task.dependencies]
