from __future__ import annotations

from workflow_service.models import Task, TaskStatus


def validate_graph(tasks: dict[str, Task]) -> None:
    for task in sorted(tasks.values(), key=lambda item: item.id):
        missing = [dependency for dependency in task.dependencies if dependency not in tasks]
        if missing:
            raise ValueError(f"task {task.id} has unknown dependencies: {', '.join(sorted(missing))}")

    visited: set[str] = set()
    active: list[str] = []

    def visit(task_id: str) -> None:
        if task_id in active:
            start = active.index(task_id)
            raise ValueError("dependency cycle: " + " -> ".join(active[start:] + [task_id]))
        if task_id in visited:
            return
        active.append(task_id)
        for dependency in sorted(tasks[task_id].dependencies):
            visit(dependency)
        active.pop()
        visited.add(task_id)

    for task_id in sorted(tasks):
        visit(task_id)


def ready_task_ids(tasks: dict[str, Task]) -> list[str]:
    ready = []
    for task in tasks.values():
        if task.status != TaskStatus.PENDING:
            continue
        if all(tasks[dependency].status == TaskStatus.SUCCEEDED for dependency in task.dependencies):
            ready.append(task.id)
    return sorted(ready)


def dependent_ids(tasks: dict[str, Task], task_id: str) -> list[str]:
    return sorted(task.id for task in tasks.values() if task_id in task.dependencies)
