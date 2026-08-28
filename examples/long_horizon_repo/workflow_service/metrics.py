from __future__ import annotations

from collections import Counter
from typing import Any

from workflow_service.models import Task, TaskStatus


def task_metrics(tasks: dict[str, Task]) -> dict[str, Any]:
    statuses = Counter(task.status.value for task in tasks.values())
    attempts = sum(task.attempts for task in tasks.values())
    terminal_statuses = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    terminal = sum(task.status in terminal_statuses for task in tasks.values())
    return {
        "total": len(tasks),
        "status_counts": dict(sorted(statuses.items())),
        "attempts": attempts,
        "terminal": terminal,
        "terminal_rate": terminal / len(tasks) if tasks else 0.0,
        "retry_count": sum(max(0, task.attempts - 1) for task in tasks.values()),
    }
