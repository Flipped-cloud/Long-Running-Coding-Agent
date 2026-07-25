from __future__ import annotations

from collections import Counter
from typing import Any

from workflow_service.models import Task


def task_metrics(tasks: dict[str, Task]) -> dict[str, Any]:
    statuses = Counter(task.status.value for task in tasks.values())
    return {
        "total": len(tasks),
        "status_counts": dict(sorted(statuses.items())),
        "attempts": sum(task.attempts for task in tasks.values()),
    }
