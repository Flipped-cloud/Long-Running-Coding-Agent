from __future__ import annotations

from typing import Any

from workflow_service.models import Task


def task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "name": task.name,
        "status": task.status.value,
        "dependencies": list(task.dependencies),
        "attempts": task.attempts,
        "max_attempts": task.max_attempts,
        "result": task.result,
        "error": task.error,
        "cancel_reason": task.cancel_reason,
        "metadata": task.metadata,
        "version": task.version,
    }


def task_from_dict(payload: dict[str, Any]) -> Task:
    return Task(
        id=payload["id"],
        name=payload["name"],
        status=payload.get("status", "pending"),
        dependencies=tuple(payload.get("dependencies", [])),
        attempts=payload.get("attempts", 0),
        max_attempts=payload.get("max_attempts", 3),
        result=payload.get("result"),
        error=payload.get("error"),
        cancel_reason=payload.get("cancel_reason"),
        metadata=payload.get("metadata", {}),
        version=payload.get("version", 0),
    )
