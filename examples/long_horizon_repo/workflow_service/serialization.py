from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from workflow_service.models import Task

_TASK_FIELDS = {
    "id",
    "name",
    "status",
    "dependencies",
    "attempts",
    "max_attempts",
    "result",
    "error",
    "cancel_reason",
    "metadata",
    "version",
}


def task_to_dict(task: Task) -> dict[str, Any]:
    payload = {
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
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("result must be JSON serializable") from exc
    return payload


def task_from_dict(payload: dict[str, Any]) -> Task:
    if not isinstance(payload, Mapping):
        raise ValueError("task payload must be an object")
    unknown = set(payload) - _TASK_FIELDS
    if unknown:
        raise ValueError(f"unknown task fields: {', '.join(sorted(unknown))}")
    dependencies = payload.get("dependencies", [])
    if not isinstance(dependencies, (list, tuple)):
        raise ValueError("dependencies must be a sequence of task ids")
    return Task(
        id=payload["id"],
        name=payload["name"],
        status=payload.get("status", "pending"),
        dependencies=tuple(dependencies),
        attempts=payload.get("attempts", 0),
        max_attempts=payload.get("max_attempts", 3),
        result=payload.get("result"),
        error=payload.get("error"),
        cancel_reason=payload.get("cancel_reason"),
        metadata=payload.get("metadata", {}),
        version=payload.get("version", 0),
    )
