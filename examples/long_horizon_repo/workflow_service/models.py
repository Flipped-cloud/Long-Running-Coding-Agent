from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: tuple[str, ...] = ()
    attempts: int = 0
    max_attempts: int = 3
    result: Any = None
    error: str | None = None
    cancel_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("task id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("task name must be a non-empty string")
        self.id = self.id.strip()
        self.name = self.name.strip()
        try:
            self.status = TaskStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid task status") from exc
        if not isinstance(self.dependencies, (tuple, list)):
            raise ValueError("dependencies must be a sequence of task ids")
        normalized_dependencies: list[str] = []
        for dependency in self.dependencies:
            if not isinstance(dependency, str) or not dependency.strip():
                raise ValueError("dependency must be a non-empty string")
            normalized_dependencies.append(dependency.strip())
        if len(normalized_dependencies) != len(set(normalized_dependencies)):
            raise ValueError("dependencies must be unique")
        self.dependencies = tuple(normalized_dependencies)
        if self.id in self.dependencies:
            raise ValueError("task cannot depend on itself")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise ValueError("attempts must be an integer")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("version must be an integer")
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("error must be a string or null")
        if self.cancel_reason is not None and not isinstance(self.cancel_reason, str):
            raise ValueError("cancel_reason must be a string or null")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        self.metadata = dict(self.metadata)
        for field_name, value in (("result", self.result), ("metadata", self.metadata)):
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be JSON serializable") from exc
