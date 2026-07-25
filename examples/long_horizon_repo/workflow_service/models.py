from __future__ import annotations

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
        self.status = TaskStatus(self.status)
        self.dependencies = tuple(self.dependencies)
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
