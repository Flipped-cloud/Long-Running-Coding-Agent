from __future__ import annotations

from collections.abc import Callable
from typing import Any

from workflow_service.audit import AuditLog
from workflow_service.graph import ready_task_ids, validate_graph
from workflow_service.models import Task, TaskStatus
from workflow_service.retry import can_retry
from workflow_service.storage import JsonTaskStore


class WorkflowService:
    def __init__(self, store: JsonTaskStore, audit: AuditLog | None = None):
        self.store = store
        self.audit = audit

    def create(
        self,
        task_id: str,
        name: str,
        *,
        dependencies: tuple[str, ...] = (),
        max_attempts: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        tasks = self.store.load()
        if task_id in tasks:
            raise ValueError(f"task already exists: {task_id}")
        task = Task(task_id, name, dependencies=dependencies, max_attempts=max_attempts, metadata=metadata or {})
        tasks[task.id] = task
        validate_graph(tasks)
        self.store.save(tasks)
        self._event("task_created", task.id)
        return task

    def get(self, task_id: str) -> Task:
        try:
            return self.store.load()[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task id: {task_id}") from exc

    def list(self, status: TaskStatus | None = None) -> list[Task]:
        tasks = self.store.load().values()
        return sorted((task for task in tasks if status is None or task.status == status), key=lambda task: task.id)

    def ready(self) -> list[Task]:
        tasks = self.store.load()
        return [tasks[task_id] for task_id in ready_task_ids(tasks)]

    def run(self, task_id: str, operation: Callable[[Task], Any]) -> Task:
        tasks = self.store.load()
        task = tasks[task_id]
        if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
            raise ValueError(f"cannot run task from status: {task.status.value}")
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        self._event("task_started", task.id, {"attempt": task.attempts})
        try:
            task.result = operation(task)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            self._event("task_failed", task.id, {"error": task.error})
        else:
            task.status = TaskStatus.SUCCEEDED
            task.error = None
            self._event("task_succeeded", task.id)
        self.store.save(tasks)
        return task

    def retry(self, task_id: str) -> Task:
        tasks = self.store.load()
        task = tasks[task_id]
        if task.status != TaskStatus.FAILED:
            raise ValueError("only failed tasks can be retried")
        if not can_retry(task.attempts, task.max_attempts):
            raise ValueError("retry limit reached")
        task.status = TaskStatus.PENDING
        task.error = None
        self.store.save(tasks)
        return task

    def cancel(self, task_id: str, reason: str) -> Task:
        task = self.get(task_id)
        task.status = TaskStatus.CANCELLED
        task.cancel_reason = reason
        tasks = self.store.load()
        tasks[task.id] = task
        self.store.save(tasks)
        return task

    def _event(self, event_type: str, task_id: str, details: dict[str, Any] | None = None) -> None:
        if self.audit:
            self.audit.append(event_type, task_id, details)
