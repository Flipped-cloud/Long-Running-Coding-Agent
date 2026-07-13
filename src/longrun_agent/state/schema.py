from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE_COMPLETE = "candidate_complete"
    BLOCKED = "blocked"
    FAILED = "failed"
    SESSION_LIMIT_REACHED = "session_limit_reached"
    TIME_LIMIT_REACHED = "time_limit_reached"


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    CANDIDATE_COMPLETE = "candidate_complete"
    DECOMPOSED = "decomposed"
    FAILED = "failed"


class CompletionCandidate(BaseModel):
    task_id: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    successful_tests: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=utc_now)


class TaskNode(BaseModel):
    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    depth: int = Field(default=0, ge=0)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    session_ids: list[str] = Field(default_factory=list)
    progress_notes: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    consecutive_no_progress_sessions: int = Field(default=0, ge=0)
    last_handoff_summary: str | None = None
    blocker: str | None = None
    completion_summary: str | None = None
    completion_candidate: CompletionCandidate | None = None
    auto_completion_recovered: bool = False
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_acceptance(self) -> TaskNode:
        if not self.acceptance_criteria:
            raise ValueError("task acceptance_criteria must not be empty")
        return self


class PlanRevision(BaseModel):
    revision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=utc_now)
    trigger: str
    task_id: str | None = None
    reason: str
    candidate_ids: list[str] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    added_task_ids: list[str] = Field(default_factory=list)
    superseded_task_ids: list[str] = Field(default_factory=list)


class ProjectState(BaseModel):
    project_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    status: ProjectStatus = ProjectStatus.ACTIVE
    plan_version: int = Field(default=0, ge=0)
    tasks: list[TaskNode] = Field(default_factory=list)
    active_task_id: str | None = None
    revisions: list[PlanRevision] = Field(default_factory=list)
    session_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph(self) -> ProjectState:
        ids = {task.id for task in self.tasks}
        keys = [task.key for task in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("task keys must be unique within a project")
        in_progress = [task.id for task in self.tasks if task.status == TaskStatus.IN_PROGRESS]
        if len(in_progress) > 1:
            raise ValueError("only one task may be IN_PROGRESS")
        if self.active_task_id is not None and self.active_task_id not in ids:
            raise ValueError("active_task_id must reference an existing task")
        if in_progress and self.active_task_id != in_progress[0]:
            raise ValueError("active_task_id must match the IN_PROGRESS task")
        for task in self.tasks:
            for dependency in task.dependencies:
                if dependency not in ids:
                    raise ValueError(f"dependency does not reference an existing task: {dependency}")
            if task.parent_id is not None:
                parent = next((candidate for candidate in self.tasks if candidate.id == task.parent_id), None)
                if parent is None:
                    raise ValueError(f"parent_id does not reference an existing task: {task.parent_id}")
                if task.depth != parent.depth + 1:
                    raise ValueError("task depth must be parent.depth + 1")
            elif task.depth != 0:
                raise ValueError("root task depth must be 0")
        return self

    def task_by_id(self, task_id: str) -> TaskNode:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def leaf_tasks(self) -> list[TaskNode]:
        parent_ids = {task.parent_id for task in self.tasks if task.parent_id is not None}
        return [task for task in self.tasks if task.id not in parent_ids]
