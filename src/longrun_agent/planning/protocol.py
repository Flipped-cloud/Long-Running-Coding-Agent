from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskDraft(BaseModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on_keys: list[str] = Field(default_factory=list)


class ChildTaskDraft(BaseModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on_child_keys: list[str] = Field(default_factory=list)


class RecoveryCandidate(BaseModel):
    id: str = Field(min_length=1)
    kind: Literal["retry_with_guidance", "decompose", "mark_blocked"]
    description: str
    rationale: str
    expected_benefit: str
    risks: str
    testability: str
    child_tasks: list[ChildTaskDraft] = Field(default_factory=list)


class CandidateScore(BaseModel):
    candidate_id: str
    feasibility: int = Field(ge=1, le=5)
    testability: int = Field(ge=1, le=5)
    scope_control: int = Field(ge=1, le=5)
    recovery_value: int = Field(ge=1, le=5)


class RecoverySelection(BaseModel):
    selected_candidate_id: str
    scores: list[CandidateScore]
    selection_reason: str
