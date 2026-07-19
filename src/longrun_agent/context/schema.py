from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ContextMode(StrEnum):
    FULL_HISTORY = "full_history"
    RECENT_WINDOW = "recent_window"
    DETERMINISTIC_PRUNE = "deterministic_prune"
    STRUCTURED_RESET = "structured_reset"


class ContextItemStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    COMPACTED = "compacted"
    EXTERNALIZED = "externalized"


class ContextItemKind(StrEnum):
    SYSTEM = "system"
    TASK_ANCHOR = "task_anchor"
    HANDOFF = "handoff"
    ASSISTANT_TOOL_CALL = "assistant_tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT_FINAL = "assistant_final"
    USER_REMINDER = "user_reminder"
    PROTOCOL_CORRECTION = "protocol_correction"
    CURRENT_INSTRUCTION = "current_instruction"
    EPISODIC_MEMORY = "episodic_memory"
    PROCEDURAL_SKILL = "procedural_skill"


class ContextBudgetSnapshot(BaseModel):
    estimated_message_tokens: int = Field(ge=0)
    estimated_tool_schema_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    estimated_total_tokens: int = Field(ge=0)
    context_limit: int = Field(gt=0)
    usage_ratio: float = Field(ge=0)
    trigger_exceeded: bool
    hard_stop_exceeded: bool
    estimator_type: str
    timestamp: str = Field(default_factory=utc_now)


class ContextSegmentRecord(BaseModel):
    segment_id: int = Field(ge=1)
    run_id: str
    project_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    start_step: int = Field(default=0, ge=0)
    end_step: int | None = Field(default=None, ge=0)
    reset_reason: str | None = None
    handoff_id: str | None = None
    estimated_tokens_at_close: int = Field(default=0, ge=0)
    actual_input_tokens_max: int = Field(default=0, ge=0)
    pruned_item_count: int = Field(default=0, ge=0)
    stale_item_count: int = Field(default=0, ge=0)
    compacted_item_count: int = Field(default=0, ge=0)


class TaskContextSeed(BaseModel):
    project_id: str | None = None
    project_objective: str
    plan_version: int = Field(ge=0)
    task_id: str
    task_title: str
    task_objective: str
    task_status: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependency_summaries: list[str] = Field(default_factory=list)
    blocker: str | None = None
    attempts: int = Field(ge=0)
    progress_summary: str | None = None
    files_touched: list[str] = Field(default_factory=list)
    latest_handoff_id: str | None = None
    knowledge_context: str | None = None
    knowledge_retrieval_id: str | None = None


class ContextPreparationAction(StrEnum):
    UNCHANGED = "unchanged"
    PRUNED = "pruned"
    RESET = "reset"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ContextPreparationResult(BaseModel):
    action: ContextPreparationAction
    messages: list[dict]
    budget_before: ContextBudgetSnapshot
    budget_after: ContextBudgetSnapshot
    pruned: bool = False
    handoff_id: str | None = None
    reset_performed: bool = False
    segment_id: int = Field(ge=1)
    hard_stop_reason: str | None = None


class PruneReport(BaseModel):
    estimated_tokens_before: int = Field(ge=0)
    estimated_tokens_after: int = Field(ge=0)
    tokens_removed: int = Field(ge=0)
    stale_items_removed: int = Field(default=0, ge=0)
    superseded_items_removed: int = Field(default=0, ge=0)
    compacted_items: int = Field(default=0, ge=0)
    items_removed: int = Field(default=0, ge=0)


class ConfirmedFact(BaseModel):
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_step: int = Field(ge=0)


class WorkingHypothesis(BaseModel):
    statement: str = Field(min_length=1)
    basis: str = Field(default="")
    unverified: Literal[True] = True
    source_step: int = Field(ge=0)


class FileState(BaseModel):
    path: str = Field(min_length=1)
    sha256: str | None = None
    status: str = "current"
    summary: str = ""
    last_write_step: int | None = Field(default=None, ge=0)
    evidence_id: str | None = None


class VerificationState(BaseModel):
    command: str
    exit_code: int
    result: str
    code_epoch: int = Field(ge=0)
    artifact_path: str | None = None
    source_step: int = Field(ge=0)
    stale: bool = False


class UnresolvedError(BaseModel):
    summary: str
    source_step: int = Field(ge=0)


class NextAction(BaseModel):
    action: str
    rationale: str = ""


class EvidenceReference(BaseModel):
    evidence_id: str
    type: str
    event_path: str | None = None
    artifact_path: str | None = None
    step: int = Field(ge=0)
    description: str = ""


class HandoffRecord(BaseModel):
    handoff_id: str
    project_id: str | None = None
    task_id: str
    session_id: str
    source_segment_id: int = Field(ge=1)
    target_segment_id: int = Field(ge=1)
    plan_version: int = Field(ge=0)
    task_objective: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    confirmed_progress: list[ConfirmedFact] = Field(default_factory=list)
    files_current: list[FileState] = Field(default_factory=list)
    verifications: list[VerificationState] = Field(default_factory=list)
    unresolved_errors: list[UnresolvedError] = Field(default_factory=list)
    hypotheses: list[WorkingHypothesis] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    stale_items_excluded: list[str] = Field(default_factory=list)
    source_steps: list[int] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    generator: Literal["model", "deterministic"] = "deterministic"
    schema_version: str = "context_handoff.v1"

    @model_validator(mode="after")
    def validate_evidence(self) -> HandoffRecord:
        evidence_ids = {item.evidence_id for item in self.evidence_references}
        for fact in self.confirmed_progress:
            missing = [item for item in fact.evidence_ids if item not in evidence_ids]
            if missing:
                raise ValueError(f"confirmed_progress references unknown evidence ids: {missing}")
        hypothesis_statements = {item.statement.strip().lower() for item in self.hypotheses}
        for fact in self.confirmed_progress:
            if fact.statement.strip().lower() in hypothesis_statements:
                raise ValueError("hypothesis cannot be promoted into confirmed_progress")
        return self
