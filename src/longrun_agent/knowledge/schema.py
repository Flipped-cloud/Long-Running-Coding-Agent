from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def skill_derivation_key(source_memory_ids: list[str]) -> str:
    source_ids = sorted(set(source_memory_ids))
    if not source_ids:
        return ""
    payload = json.dumps(source_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def skill_semantic_signature(record: SkillRecord) -> str:
    payload = {
        "title": record.title,
        "summary": record.summary,
        "procedure": [step.model_dump(mode="json") for step in record.procedure],
        "verification": record.verification,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MemoryKind(StrEnum):
    FAILURE_REFLECTION = "failure_reflection"
    SUCCESSFUL_LESSON = "successful_lesson"
    REPOSITORY_CONSTRAINT = "repository_constraint"
    TOOL_USAGE_LESSON = "tool_usage_lesson"


class MemoryScope(StrEnum):
    TASK = "task"
    PROJECT = "project"
    REPOSITORY = "repository"
    PORTABLE = "portable"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    QUARANTINED = "quarantined"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class SkillStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class KnowledgeUseType(StrEnum):
    RETRIEVED = "retrieved"
    EXPOSED = "exposed"
    REFERENCED = "referenced"
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


class EpisodeOutcome(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    NO_PROGRESS = "no_progress"
    FAILED = "failed"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


class ExperienceEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    project_id: str
    task_id: str
    session_id: str
    run_id: str
    context_segment_id: int = Field(default=1, ge=1)
    step: int = Field(default=0, ge=0)
    event_type: str
    tool_name: str | None = None
    command: str | None = None
    path: str | None = None
    summary: str = ""
    success: bool | None = None
    exit_code: int | None = None
    output_excerpt: str | None = None
    error_type: str | None = None
    artifact_path: str | None = None
    file_sha256: str | None = None
    code_epoch: int | None = None
    timestamp: str = Field(default_factory=utc_now)


class ExperienceEvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(default_factory=lambda: new_id("EP"))
    project_id: str
    task_id: str
    task_objective: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    session_id: str
    run_id: str
    plan_version: int = Field(ge=0)
    repository_fingerprint: str = ""
    language_tags: list[str] = Field(default_factory=list)
    framework_tags: list[str] = Field(default_factory=list)
    tool_tags: list[str] = Field(default_factory=list)
    starting_task_status: str = ""
    ending_task_status: str = ""
    terminal_signal: str | None = None
    run_status: str = ""
    progress_made: bool = False
    no_progress: bool = False
    files_read: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    failed_verifications: list[str] = Field(default_factory=list)
    successful_verifications: list[str] = Field(default_factory=list)
    repeated_actions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    context_reset_count: int = Field(default=0, ge=0)
    evidence_items: list[ExperienceEvidenceItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> ExperienceEvidencePack:
        evidence_ids = [item.evidence_id for item in self.evidence_items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("experience evidence ids must be unique")
        return self

    def evidence_by_id(self) -> dict[str, ExperienceEvidenceItem]:
        return {item.evidence_id: item for item in self.evidence_items}


FailureCategory = Literal[
    "incorrect_repository_assumption",
    "stale_state_usage",
    "incomplete_verification",
    "wrong_tool_usage",
    "environment_setup",
    "dependency_mismatch",
    "repeated_action_loop",
    "scope_violation",
    "implementation_error",
    "planning_error",
    "unknown",
]


class ReflectionCandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_category: FailureCategory = "unknown"
    root_cause: str = Field(min_length=1)
    incorrect_assumptions: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    avoid_actions: list[str] = Field(default_factory=list)
    applicability_conditions: list[str] = Field(default_factory=list)
    anti_conditions: list[str] = Field(default_factory=list)
    proposed_scope: MemoryScope = MemoryScope.REPOSITORY
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ReflectionCandidate(ReflectionCandidateDraft):
    candidate_id: str = Field(default_factory=lambda: new_id("REF"))
    episode_id: str
    created_at: str = Field(default_factory=utc_now)


class EvidenceGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = False
    decision: Literal["accepted", "quarantined", "rejected"] = "rejected"
    rejection_reasons: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    validated_evidence_ids: list[str] = Field(default_factory=list)
    final_scope: MemoryScope | None = None
    final_confidence: float = Field(default=0.0, ge=0, le=1)


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default_factory=lambda: new_id("MEM"))
    kind: MemoryKind
    scope: MemoryScope
    status: MemoryStatus = MemoryStatus.CANDIDATE
    title: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    recommended_actions: list[str] = Field(default_factory=list)
    avoid_actions: list[str] = Field(default_factory=list)
    applicability_conditions: list[str] = Field(default_factory=list)
    anti_conditions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    project_id: str | None = None
    repository_fingerprint: str | None = None
    source_episode_ids: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)
    source_session_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    expires_at: str | None = None
    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None
    retrieval_count: int = Field(default=0, ge=0)
    exposure_count: int = Field(default=0, ge=0)
    reference_count: int = Field(default=0, ge=0)
    helpful_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    last_retrieved_at: str | None = None
    last_used_at: str | None = None


class SkillStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    tool_name: str | None = None
    command: str | None = None


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(default_factory=lambda: new_id("SKILL"))
    version: int = Field(default=1, ge=1)
    status: SkillStatus = SkillStatus.DRAFT
    scope: MemoryScope = MemoryScope.REPOSITORY
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    preconditions: list[str] = Field(min_length=1)
    anti_conditions: list[str] = Field(min_length=1)
    parameters: list[str] = Field(default_factory=list)
    procedure: list[SkillStep] = Field(min_length=1)
    verification: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    repository_fingerprint: str | None = None
    source_memory_ids: list[str] = Field(default_factory=list)
    derivation_key: str = ""
    semantic_signature: str = ""
    source_episode_ids: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    distinct_task_ids: list[str] = Field(default_factory=list)
    distinct_repository_fingerprints: list[str] = Field(default_factory=list)
    previous_version_id: str | None = None
    next_version_id: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    retrieval_count: int = Field(default=0, ge=0)
    exposure_count: int = Field(default=0, ge=0)
    reference_count: int = Field(default=0, ge=0)
    helpful_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    last_retrieved_at: str | None = None
    last_used_at: str | None = None

    @model_validator(mode="after")
    def validate_safe_procedure(self) -> SkillRecord:
        expected_derivation_key = skill_derivation_key(self.source_memory_ids)
        if expected_derivation_key:
            self.derivation_key = expected_derivation_key
        self.semantic_signature = skill_semantic_signature(self)
        dangerous = {"rm", "del", "format", "shutdown", "mkfs"}
        for step in self.procedure:
            command = (step.command or "").strip().lower()
            first = command.split(maxsplit=1)[0] if command else ""
            if first in dangerous:
                raise ValueError(f"dangerous command is not allowed in skill: {first}")
        return self


class SkillCandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    scope: MemoryScope = MemoryScope.REPOSITORY
    preconditions: list[str] = Field(min_length=1)
    anti_conditions: list[str] = Field(min_length=1)
    parameters: list[str] = Field(default_factory=list)
    procedure: list[SkillStep] = Field(min_length=1)
    verification: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class KnowledgeRetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_objective: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    repository_fingerprint: str = ""
    language_tags: list[str] = Field(default_factory=list)
    framework_tags: list[str] = Field(default_factory=list)
    tool_tags: list[str] = Field(default_factory=list)
    project_id: str | None = None
    blocker: str | None = None
    recent_error_signatures: list[str] = Field(default_factory=list)


class RetrievalScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    total_score: float
    lexical_score: float = 0.0
    tag_score: float = 0.0
    scope_score: float = 0.0
    evidence_score: float = 0.0
    utility_score: float = 0.0
    freshness_score: float = 0.0
    conflict_penalty: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class RetrievedKnowledgeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryRecord] = Field(default_factory=list)
    skills: list[SkillRecord] = Field(default_factory=list)
    primary_memory_ids: list[str] = Field(default_factory=list)
    primary_skill_ids: list[str] = Field(default_factory=list)
    shadowed_memory_ids: list[str] = Field(default_factory=list)
    provenance_links: dict[str, list[str]] = Field(default_factory=dict)
    total_estimated_tokens: int = Field(default=0, ge=0)
    retrieval_id: str = Field(default_factory=lambda: new_id("RET"))
    created_at: str = Field(default_factory=utc_now)


class KnowledgeUseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_id: str = Field(default_factory=lambda: new_id("USE"))
    project_id: str
    task_id: str
    session_id: str
    retrieval_id: str | None = None
    memory_id: str | None = None
    skill_id: str | None = None
    use_type: KnowledgeUseType
    reason: str = ""
    timestamp: str = Field(default_factory=utc_now)


class KnowledgeConflictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_id: str = Field(default_factory=lambda: new_id("CONF"))
    left_id: str
    right_id: str
    conflict_type: str
    severity: Literal["low", "medium", "high"] = "medium"
    reason: str
    timestamp: str = Field(default_factory=utc_now)
