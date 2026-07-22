from __future__ import annotations

import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from longrun_agent.state.schema import utc_now

EVALUATION_SEMANTICS_VERSION = "v0.5.5-sandbox-attribution"


class TrialStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class TerminationReason(StrEnum):
    COMPLETED = "completed"
    CONTEXT_LIMIT = "context_limit"
    INVALID_FORMAT = "invalid_format"
    INVALID_ACTION = "invalid_action"
    TASK_LIMIT = "task_limit"
    TIME_LIMIT = "time_limit"
    SESSION_LIMIT = "session_limit"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    ENVIRONMENT_ERROR = "environment_error"
    VERIFICATION_INCONCLUSIVE = "verification_inconclusive"
    CONTRACT_INVALID = "contract_invalid"
    UNKNOWN = "unknown"


class FailureLayer(StrEnum):
    ENVIRONMENT = "environment"
    PROVIDER_PROTOCOL = "provider_protocol"
    TOOL_ACI = "tool_aci"
    PLANNING = "planning"
    STATE = "state"
    CONTEXT = "context"
    KNOWLEDGE = "knowledge"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    COMPLETION_CONTROL = "completion_control"
    RECOVERY = "recovery"
    RESOURCE_BUDGET = "resource_budget"
    UNKNOWN = "unknown"


class EvaluationTaskCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    adapter: str = "local_project"
    fixture: Path | None = None
    task_file: Path | None = None
    contract_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentConfigReference(BaseModel):
    config_id: str
    path: Path
    mode: str = "default"


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(default_factory=lambda: f"EVAL-{uuid.uuid4().hex[:12]}")
    task_cases: list[EvaluationTaskCase]
    agent_configs: list[AgentConfigReference]
    trial_count: int = Field(default=1, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0])
    budgets: dict[str, int | float] = Field(default_factory=dict)
    reset_policy: str = "per_trial"
    verification_policy: str = "contract"
    output_root: Path
    shared_knowledge: bool = False
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def accept_task_condition_manifest(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "tasks" in migrated and "task_cases" not in migrated:
            migrated["task_cases"] = [
                item
                if isinstance(item, dict)
                else {
                    "case_id": item,
                    "fixture": f"../../examples/verification_bench/{item}",
                    "task_file": "TASK.md",
                    "contract_path": f"../../examples/verification_bench/contracts/{item}.yaml",
                }
                for item in migrated.pop("tasks")
            ]
        if "conditions" in migrated and "agent_configs" not in migrated:
            migrated["agent_configs"] = [
                item
                if isinstance(item, dict)
                else {
                    "config_id": item,
                    "mode": item,
                    "path": f"../../configs/{item}.yaml",
                }
                for item in migrated.pop("conditions")
            ]
        if "trials" in migrated and "trial_count" not in migrated:
            migrated["trial_count"] = migrated.pop("trials")
        return migrated

    @model_validator(mode="after")
    def validate_manifest(self) -> EvaluationManifest:
        if not self.task_cases or not self.agent_configs:
            raise ValueError("evaluation manifest requires task cases and agent configs")
        if not self.seeds:
            raise ValueError("evaluation manifest requires at least one seed")
        return self


class TrialDescriptor(BaseModel):
    evaluation_id: str
    case_id: str
    config_id: str
    trial_id: str
    trial_number: int
    seed: int
    trial_dir: Path
    config_mode: str = "default"
    shared_knowledge_root: Path | None = None
    status: TrialStatus = TrialStatus.PENDING


class VerificationMilestone(BaseModel):
    milestone_id: str
    title: str
    weight: float = Field(gt=0)
    check_ids: list[str] = Field(default_factory=list)
    required: bool = True
    public: bool = True


class ProgressSnapshot(BaseModel):
    timestamp: str = Field(default_factory=utc_now)
    project_session: int = 0
    step: int = 0
    passed_milestones: list[str] = Field(default_factory=list)
    failed_milestones: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0, le=1)
    source_report_id: str


class TrajectoryFeatures(BaseModel):
    first_protocol_error: str | None = None
    first_invalid_tool_call: str | None = None
    first_repeated_action: str | None = None
    first_no_progress_session: str | None = None
    first_stale_context_use: str | None = None
    first_harmful_knowledge_use: str | None = None
    first_completion_request: str | None = None
    latest_valid_verification_before_completion: str | None = None
    first_failed_required_check: str | None = None
    first_integrity_violation: str | None = None
    first_reopen: str | None = None
    budget_exhaustion_event: str | None = None
    final_outcome_event: str | None = None
    first_observable_symptom: str | None = None
    first_causal_divergence: str | None = None


class FailureAttribution(BaseModel):
    attribution_id: str = Field(default_factory=lambda: f"ATTR-{uuid.uuid4().hex[:12]}")
    case_id: str
    trial_id: str
    termination_reason: TerminationReason
    primary_layer: FailureLayer = FailureLayer.UNKNOWN
    primary_code: str = "UNKNOWN"
    secondary_codes: list[str] = Field(default_factory=list)
    first_symptom_event_id: str | None = None
    first_divergence_event_id: str | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    deterministic: bool = True
    needs_human_review: bool = False
    reviewer_label: str | None = None
    reviewed_at: str | None = None


class AdapterVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_report_id: str | None = None
    runtime_verdict: str | None = None
    oracle_report_id: str
    oracle_verdict: str
    oracle_report_private_path: str | None = None
    oracle_f2p_rate: float
    oracle_p2p_rate: float
    oracle_integrity_passed: bool
    oracle_partial_resolution: bool
    oracle_required_checks_passed: int
    oracle_required_checks_failed: int
    oracle_total_requirements: int
    oracle_verified_requirements: int
    oracle_verifier_seconds: float
    oracle_contract_id: str
    oracle_contract_hash: str
    oracle_baseline_fingerprint: str
    oracle_candidate_fingerprint: str
    sanitized_summary: str
    created_at: str = Field(default_factory=utc_now)


class EvaluationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    case_id: str
    config_id: str
    trial_id: str
    seed: int
    project_id: str
    project_status: str
    verification_verdict: str | None = None
    runtime_verification_verdict: str | None = None
    runtime_verification_report_id: str | None = None
    oracle_verification_verdict: str
    oracle_verification_report_id: str
    oracle_total_requirements: int = 0
    oracle_verified_requirements: int = 0
    oracle_verifier_seconds: float = 0.0
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    full_resolution: bool = False
    partial_resolution: bool = False
    f2p_rate: float = 0.0
    p2p_rate: float = 0.0
    integrity_passed: bool = False
    completion_requests: int = 0
    false_completion_requests: int = 0
    task_verified_count: int = 0
    task_reopened_count: int = 0
    wall_clock_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    compactor_tokens: int = 0
    knowledge_tokens: int = 0
    verifier_seconds: float = 0.0
    tool_calls: int = 0
    sessions: int = 0
    context_resets: int = 0
    plan_revisions: int = 0
    memory_uses: int = 0
    skill_uses: int = 0
    progress_snapshots: list[ProgressSnapshot] = Field(default_factory=list)
    failure_attribution_id: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    test_candidates: int = 0
    well_formed_test_candidates: int = 0
    f2p_tests: int = 0
    p2p_irrelevant_tests: int = 0
    harmful_tests: int = 0


class TrialResult(BaseModel):
    descriptor: TrialDescriptor
    outcome: EvaluationOutcome | None = None
    attribution: FailureAttribution | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialAttempt(BaseModel):
    evaluation_id: str
    trial_id: str
    attempt_number: int = Field(ge=1)
    status: TrialStatus
    error: str | None = None
    started_at: str
    finished_at: str
    retry_reason: str | None = None
    outcome_present: bool
    result_fingerprint: str
    result: TrialResult


def latest_trial_results(results: list[TrialResult]) -> list[TrialResult]:
    latest: dict[str, TrialResult] = {}
    for result in results:
        latest[result.descriptor.trial_id] = result
    return list(latest.values())


def load_evaluation_manifest(path: Path) -> EvaluationManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent
    manifest = EvaluationManifest.model_validate(payload)
    if not manifest.output_root.is_absolute():
        manifest.output_root = (base / manifest.output_root).resolve()
    for case in manifest.task_cases:
        for field in ("fixture", "contract_path"):
            value = getattr(case, field)
            if value is not None and not value.is_absolute():
                setattr(case, field, (base / value).resolve())
    for config in manifest.agent_configs:
        if not config.path.is_absolute():
            config.path = (base / config.path).resolve()
    return manifest
