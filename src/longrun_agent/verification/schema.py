from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from longrun_agent.state.schema import utc_now


def new_verification_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CheckKind(StrEnum):
    RESOLUTION = "resolution"
    REGRESSION = "regression"
    CANDIDATE_ONLY = "candidate_only"
    STATIC = "static"
    INTEGRITY = "integrity"
    GENERATED_TEST = "generated_test"


class CheckVisibility(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"


class ExpectedState(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ANY = "any"
    NOT_RUN = "not_run"


class ExecutionStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class TestTransition(StrEnum):
    F2P = "F2P"
    F2F = "F2F"
    P2P = "P2P"
    P2F = "P2F"
    X2P = "X2P"
    X2F = "X2F"
    P2X = "P2X"
    F2X = "F2X"
    X2X = "X2X"


class VerificationVerdict(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    REOPENED = "reopened"
    INCONCLUSIVE = "inconclusive"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CONTRACT_INVALID = "contract_invalid"


class VerificationPurpose(StrEnum):
    RUNTIME = "runtime"
    ORACLE = "oracle"


def validate_safe_argv(argv: list[str]) -> list[str]:
    if not argv or not all(isinstance(item, str) and item.strip() for item in argv):
        raise ValueError("verification argv must be a non-empty list[str]")
    forbidden = {"cd", "|", ">", ">>", "<", "&&", "||", ";"}
    if any(item.strip() in forbidden for item in argv):
        raise ValueError("verification argv contains forbidden shell syntax")
    return argv


class TaskSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specification_id: str = Field(default_factory=lambda: new_verification_id("SPEC"))
    project_id: str
    objective: str
    public_acceptance_criteria: list[str] = Field(default_factory=list)
    allowed_change_patterns: list[str] = Field(default_factory=list)
    forbidden_change_patterns: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    public_check_descriptions: list[str] = Field(default_factory=list)
    resource_budget: dict[str, int | float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    schema_version: str = "0.5"


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    kind: CheckKind
    visibility: CheckVisibility = CheckVisibility.PUBLIC
    argv: list[str] = Field(default_factory=list)
    cwd: str = "."
    timeout_seconds: int = Field(default=120, ge=1)
    required: bool = True
    weight: float = Field(default=1.0, gt=0)
    baseline_expected: ExpectedState | None = None
    candidate_expected: ExpectedState | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    artifact_patterns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_check(self) -> VerificationCheck:
        if self.kind != CheckKind.INTEGRITY:
            validate_safe_argv(self.argv)
        elif self.argv:
            raise ValueError("integrity checks do not execute argv")
        path = PurePosixPath(self.cwd.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("verification check cwd must stay inside snapshot workspace")
        if self.baseline_expected is None:
            self.baseline_expected = ExpectedState.FAIL if self.kind == CheckKind.RESOLUTION else ExpectedState.PASS
            if self.kind in {CheckKind.CANDIDATE_ONLY, CheckKind.STATIC, CheckKind.GENERATED_TEST}:
                self.baseline_expected = ExpectedState.NOT_RUN
        if self.candidate_expected is None:
            self.candidate_expected = ExpectedState.PASS
        return self


class IntegrityRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protected_paths: list[str] = Field(default_factory=list)
    trusted_test_patterns: list[str] = Field(default_factory=list)
    forbidden_change_patterns: list[str] = Field(default_factory=list)
    allowed_change_patterns: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    contract_paths: list[str] = Field(default_factory=list)
    hidden_asset_markers: list[str] = Field(default_factory=list)
    max_deleted_files: int = Field(default=100, ge=0)
    max_binary_file_bytes: int = Field(default=10_000_000, ge=1)
    allow_project_config_changes: bool = False
    allowed_test_directories: list[str] = Field(default_factory=lambda: ["tests"])


class GeneratedTestPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    can_be_required: bool = False
    max_candidates_per_task: int = Field(default=3, ge=0)


class AgentVisibleIntegrityRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protected_paths: list[str] = Field(default_factory=list)
    trusted_test_patterns: list[str] = Field(default_factory=list)
    forbidden_change_patterns: list[str] = Field(default_factory=list)
    allowed_change_patterns: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    max_deleted_files: int = Field(default=100, ge=0)
    max_binary_file_bytes: int = Field(default=10_000_000, ge=1)
    allow_project_config_changes: bool = False
    allowed_test_directories: list[str] = Field(default_factory=lambda: ["tests"])


class AgentVisibleContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_visibility: Literal["agent"] = "agent"
    contract_id: str
    contract_version: int = Field(default=1, ge=1)
    project_id: str
    task_id: str | None = None
    task_key: str | None = None
    scope: Literal["project", "task"] = "project"
    source: Literal["file", "legacy", "fixture", "external_adapter"] = "file"
    checks: list[VerificationCheck] = Field(default_factory=list)
    integrity_rules: AgentVisibleIntegrityRules = Field(default_factory=AgentVisibleIntegrityRules)
    generated_test_policy: GeneratedTestPolicy = Field(default_factory=GeneratedTestPolicy)
    completion_requirements: list[str] = Field(default_factory=list)
    opaque_contract_fingerprint: str
    inconclusive_action: Literal["block", "ready"] = "block"
    schema_version: str = "0.5.5"

    @model_validator(mode="after")
    def public_checks_only(self) -> AgentVisibleContract:
        if any(check.visibility != CheckVisibility.PUBLIC for check in self.checks):
            raise ValueError("agent-visible contract cannot contain hidden checks")
        return self

    def to_runtime_contract(self) -> VerificationContract:
        rules = IntegrityRules(**self.integrity_rules.model_dump())
        return VerificationContract(
            contract_id=self.contract_id,
            contract_version=self.contract_version,
            project_id=self.project_id,
            task_id=self.task_id,
            task_key=self.task_key,
            scope=self.scope,
            source=self.source,
            checks=self.checks,
            integrity_rules=rules,
            generated_test_policy=self.generated_test_policy,
            hidden_assets_root=None,
            inconclusive_action=self.inconclusive_action,
            schema_version=self.schema_version,
        )


class BaselineSnapshotReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: str = ""
    manifest_path: str = ""
    source_revision: str | None = None


class VerificationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_id: str = Field(default_factory=lambda: new_verification_id("CONTRACT"))
    contract_version: int = Field(default=1, ge=1)
    project_id: str
    task_id: str | None = None
    task_key: str | None = None
    scope: Literal["project", "task"] = "project"
    source: Literal["file", "legacy", "fixture", "external_adapter"] = "file"
    frozen_at: str | None = None
    contract_hash: str = ""
    checks: list[VerificationCheck] = Field(default_factory=list)
    integrity_rules: IntegrityRules = Field(default_factory=IntegrityRules)
    generated_test_policy: GeneratedTestPolicy = Field(default_factory=GeneratedTestPolicy)
    environment_requirements: dict[str, str] = Field(default_factory=dict)
    baseline_snapshot: BaselineSnapshotReference | None = None
    hidden_assets_root: Path | None = None
    inconclusive_action: Literal["block", "ready"] = "block"
    schema_version: str = "0.5"

    @model_validator(mode="after")
    def validate_contract(self) -> VerificationContract:
        check_ids = [check.check_id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("verification check IDs must be unique")
        if self.scope == "task" and not (self.task_id or self.task_key):
            raise ValueError("task-scoped contract requires task_id or task_key")
        if self.generated_test_policy.can_be_required:
            raise ValueError("generated tests cannot be required in v0.5")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"contract_hash", "frozen_at"})

    def canonical_hash(self) -> str:
        payload = json.dumps(self.canonical_payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def freeze(self) -> VerificationContract:
        frozen = self.model_copy(deep=True)
        frozen.frozen_at = frozen.frozen_at or utc_now()
        frozen.contract_hash = frozen.canonical_hash()
        return frozen


class OraclePrivateContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_visibility: Literal["oracle_private"] = "oracle_private"
    contract: VerificationContract
    private_fingerprint: str
    schema_version: str = "0.5.5"

    @model_validator(mode="after")
    def validate_fingerprint(self) -> OraclePrivateContract:
        if self.contract.contract_hash != self.private_fingerprint:
            raise ValueError("oracle private contract fingerprint mismatch")
        return self


class FileManifestEntry(BaseModel):
    relative_path: str
    sha256: str
    size: int = Field(ge=0)
    mode: int = Field(ge=0)


class WorkspaceManifest(BaseModel):
    workspace_kind: Literal["baseline", "candidate", "gold", "current"]
    root: str
    fingerprint: str
    files: list[FileManifestEntry] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class CheckExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    kind: CheckKind
    visibility: CheckVisibility
    workspace_kind: Literal["baseline", "candidate", "gold", "current"]
    started_at: str
    finished_at: str
    duration_seconds: float = Field(ge=0)
    exit_code: int | None = None
    timed_out: bool = False
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    stdout_artifact: str | None = None
    stderr_artifact: str | None = None
    status: ExecutionStatus
    infrastructure_error: str | None = None
    environment_fingerprint: str = ""


class CheckTransitionResult(BaseModel):
    check_id: str
    kind: CheckKind
    required: bool
    transition: TestTransition
    baseline_status: ExecutionStatus = ExecutionStatus.SKIPPED
    candidate_status: ExecutionStatus = ExecutionStatus.SKIPPED


class IntegrityViolation(BaseModel):
    violation_id: str = Field(default_factory=lambda: new_verification_id("VIOLATION"))
    category: str
    path: str = ""
    description: str
    severity: Literal["warning", "error", "critical"] = "error"
    evidence: list[str] = Field(default_factory=list)
    agent_visible_summary: str


class VerificationSummary(BaseModel):
    resolution_total: int = 0
    resolution_passed: int = 0
    f2p_rate: float = 0.0
    regression_total: int = 0
    regression_passed: int = 0
    p2p_rate: float = 0.0
    required_checks_passed: int = 0
    required_checks_failed: int = 0
    partial_resolution: bool = False
    integrity_passed: bool = True
    hidden_checks_passed: int = 0
    hidden_checks_failed: int = 0


class TestCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(default_factory=lambda: new_verification_id("TEST"))
    task_id: str
    session_id: str
    paths: list[str] = Field(min_length=1)
    diff_artifact: str = ""
    command_argv: list[str] = Field(min_length=1)
    issue_behavior: str = Field(min_length=1)
    expected_failure_reason: str = ""
    baseline_result: CheckExecutionResult | None = None
    candidate_result: CheckExecutionResult | None = None
    transition: TestTransition | None = None
    valid: bool = False
    valid_but_irrelevant: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    coverage_summary: dict[str, int | float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_command(self) -> TestCandidate:
        validate_safe_argv(self.command_argv)
        return self


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(default_factory=lambda: new_verification_id("REPORT"))
    purpose: VerificationPurpose = VerificationPurpose.RUNTIME
    project_id: str
    task_id: str | None = None
    contract_id: str
    contract_hash: str
    verdict: VerificationVerdict
    baseline_fingerprint: str = ""
    candidate_fingerprint: str = ""
    environment_fingerprint: str = ""
    baseline_results: list[CheckExecutionResult] = Field(default_factory=list)
    candidate_results: list[CheckExecutionResult] = Field(default_factory=list)
    transitions: list[CheckTransitionResult] = Field(default_factory=list)
    integrity_violations: list[IntegrityViolation] = Field(default_factory=list)
    test_candidates: list[TestCandidate] = Field(default_factory=list)
    summary: VerificationSummary = Field(default_factory=VerificationSummary)
    sanitized_feedback: str = ""
    infrastructure_error: str | None = None
    created_at: str = Field(default_factory=utc_now)
    artifact_paths: list[str] = Field(default_factory=list)
