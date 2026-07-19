from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from longrun_agent.config import ModelConfig


class ExperienceLearningCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    repository: Path
    task_file: Path
    reset_script: Path
    role: Literal["learning_probe", "same_repository_reuse", "positive_transfer", "negative_transfer"]
    knowledge_phase: Literal[
        "bootstrap_learning",
        "bootstrap_consolidation",
        "frozen_transfer",
        "frozen_negative_control",
    ] = "bootstrap_learning"
    initial_verification_should_pass: bool
    final_verification_should_pass: bool
    task: str = ""
    expected_memory_tags: list[str] = Field(default_factory=list)
    expected_skill_tags: list[str] = Field(default_factory=list)
    is_negative_transfer: bool = False


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    timeout_seconds: int = Field(default=30, ge=1)


class ExperienceLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 7
    backend: Literal["fake", "api"] = "fake"
    model: ModelConfig | None = None
    repeats: int = Field(default=1, ge=1)
    output_root: Path = Path(".runs/experience_learning")
    fail_fast_on_knowledge_error: bool = True
    modes: list[str] = Field(default_factory=lambda: ["disabled", "raw_episode", "reflection", "verified_memory", "memory_skill"])
    cases: list[ExperienceLearningCase]
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    @model_validator(mode="after")
    def validate_backend_model(self) -> ExperienceLearningConfig:
        if self.backend == "api" and self.model is None:
            raise ValueError("experience learning backend='api' requires top-level model configuration")
        return self


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    passed: bool


class ExperienceLearningCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    repeat: int
    case_id: str
    role: str
    knowledge_phase: str = ""
    repository: str
    project_id: str
    project_status: str
    reset_exit_code: int
    initial_verification: VerificationResult
    final_verification: VerificationResult
    exposed_memory_ids: list[str] = Field(default_factory=list)
    exposed_skill_ids: list[str] = Field(default_factory=list)
    referenced_memory_ids: list[str] = Field(default_factory=list)
    referenced_skill_ids: list[str] = Field(default_factory=list)
    helpful_memory_ids: list[str] = Field(default_factory=list)
    harmful_memory_ids: list[str] = Field(default_factory=list)
    helpful_skill_ids: list[str] = Field(default_factory=list)
    harmful_skill_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    created_memory_ids: list[str] = Field(default_factory=list)
    created_skill_ids: list[str] = Field(default_factory=list)
    available_memory_ids: list[str] = Field(default_factory=list)
    available_skill_ids: list[str] = Field(default_factory=list)
    session_count: int = 0
    tool_call_count: int = 0
    unsupported_shell_syntax_count: int = 0
    protocol_error_count: int = 0
    recoverable_protocol_error_count: int = 0
    fatal_protocol_error_count: int = 0
    provider_error_count: int = 0
    reflection_rejection_count: int = 0
    reflection_rejection_reasons: list[str] = Field(default_factory=list)
    reflection_acceptance_count: int = 0
    portable_skill_count: int = 0
    repository_skill_count: int = 0
    skills_created_without_source_memory_count: int = 0
    cross_repository_skill_reference_count: int = 0
    cross_repository_skill_helpful_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    knowledge_error: bool = False
    modified_after_run: bool = False
    knowledge_records_hash_before: str = ""
    knowledge_records_hash_after: str = ""
    snapshot_skill_ids: list[str] = Field(default_factory=list)
    mutation_violation_count: int = 0
    preexisting_skill_reference_count: int = 0
    preexisting_skill_helpful_count: int = 0
    knowledge_assisted_transfer_success: bool = Field(
        default=False,
        description="Self-reported knowledge usage outcome; this is not causal evidence that a Skill helped.",
    )
    snapshot_skill_exposed: bool = False
    frozen_transfer_completed: bool = False
    frozen_negative_control_preserved: bool = False
    operations_to_first_successful_verification: int | None = None
    changed_file_count: int = 0
    invalid_or_repeated_tool_call_count: int = 0

    @model_validator(mode="after")
    def derive_structural_readiness(self) -> ExperienceLearningCaseResult:
        self.snapshot_skill_exposed = bool(set(self.snapshot_skill_ids) & set(self.exposed_skill_ids))
        no_created_records = not self.created_memory_ids and not self.created_skill_ids
        records_unchanged = self.knowledge_records_hash_before == self.knowledge_records_hash_after
        self.frozen_transfer_completed = bool(
            self.knowledge_phase == "frozen_transfer"
            and self.final_verification.passed
            and self.snapshot_skill_exposed
            and no_created_records
            and records_unchanged
            and self.mutation_violation_count == 0
        )
        self.frozen_negative_control_preserved = bool(
            self.knowledge_phase == "frozen_negative_control"
            and self.initial_verification.passed
            and self.final_verification.passed
            and not self.modified_after_run
            and no_created_records
            and records_unchanged
        )
        return self


class ExperienceLearningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    repeat: int = 0
    case_count: int
    project_count: int = 0
    episode_count: int
    reflection_candidate_count: int = 0
    active_memory_count: int = 0
    quarantined_memory_count: int = 0
    validated_skill_count: int = 0
    active_skill_count: int = 0
    memory_retrieval_count: int = 0
    memory_exposure_count: int = 0
    memory_reference_count: int = 0
    memory_helpful_count: int = 0
    memory_harmful_count: int = 0
    skill_retrieval_count: int = 0
    skill_exposure_count: int = 0
    skill_reference_count: int = 0
    skill_helpful_count: int = 0
    skill_harmful_count: int = 0
    memory_retrieval_hit_rate: float
    skill_retrieval_hit_rate: float
    negative_transfer_count: int
    verification_success_rate: float
    positive_transfer_success_rate: float = 0.0
    knowledge_assisted_transfer_success_rate: float = Field(
        default=0.0,
        description="Rate of self-reported knowledge usage outcomes; not a causal Skill-effect metric.",
    )
    preexisting_skill_reference_count: int = 0
    preexisting_skill_helpful_count: int = 0
    target_side_skill_creation_count: int = 0
    frozen_store_mutation_count: int = 0
    duplicate_derivation_key_count: int = 0
    snapshot_skill_exposure_count: int = 0
    frozen_transfer_completion_rate: float = 0.0
    frozen_negative_control_preservation_rate: float = 0.0
    knowledge_error_count: int = 0
    session_count: int = 0
    tool_call_count: int = 0
    unsupported_shell_syntax_count: int = 0
    protocol_error_count: int = 0
    recoverable_protocol_error_count: int = 0
    fatal_protocol_error_count: int = 0
    provider_error_count: int = 0
    portable_skill_count: int = 0
    repository_skill_count: int = 0
    skills_created_without_source_memory_count: int = 0
    cross_repository_skill_reference_count: int = 0
    cross_repository_skill_helpful_count: int = 0
    reflection_rejection_count: int = 0
    reflection_rejection_reasons: list[str] = Field(default_factory=list)
    reflection_acceptance_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    mean_operations_to_first_successful_verification: float = 0.0
    changed_file_count: int = 0
    invalid_or_repeated_tool_call_count: int = 0


def _metric_categories() -> dict[str, list[str]]:
    return {
        "structural_readiness": [
            "snapshot_skill_exposure_count",
            "frozen_transfer_completion_rate",
            "frozen_negative_control_preservation_rate",
            "frozen_store_mutation_count",
            "duplicate_derivation_key_count",
        ],
        "behavioral_usage_telemetry": [
            "skill_reference_count",
            "skill_helpful_count",
            "cross_repository_skill_helpful_count",
            "preexisting_skill_reference_count",
            "preexisting_skill_helpful_count",
            "knowledge_assisted_transfer_success_rate",
        ],
        "effectiveness": [
            "verification_success_rate",
            "positive_transfer_success_rate",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "tool_call_count",
            "wall_time_seconds",
            "protocol_error_count",
            "unsupported_shell_syntax_count",
            "mean_operations_to_first_successful_verification",
            "changed_file_count",
            "invalid_or_repeated_tool_call_count",
        ],
    }


class ExperienceLearningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    results: list[ExperienceLearningResult]
    case_results: list[ExperienceLearningCaseResult] = Field(default_factory=list)
    metric_categories: dict[str, list[str]] = Field(default_factory=_metric_categories)
