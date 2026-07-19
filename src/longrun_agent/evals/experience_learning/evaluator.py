from __future__ import annotations

from pathlib import Path

from longrun_agent.evals.experience_learning.collector import collect_store_counts
from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCaseResult, ExperienceLearningResult


def evaluate_mode_repeat(
    *, mode: str, repeat: int, case_results: list[ExperienceLearningCaseResult], knowledge_root: Path, workspace_root: Path
) -> ExperienceLearningResult:
    for case in case_results:
        case.derive_structural_readiness()
    counts = collect_store_counts(knowledge_root, workspace_root) if knowledge_root.exists() else _empty_counts()
    expected_memory_cases = [case for case in case_results if case.role in {"same_repository_reuse", "positive_transfer"}]
    positive_transfer_cases = [case for case in case_results if case.role == "positive_transfer"]
    required_success_cases = [case for case in case_results if case.role != "learning_probe"]
    memory_hits = sum(1 for case in expected_memory_cases if case.exposed_memory_ids)
    skill_hits = sum(1 for case in positive_transfer_cases if case.exposed_skill_ids)
    negative_transfer_count = sum(
        1
        for case in case_results
        if case.role == "negative_transfer"
        and (case.exposed_skill_ids or case.referenced_skill_ids)
        and case.modified_after_run
        and not case.final_verification.passed
    )
    positive_success = sum(1 for case in positive_transfer_cases if case.final_verification.passed)
    frozen_transfer_cases = [case for case in case_results if case.knowledge_phase == "frozen_transfer"]
    frozen_negative_cases = [case for case in case_results if case.knowledge_phase == "frozen_negative_control"]
    assisted_success = sum(case.knowledge_assisted_transfer_success for case in frozen_transfer_cases)
    first_verification_operations = [
        case.operations_to_first_successful_verification
        for case in case_results
        if case.operations_to_first_successful_verification is not None
    ]
    derivation_keys = [
        skill.derivation_key
        for skill in collect_skills(knowledge_root, workspace_root)
        if skill.status.value != "deprecated" and skill.derivation_key
    ]
    verification_success = sum(1 for case in required_success_cases if case.final_verification.passed)
    return ExperienceLearningResult(
        mode=mode,
        repeat=repeat,
        case_count=len(case_results),
        project_count=len({case.project_id for case in case_results}),
        episode_count=sum(len(case.episode_ids) for case in case_results),
        memory_retrieval_hit_rate=memory_hits / len(expected_memory_cases) if expected_memory_cases else 0.0,
        skill_retrieval_hit_rate=skill_hits / len(positive_transfer_cases) if positive_transfer_cases else 0.0,
        negative_transfer_count=negative_transfer_count,
        verification_success_rate=verification_success / len(required_success_cases) if required_success_cases else 0.0,
        positive_transfer_success_rate=positive_success / len(positive_transfer_cases) if positive_transfer_cases else 0.0,
        knowledge_assisted_transfer_success_rate=(assisted_success / len(frozen_transfer_cases) if frozen_transfer_cases else 0.0),
        preexisting_skill_reference_count=sum(case.preexisting_skill_reference_count for case in case_results),
        preexisting_skill_helpful_count=sum(case.preexisting_skill_helpful_count for case in case_results),
        target_side_skill_creation_count=sum(
            len(case.created_skill_ids) for case in case_results if case.knowledge_phase in {"frozen_transfer", "frozen_negative_control"}
        ),
        frozen_store_mutation_count=sum(
            case.mutation_violation_count
            + int(
                case.knowledge_phase in {"frozen_transfer", "frozen_negative_control"}
                and case.knowledge_records_hash_before != case.knowledge_records_hash_after
            )
            for case in case_results
        ),
        duplicate_derivation_key_count=len(derivation_keys) - len(set(derivation_keys)),
        snapshot_skill_exposure_count=sum(case.snapshot_skill_exposed for case in case_results),
        frozen_transfer_completion_rate=(
            sum(case.frozen_transfer_completed for case in frozen_transfer_cases) / len(frozen_transfer_cases)
            if frozen_transfer_cases
            else 0.0
        ),
        frozen_negative_control_preservation_rate=(
            sum(case.frozen_negative_control_preserved for case in frozen_negative_cases) / len(frozen_negative_cases)
            if frozen_negative_cases
            else 0.0
        ),
        session_count=sum(case.session_count for case in case_results),
        tool_call_count=sum(case.tool_call_count for case in case_results),
        unsupported_shell_syntax_count=sum(case.unsupported_shell_syntax_count for case in case_results),
        protocol_error_count=sum(case.protocol_error_count for case in case_results),
        recoverable_protocol_error_count=sum(case.recoverable_protocol_error_count for case in case_results),
        fatal_protocol_error_count=sum(case.fatal_protocol_error_count for case in case_results),
        provider_error_count=sum(case.provider_error_count for case in case_results),
        cross_repository_skill_reference_count=sum(case.cross_repository_skill_reference_count for case in case_results),
        cross_repository_skill_helpful_count=sum(case.cross_repository_skill_helpful_count for case in case_results),
        input_tokens=sum(case.input_tokens for case in case_results),
        output_tokens=sum(case.output_tokens for case in case_results),
        total_tokens=sum(case.total_tokens for case in case_results),
        wall_time_seconds=sum(case.wall_time_seconds for case in case_results),
        mean_operations_to_first_successful_verification=(
            sum(first_verification_operations) / len(first_verification_operations) if first_verification_operations else 0.0
        ),
        changed_file_count=sum(case.changed_file_count for case in case_results),
        invalid_or_repeated_tool_call_count=sum(case.invalid_or_repeated_tool_call_count for case in case_results),
        **counts,
    )


def collect_skills(knowledge_root: Path, workspace_root: Path):
    from longrun_agent.knowledge.store import KnowledgeStore

    return KnowledgeStore(knowledge_root, workspace_root=workspace_root).list_skills()


def _empty_counts() -> dict[str, int]:
    return {
        "reflection_candidate_count": 0,
        "active_memory_count": 0,
        "quarantined_memory_count": 0,
        "validated_skill_count": 0,
        "active_skill_count": 0,
        "memory_retrieval_count": 0,
        "memory_exposure_count": 0,
        "memory_reference_count": 0,
        "memory_helpful_count": 0,
        "memory_harmful_count": 0,
        "skill_retrieval_count": 0,
        "skill_exposure_count": 0,
        "skill_reference_count": 0,
        "skill_helpful_count": 0,
        "skill_harmful_count": 0,
        "knowledge_error_count": 0,
        "portable_skill_count": 0,
        "repository_skill_count": 0,
        "skills_created_without_source_memory_count": 0,
        "reflection_rejection_count": 0,
        "reflection_rejection_reasons": [],
        "reflection_acceptance_count": 0,
    }
