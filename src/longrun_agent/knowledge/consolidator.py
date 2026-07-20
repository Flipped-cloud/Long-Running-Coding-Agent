from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.memory_lifecycle import MemoryLifecycleManager
from longrun_agent.knowledge.reflection import ReflectionGenerator
from longrun_agent.knowledge.schema import (
    ExperienceEvidencePack,
    KnowledgeUseType,
    SkillStatus,
)
from longrun_agent.knowledge.skill_lifecycle import SkillCandidateGenerator, SkillLifecycleManager
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.base import ModelProvider


class KnowledgeMutationPolicy(StrEnum):
    READ_WRITE = "read_write"
    FROZEN_RECORDS = "frozen_records"


class KnowledgeSessionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    task_id: str
    session_id: str
    repository_fingerprint: str
    referenced_memory_ids: list[str] = Field(default_factory=list)
    referenced_skill_ids: list[str] = Field(default_factory=list)
    attribution: KnowledgeUseType
    verification_passed: bool
    candidate_complete: bool
    verified: bool = False
    experience_pack: ExperienceEvidencePack


class KnowledgeConsolidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_memory_ids: list[str] = Field(default_factory=list)
    created_skill_ids: list[str] = Field(default_factory=list)
    reused_skill_ids: list[str] = Field(default_factory=list)
    skipped_reasons: list[str] = Field(default_factory=list)


class KnowledgeConsolidator:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore, model: ModelProvider):
        self.config = config
        self.store = store
        self.model = model

    def consolidate(self, outcome: KnowledgeSessionOutcome) -> KnowledgeConsolidationResult:
        before_memory_ids = {memory.memory_id for memory in self.store.list_memories()}
        before_skill_ids = {skill.skill_id for skill in self.store.list_skills()}
        skipped_reasons: list[str] = []
        referenced_memory_ids = _dedupe(outcome.referenced_memory_ids)
        referenced_skill_ids = _dedupe(outcome.referenced_skill_ids)

        self._step(outcome, "persist_usage_attribution")
        for memory_id in referenced_memory_ids:
            self._record_use(outcome, memory_id=memory_id, use_type=KnowledgeUseType.REFERENCED)
            self._record_use(outcome, memory_id=memory_id, use_type=outcome.attribution)
        for skill_id in referenced_skill_ids:
            self._record_use(outcome, skill_id=skill_id, use_type=KnowledgeUseType.REFERENCED)
            self._record_use(outcome, skill_id=skill_id, use_type=outcome.attribution)

        self._step(outcome, "update_existing_utility")
        lifecycle = SkillLifecycleManager(self.config, self.store)
        if self.store.record_mutation_policy == KnowledgeMutationPolicy.READ_WRITE.value:
            for skill_id in referenced_skill_ids:
                if outcome.attribution == KnowledgeUseType.HELPFUL:
                    lifecycle.record_success(
                        skill_id,
                        task_id=outcome.task_id,
                        repository_fingerprint=outcome.repository_fingerprint,
                    )
                elif outcome.attribution == KnowledgeUseType.HARMFUL:
                    lifecycle.record_harmful_use(skill_id, reason="verification failed after referenced skill use")
        else:
            skipped_reasons.append("record_utility_frozen")

        self._step(outcome, "consolidate_reflection")
        self._persist_episode_and_reflection(outcome, skipped_reasons)
        helpful_source_memory_ids = [
            memory_id
            for memory_id in referenced_memory_ids
            if outcome.attribution == KnowledgeUseType.HELPFUL and self.store.load_memory(memory_id).status.value == "active"
        ]

        self._step(outcome, "query_existing_derived_skill")
        existing = self.store.find_skill_by_derivation(
            source_memory_ids=helpful_source_memory_ids,
            statuses={SkillStatus.VALIDATED, SkillStatus.ACTIVE},
        )
        reused_skill_ids = list(referenced_skill_ids)
        if existing is not None:
            reused_skill_ids.append(existing.skill_id)
            self.store.append_event(
                "existing_derived_skill_reused",
                project_id=outcome.project_id,
                task_id=outcome.task_id,
                session_id=outcome.session_id,
                skill_id=existing.skill_id,
                derivation_key=existing.derivation_key,
                source_memory_ids=helpful_source_memory_ids,
            )

        self._step(outcome, "induce_skill_if_allowed")
        generated_skill = None
        generation_result = None
        rejection_reasons: list[str] = []
        can_induce = (
            self.config.mode == "memory_skill" and outcome.verified and outcome.verification_passed and bool(helpful_source_memory_ids)
        )
        if referenced_skill_ids and outcome.attribution == KnowledgeUseType.HELPFUL:
            rejection_reasons.append("existing_helpful_skill_reused")
        elif existing is not None:
            rejection_reasons.append("existing_derived_skill_reused")
        elif self.store.record_mutation_policy == KnowledgeMutationPolicy.FROZEN_RECORDS.value:
            rejection_reasons.append("record_mutation_policy_frozen")
        elif not can_induce:
            if not outcome.verified or not outcome.verification_passed:
                rejection_reasons.append("verification_failed")
            if referenced_memory_ids and outcome.attribution != KnowledgeUseType.HELPFUL:
                rejection_reasons.append("referenced_memory_not_helpful")
            if not referenced_memory_ids:
                rejection_reasons.append("no_referenced_memory")
            if not rejection_reasons:
                rejection_reasons.append("skill_induction_preconditions_not_met")
        else:
            source_memories = [self.store.load_memory(memory_id) for memory_id in helpful_source_memory_ids]
            generation_result = SkillCandidateGenerator(self.model, self.config.skill).generate_result(
                outcome.experience_pack,
                source_memory_ids=helpful_source_memory_ids,
                source_memories=source_memories,
            )
            if generation_result.draft is not None:
                generated_skill = lifecycle.candidate_from_success(
                    outcome.experience_pack,
                    draft=generation_result.draft,
                    source_memory_ids=helpful_source_memory_ids,
                )
            if generated_skill is None:
                rejection_reasons.append(
                    "portability_gate_failed" if generation_result.draft is not None else "candidate_generation_failed"
                )

        self._step(outcome, "deterministic_derivation_upsert")
        if generated_skill is not None and generated_skill.skill_id in before_skill_ids:
            reused_skill_ids.append(generated_skill.skill_id)

        self._step(outcome, "persist_consolidation_result")
        after_memory_ids = {memory.memory_id for memory in self.store.list_memories()}
        after_skill_ids = {skill.skill_id for skill in self.store.list_skills()}
        created_memory_ids = sorted(after_memory_ids - before_memory_ids)
        created_skill_ids = sorted(after_skill_ids - before_skill_ids)
        diagnostics = generation_result.model_dump(mode="json") if generation_result is not None else {}
        if self.config.mode == "memory_skill":
            source_memories = [self.store.load_memory(memory_id) for memory_id in helpful_source_memory_ids]
            self.store.append_event(
                "skill_candidate_evaluated",
                project_id=outcome.project_id,
                task_id=outcome.task_id,
                session_id=outcome.session_id,
                referenced_memory_ids=referenced_memory_ids,
                helpful_memory_ids=helpful_source_memory_ids,
                source_memory_ids=helpful_source_memory_ids,
                source_memory_statuses={memory.memory_id: memory.status.value for memory in source_memories},
                source_memory_scopes={memory.memory_id: memory.scope.value for memory in source_memories},
                verification_passed=outcome.verification_passed and outcome.verified,
                candidate_created=bool(created_skill_ids),
                candidate_id=generated_skill.skill_id if generated_skill is not None else "",
                final_skill_scope=generated_skill.scope.value if generated_skill is not None else "",
                rejection_reasons=rejection_reasons,
                failure_stage=diagnostics.get("failure_stage") or "",
                exception_type=diagnostics.get("exception_type") or "",
                validation_errors=diagnostics.get("validation_errors") or [],
                raw_response_excerpt=diagnostics.get("raw_response_excerpt") or "",
                attempt_count=diagnostics.get("attempt_count") or 0,
                generation_origin=diagnostics.get("generation_origin") or "",
                repair_actions=diagnostics.get("repair_actions") or [],
            )
        skipped_reasons.extend(rejection_reasons)
        result = KnowledgeConsolidationResult(
            created_memory_ids=created_memory_ids,
            created_skill_ids=created_skill_ids,
            reused_skill_ids=sorted(set(reused_skill_ids)),
            skipped_reasons=_dedupe(skipped_reasons),
        )
        self.store.append_event(
            "knowledge_consolidation_completed",
            project_id=outcome.project_id,
            task_id=outcome.task_id,
            session_id=outcome.session_id,
            **result.model_dump(mode="json"),
        )
        return result

    def _persist_episode_and_reflection(self, outcome: KnowledgeSessionOutcome, skipped_reasons: list[str]) -> None:
        pack = outcome.experience_pack
        project_dir = self.store.root.parent / "projects" / outcome.project_id
        if self.config.episode.save_all_sessions or pack.no_progress or pack.blockers or pack.failed_verifications:
            self.store.save_episode(project_dir, pack)
        lifecycle = MemoryLifecycleManager(self.config, self.store)
        should_reflect, reason = lifecycle.should_reflect(pack)
        if pack.infrastructure_error:
            should_reflect = False
            reason = "verification infrastructure errors do not trigger implementation reflection"
            skipped_reasons.append("verification_infrastructure_error")
        if pack.verification_verdict == "contract_invalid":
            should_reflect = False
            reason = "invalid contracts do not generate knowledge"
            skipped_reasons.append("verification_contract_invalid")
        if self.store.record_mutation_policy == KnowledgeMutationPolicy.FROZEN_RECORDS.value:
            should_reflect = False
            reason = "record mutation policy frozen"
            skipped_reasons.append("reflection_frozen")
        self.store.append_event(
            "reflection_triggered" if should_reflect else "reflection_skipped",
            project_id=outcome.project_id,
            task_id=outcome.task_id,
            session_id=outcome.session_id,
            episode_id=pack.episode_id,
            reason=reason,
        )
        if not should_reflect:
            return
        candidate = ReflectionGenerator(self.model, self.config.reflection).generate(
            pack,
            related_memory_ids=outcome.referenced_memory_ids,
        )
        if candidate is not None:
            lifecycle.persist_candidate(candidate, pack)

    def _record_use(
        self,
        outcome: KnowledgeSessionOutcome,
        *,
        use_type: KnowledgeUseType,
        memory_id: str | None = None,
        skill_id: str | None = None,
    ) -> None:
        kwargs = {
            "project_id": outcome.project_id,
            "task_id": outcome.task_id,
            "session_id": outcome.session_id,
            "reason": "consolidator attribution from verification evidence",
        }
        if memory_id:
            result = self.store.add_memory_usage(memory_id, use_type, **kwargs)
            if result.created:
                self.store.append_event(
                    f"memory_{use_type.value}",
                    project_id=outcome.project_id,
                    task_id=outcome.task_id,
                    memory_id=memory_id,
                )
        elif skill_id:
            result = self.store.add_skill_usage(skill_id, use_type, **kwargs)
            if result.created:
                self.store.append_event(
                    f"skill_{use_type.value}",
                    project_id=outcome.project_id,
                    task_id=outcome.task_id,
                    skill_id=skill_id,
                )

    def _step(self, outcome: KnowledgeSessionOutcome, step: str) -> None:
        self.store.append_event(
            "knowledge_consolidation_step",
            project_id=outcome.project_id,
            task_id=outcome.task_id,
            session_id=outcome.session_id,
            step=step,
        )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
