from __future__ import annotations

import re
from datetime import UTC, datetime

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.conflict import MemoryConflictDetector
from longrun_agent.knowledge.schema import (
    EvidenceGateResult,
    ExperienceEvidencePack,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    ReflectionCandidate,
)
from longrun_agent.knowledge.store import KnowledgeStore

LEAK_PATTERNS = [
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"/(?:Users|home|tmp|var)/"),
    re.compile(r"\b[A-Fa-f0-9]{40}\b"),
    re.compile(r"\$\{?[A-Z_][A-Z0-9_]{3,}\}?"),
]


class ReflectionEvidenceGate:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore | None = None):
        self.config = config
        self.store = store

    def evaluate(self, candidate: ReflectionCandidate, pack: ExperienceEvidencePack) -> EvidenceGateResult:
        reasons: list[str] = []
        final_scope = candidate.proposed_scope
        evidence = pack.evidence_by_id()
        missing = [item for item in candidate.evidence_ids if item not in evidence]
        if missing:
            reasons.append(f"unknown evidence ids: {missing}")
        if len(candidate.evidence_ids) < self.config.reflection.minimum_evidence_items:
            reasons.append("insufficient evidence ids")
        if candidate.confidence < self.config.reflection.minimum_confidence:
            reasons.append("confidence below threshold")
        if not candidate.recommended_actions:
            reasons.append("recommended actions required")
        if not candidate.applicability_conditions:
            reasons.append("applicability conditions required")
        cited_evidence = [evidence[item] for item in candidate.evidence_ids if item in evidence]
        failure_evidence = [item for item in cited_evidence if _is_failure_evidence(item)]
        if not failure_evidence:
            reasons.append("candidate must cite at least one failure evidence item")
        for item in failure_evidence:
            if item.event_type == "test_command" and not (item.output_excerpt or item.artifact_path):
                reasons.append(f"failed test evidence lacks output excerpt or artifact: {item.evidence_id}")
        if candidate.proposed_scope == MemoryScope.PORTABLE and not candidate.anti_conditions:
            reasons.append("portable memory requires anti-conditions")
        if candidate.proposed_scope == MemoryScope.PORTABLE and _contains_leak(candidate):
            reasons.append("portable memory contains absolute path or secret-like text")
        if candidate.proposed_scope == MemoryScope.PORTABLE and _contains_source_specific_text(candidate):
            final_scope = MemoryScope.REPOSITORY
        if _successful_verification_as_failure(candidate, pack):
            reasons.append("successful verification is described as failure")
        conflict_ids: list[str] = []
        if self.store is not None:
            detector = MemoryConflictDetector(self.store)
            conflicts = detector.find_conflicts_for_candidate(candidate)
            conflict_ids = [record.conflict_id for record in conflicts]
            if any(record.severity == "high" for record in conflicts):
                reasons.append("high severity memory conflict")
        if any("confidence below threshold" == reason for reason in reasons) and not missing:
            return EvidenceGateResult(
                accepted=False,
                decision="quarantined",
                rejection_reasons=reasons,
                conflict_ids=conflict_ids,
                validated_evidence_ids=[item for item in candidate.evidence_ids if item in evidence],
                final_scope=final_scope,
                final_confidence=candidate.confidence,
            )
        if reasons:
            return EvidenceGateResult(
                accepted=False,
                decision="rejected",
                rejection_reasons=reasons,
                conflict_ids=conflict_ids,
                validated_evidence_ids=[item for item in candidate.evidence_ids if item in evidence],
                final_scope=final_scope,
                final_confidence=candidate.confidence,
            )
        return EvidenceGateResult(
            accepted=True,
            decision="accepted",
            conflict_ids=conflict_ids,
            validated_evidence_ids=candidate.evidence_ids,
            final_scope=final_scope,
            final_confidence=candidate.confidence,
        )


class MemoryLifecycleManager:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore):
        self.config = config
        self.store = store

    def should_reflect(self, pack: ExperienceEvidencePack) -> tuple[bool, str]:
        if self.config.mode in {"disabled", "raw_episode"}:
            return False, "mode does not generate reflection"
        if not self.config.reflection.enabled:
            return False, "reflection disabled"
        if pack.run_status == "provider_error" and not self.config.reflection.trigger_on_provider_error:
            return False, "provider error does not trigger reflection"
        if pack.context_reset_count and not self.config.reflection.trigger_on_context_reset and not _failure_signal(pack):
            return False, "context reset alone does not trigger reflection"
        if pack.blockers and self.config.reflection.trigger_on_blocked:
            return True, "blocked"
        if pack.no_progress and self.config.reflection.trigger_on_no_progress:
            return True, "no_progress"
        if pack.run_status in {"max_steps_reached", "terminal_signal_missing", "final_without_terminal_signal"}:
            return True, "failed_session"
        if pack.failed_verifications and self.config.reflection.trigger_on_failed:
            return True, "failed_verification"
        if pack.repeated_actions:
            return True, "repeated_action"
        return False, "no reflection trigger"

    def persist_candidate(self, candidate: ReflectionCandidate, pack: ExperienceEvidencePack) -> MemoryRecord | None:
        gate = ReflectionEvidenceGate(self.config, self.store)
        result = gate.evaluate(candidate, pack)
        self.store.append_event(
            "reflection_candidate_created",
            project_id=pack.project_id,
            task_id=pack.task_id,
            session_id=pack.session_id,
            episode_id=pack.episode_id,
            evidence_ids=candidate.evidence_ids,
            status_after=result.decision,
        )
        if result.decision == "rejected" and self.config.mode != "reflection":
            self.store.append_event(
                "reflection_evidence_rejected",
                project_id=pack.project_id,
                task_id=pack.task_id,
                session_id=pack.session_id,
                episode_id=pack.episode_id,
                reason="; ".join(result.rejection_reasons),
            )
            self.store.save_reflection_rejection(candidate, pack, result)
            return None
        status = MemoryStatus.ACTIVE if result.accepted else MemoryStatus.QUARANTINED
        if self.config.mode == "reflection":
            status = MemoryStatus.CANDIDATE
        if result.decision == "quarantined" and self.config.mode != "reflection":
            status = MemoryStatus.QUARANTINED
        record_scope = result.final_scope or candidate.proposed_scope
        record = MemoryRecord(
            kind=MemoryKind.FAILURE_REFLECTION,
            scope=record_scope,
            status=status,
            title=candidate.failure_category.replace("_", " ").title(),
            lesson=candidate.root_cause,
            recommended_actions=candidate.recommended_actions,
            avoid_actions=candidate.avoid_actions,
            applicability_conditions=candidate.applicability_conditions,
            anti_conditions=candidate.anti_conditions,
            tags=candidate.tags,
            project_id=pack.project_id if record_scope in {MemoryScope.PROJECT, MemoryScope.TASK} else None,
            repository_fingerprint=pack.repository_fingerprint if record_scope == MemoryScope.REPOSITORY else None,
            source_episode_ids=[pack.episode_id],
            source_task_ids=[pack.task_id],
            source_session_ids=[pack.session_id],
            evidence_ids=result.validated_evidence_ids,
            confidence=result.final_confidence,
        )
        duplicate = self.store.find_duplicate_memory(record)
        if duplicate is not None:
            duplicate.source_episode_ids = _dedupe([*duplicate.source_episode_ids, pack.episode_id])
            duplicate.source_task_ids = _dedupe([*duplicate.source_task_ids, pack.task_id])
            duplicate.source_session_ids = _dedupe([*duplicate.source_session_ids, pack.session_id])
            duplicate.evidence_ids = _dedupe([*duplicate.evidence_ids, *record.evidence_ids])
            duplicate.updated_at = datetime.now(UTC).isoformat()
            self.store.save_memory(duplicate)
            return duplicate
        self.store.save_memory(record)
        return record


def _is_failure_evidence(item) -> bool:
    return bool(
        (item.event_type == "test_command" and item.success is False)
        or (item.exit_code is not None and item.exit_code != 0)
        or item.event_type in {"blocker", "no_progress", "repeated_tool_call"}
    )


def _contains_leak(candidate: ReflectionCandidate) -> bool:
    text = " ".join(
        [
            candidate.root_cause,
            *candidate.recommended_actions,
            *candidate.avoid_actions,
            *candidate.applicability_conditions,
            *candidate.anti_conditions,
        ]
    )
    return any(pattern.search(text) for pattern in LEAK_PATTERNS)


GENERIC_OPERATION_NAMES = {"strip", "bool", "str", "len", "lower", "upper", "isinstance", "pytest"}


def _contains_source_specific_text(candidate: ReflectionCandidate) -> bool:
    text = " ".join(
        [
            candidate.root_cause,
            *candidate.recommended_actions,
            *candidate.avoid_actions,
            *candidate.applicability_conditions,
            *candidate.anti_conditions,
        ]
    )
    function_names = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)", text))
    source_specific_functions = function_names - GENERIC_OPERATION_NAMES
    return bool(re.search(r"\b[\w.-]+\.py\b", text) or source_specific_functions)


def _successful_verification_as_failure(candidate: ReflectionCandidate, pack: ExperienceEvidencePack) -> bool:
    if not pack.successful_verifications:
        return False
    text = " ".join([candidate.root_cause, *candidate.recommended_actions]).lower()
    return "pytest failed" in text or "verification failed" in text


def _failure_signal(pack: ExperienceEvidencePack) -> bool:
    return bool(pack.no_progress or pack.blockers or pack.failed_verifications or pack.repeated_actions)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
