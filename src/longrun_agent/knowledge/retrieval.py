from __future__ import annotations

from datetime import datetime
from math import exp

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.arbitrator import KnowledgeArbitrator
from longrun_agent.knowledge.conflict import MemoryConflictDetector
from longrun_agent.knowledge.schema import (
    KnowledgeRetrievalQuery,
    KnowledgeUseType,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    RetrievalScore,
    RetrievedKnowledgeBundle,
    SkillRecord,
    SkillStatus,
)
from longrun_agent.knowledge.store import KnowledgeStore


class MemoryRetriever:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore):
        self.config = config
        self.store = store

    def retrieve(self, query: KnowledgeRetrievalQuery) -> tuple[list[MemoryRecord], list[RetrievalScore]]:
        scored: list[tuple[MemoryRecord, RetrievalScore]] = []
        for memory in self.store.list_memories():
            if not _memory_allowed(memory, query):
                continue
            score = self.score(memory, query)
            if score.total_score >= self.config.memory.minimum_retrieval_score:
                scored.append((memory, score))
        scored.sort(key=lambda item: item[1].total_score, reverse=True)
        selected = scored[: self.config.memory.max_retrieved]
        for memory, _score in selected:
            self.store.add_memory_usage(memory.memory_id, KnowledgeUseType.RETRIEVED)
        return [item[0] for item in selected], [item[1] for item in selected]

    def score(self, memory: MemoryRecord, query: KnowledgeRetrievalQuery) -> RetrievalScore:
        weights = self.config.retrieval
        lexical = _lexical_score(" ".join([memory.title, memory.lesson, *memory.tags]), query.task_objective)
        tags = _tag_score(memory.tags, [*query.language_tags, *query.framework_tags, *query.tool_tags])
        scope = _scope_score(memory.scope, query)
        evidence = min(1.0, len(memory.evidence_ids) / 3)
        utility = (memory.helpful_count + 1) / (memory.helpful_count + memory.harmful_count + 2)
        freshness = _freshness(memory.updated_at)
        conflict = MemoryConflictDetector(self.store).conflict_risk(memory)
        total = (
            weights.lexical_weight * lexical
            + weights.tag_weight * tags
            + weights.scope_weight * scope
            + weights.evidence_weight * evidence
            + weights.utility_weight * utility
            + weights.freshness_weight * freshness
            - weights.conflict_penalty * conflict
        )
        return RetrievalScore(
            item_id=memory.memory_id,
            total_score=max(0.0, total),
            lexical_score=lexical,
            tag_score=tags,
            scope_score=scope,
            evidence_score=evidence,
            utility_score=utility,
            freshness_score=freshness,
            conflict_penalty=conflict,
            reasons=[reason for reason in ["lexical" if lexical else "", "tag" if tags else "", "scope" if scope else ""] if reason],
        )


class SkillRetriever:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore):
        self.config = config
        self.store = store

    def retrieve(self, query: KnowledgeRetrievalQuery) -> tuple[list[SkillRecord], list[RetrievalScore]]:
        selected, scores, _audits = self.retrieve_with_audit(query)
        return selected, scores

    def retrieve_with_audit(
        self,
        query: KnowledgeRetrievalQuery,
        *,
        selected_memory_ids: list[str] | None = None,
    ) -> tuple[list[SkillRecord], list[RetrievalScore], list[dict]]:
        if not self.config.skill.enabled:
            return [], [], []
        selected_memory_ids = selected_memory_ids or []
        selected_memory_set = set(selected_memory_ids)
        provenance_scored: list[tuple[SkillRecord, RetrievalScore]] = []
        semantic_scored: list[tuple[SkillRecord, RetrievalScore]] = []
        audits: list[dict] = []
        for skill in self.store.list_skills():
            score = self.score(skill, query)
            parent_memory_ids = sorted(selected_memory_set & set(skill.source_memory_ids))
            provenance_rejections = _provenance_skill_rejection_reasons(skill, query, self.config)
            provenance_match = bool(parent_memory_ids) and not provenance_rejections
            direct_rejections = _skill_rejection_reasons(skill, query, score, self.config)
            direct_match = not direct_rejections and score.total_score >= self.config.memory.minimum_retrieval_score
            if provenance_match:
                provenance_scored.append((skill, score))
            if direct_match:
                semantic_scored.append((skill, score))
            rejection_reasons = []
            if not provenance_match and not direct_match:
                rejection_reasons = list(dict.fromkeys([*provenance_rejections, *direct_rejections]))
                if not direct_rejections and score.total_score < self.config.memory.minimum_retrieval_score:
                    rejection_reasons.append("below_minimum_score")
                if not parent_memory_ids:
                    rejection_reasons.append("no_selected_source_memory")
            audits.append(
                _skill_audit(
                    skill,
                    query,
                    score,
                    accepted=provenance_match or direct_match,
                    rejection_reasons=rejection_reasons,
                    origin="derived_from_selected_memory" if provenance_match else "direct_skill",
                    parent_memory_ids=parent_memory_ids,
                    provenance_match=provenance_match,
                )
            )
        provenance_scored.sort(key=lambda item: (item[0].status == SkillStatus.ACTIVE, item[1].total_score), reverse=True)
        semantic_scored.sort(key=lambda item: (item[0].status == SkillStatus.ACTIVE, item[1].total_score), reverse=True)
        merged: list[tuple[SkillRecord, RetrievalScore]] = []
        merged_ids: set[str] = set()
        for item in [*provenance_scored, *semantic_scored]:
            if item[0].skill_id in merged_ids:
                continue
            merged.append(item)
            merged_ids.add(item[0].skill_id)
        selected = merged[: self.config.skill.max_retrieved]
        selected_ids = {skill.skill_id for skill, _score in selected}
        for audit in audits:
            audit["selected"] = audit["skill_id"] in selected_ids
            if audit["accepted"] and not audit["selected"]:
                audit["rejection_reasons"].append("outside_top_k")
        for skill, _score in selected:
            self.store.add_skill_usage(skill.skill_id, KnowledgeUseType.RETRIEVED)
        return [item[0] for item in selected], [item[1] for item in selected], audits

    def score(self, skill: SkillRecord, query: KnowledgeRetrievalQuery) -> RetrievalScore:
        semantic_text = " ".join([skill.title, skill.summary, *skill.preconditions, *skill.tags])
        query_text = _query_text(query)
        lexical = _lexical_score(semantic_text, query_text)
        tags = _tag_score(skill.tags, [*query.language_tags, *query.framework_tags, *query.tool_tags])
        if skill.scope == MemoryScope.PORTABLE:
            scope = 0.8
            scope_reason = "portable_semantic_match" if lexical >= self.config.skill.portable_minimum_semantic_score else ""
        else:
            scope = 1.0 if skill.repository_fingerprint == query.repository_fingerprint else 0.0
            scope_reason = "repository_match" if scope else ""
        evidence = min(1.0, len(skill.evidence_ids) / 3)
        utility = (skill.helpful_count + skill.success_count + 1) / (
            skill.helpful_count + skill.harmful_count + skill.success_count + skill.failure_count + 2
        )
        active_bonus = 0.15 if skill.status == SkillStatus.ACTIVE else 0.0
        total = 0.4 * lexical + 0.2 * tags + 0.15 * scope + 0.15 * evidence + 0.1 * utility + active_bonus
        anti_hit = _anti_condition_hit(skill.anti_conditions, query, threshold=self.config.skill.anti_condition_match_threshold)
        if anti_hit:
            total = 0.0
        return RetrievalScore(
            item_id=skill.skill_id,
            total_score=max(0.0, total),
            lexical_score=lexical,
            tag_score=tags,
            scope_score=scope,
            evidence_score=evidence,
            utility_score=utility,
            freshness_score=_freshness(skill.updated_at),
            reasons=[
                reason
                for reason in [
                    "lexical" if lexical else "",
                    "tag" if tags else "",
                    "precondition_match" if lexical else "",
                    scope_reason,
                    "status=active" if active_bonus else f"status={skill.status.value}",
                    "evidence" if evidence else "",
                    "anti_condition_rejected" if anti_hit else "",
                ]
                if reason
            ],
        )


def retrieve_bundle(
    config: KnowledgeConfig, store: KnowledgeStore, query: KnowledgeRetrievalQuery
) -> tuple[RetrievedKnowledgeBundle, list[RetrievalScore]]:
    if config.mode in {"disabled", "raw_episode", "reflection"}:
        return RetrievedKnowledgeBundle(), []
    memories, memory_scores = (
        MemoryRetriever(config, store).retrieve(query) if config.mode in {"verified_memory", "memory_skill"} else ([], [])
    )
    skill_candidates = []
    if config.mode == "memory_skill":
        skills, skill_scores, skill_candidates = SkillRetriever(config, store).retrieve_with_audit(
            query, selected_memory_ids=[memory.memory_id for memory in memories]
        )
    else:
        skills, skill_scores = [], []
    bundle = KnowledgeArbitrator().arbitrate(
        RetrievedKnowledgeBundle(memories=memories, skills=skills),
        memory_skill_mode=config.mode == "memory_skill",
    )
    retrieval_audit = [
        {
            "item_id": memory.memory_id,
            "origin": "direct_memory",
            "parent_memory_ids": [],
            "direct_semantic_score": score.lexical_score,
            "provenance_match": False,
            "anti_condition_result": "passed",
            "selected": memory.memory_id in bundle.primary_memory_ids,
            "rejection_reasons": [],
        }
        for memory, score in zip(memories, memory_scores, strict=True)
    ]
    retrieval_audit.extend(skill_candidates)
    store.append_retrieval(
        {
            "retrieval_id": bundle.retrieval_id,
            "memory_ids": [memory.memory_id for memory in memories],
            "skill_ids": [skill.skill_id for skill in skills],
            "scores": [score.model_dump(mode="json") for score in [*memory_scores, *skill_scores]],
            "skill_candidates": skill_candidates,
            "retrieval_audit": retrieval_audit,
            "primary_memory_ids": bundle.primary_memory_ids,
            "primary_skill_ids": bundle.primary_skill_ids,
            "shadowed_memory_ids": bundle.shadowed_memory_ids,
            "provenance_links": bundle.provenance_links,
        }
    )
    return bundle, [*memory_scores, *skill_scores]


def _memory_allowed(memory: MemoryRecord, query: KnowledgeRetrievalQuery) -> bool:
    if memory.status != MemoryStatus.ACTIVE:
        return False
    if memory.expires_at and datetime.fromisoformat(memory.expires_at) < datetime.now(datetime.fromisoformat(memory.expires_at).tzinfo):
        return False
    if memory.scope == MemoryScope.PROJECT and memory.project_id != query.project_id:
        return False
    if memory.scope == MemoryScope.REPOSITORY and memory.repository_fingerprint != query.repository_fingerprint:
        return False
    if _anti_condition_hit(memory.anti_conditions, query):
        return False
    return True


def _skill_allowed(skill: SkillRecord, query: KnowledgeRetrievalQuery, score: RetrievalScore, config: KnowledgeConfig) -> bool:
    return not _skill_rejection_reasons(skill, query, score, config)


def _skill_rejection_reasons(
    skill: SkillRecord, query: KnowledgeRetrievalQuery, score: RetrievalScore, config: KnowledgeConfig
) -> list[str]:
    reasons = []
    if skill.status not in {SkillStatus.ACTIVE, SkillStatus.VALIDATED}:
        reasons.append(f"status_not_retrievable:{skill.status.value}")
    if skill.scope == MemoryScope.REPOSITORY and skill.repository_fingerprint != query.repository_fingerprint:
        reasons.append("repository_fingerprint_mismatch")
    if _anti_condition_hit(skill.anti_conditions, query, threshold=config.skill.anti_condition_match_threshold):
        reasons.append("anti_condition_matched")
    if skill.scope == MemoryScope.PORTABLE and not _portable_skill_match(skill, query, score, config):
        reasons.append("portable_semantic_mismatch")
    return reasons


def _provenance_skill_rejection_reasons(
    skill: SkillRecord,
    query: KnowledgeRetrievalQuery,
    config: KnowledgeConfig,
) -> list[str]:
    reasons = []
    if skill.status not in {SkillStatus.ACTIVE, SkillStatus.VALIDATED}:
        reasons.append(f"status_not_retrievable:{skill.status.value}")
    if skill.scope == MemoryScope.REPOSITORY and skill.repository_fingerprint != query.repository_fingerprint:
        reasons.append("repository_fingerprint_mismatch")
    if _anti_condition_hit(skill.anti_conditions, query, threshold=config.skill.anti_condition_match_threshold):
        reasons.append("anti_condition_matched")
    return reasons


def _skill_audit(
    skill: SkillRecord,
    query: KnowledgeRetrievalQuery,
    score: RetrievalScore,
    *,
    accepted: bool,
    rejection_reasons: list[str],
    origin: str = "direct_skill",
    parent_memory_ids: list[str] | None = None,
    provenance_match: bool = False,
) -> dict:
    anti_condition_score = _anti_condition_score(skill.anti_conditions, query)
    return {
        "item_id": skill.skill_id,
        "skill_id": skill.skill_id,
        "origin": origin,
        "parent_memory_ids": parent_memory_ids or [],
        "direct_semantic_score": score.lexical_score,
        "provenance_match": provenance_match,
        "anti_condition_result": "rejected" if anti_condition_score >= 0.75 else "passed",
        "status": skill.status.value,
        "scope": skill.scope.value,
        "repository_match": skill.repository_fingerprint == query.repository_fingerprint,
        "semantic_score": score.lexical_score,
        "precondition_score": _precondition_score(skill, query),
        "anti_condition_score": anti_condition_score,
        "total_score": score.total_score,
        "accepted": accepted,
        "selected": False,
        "rejection_reasons": list(rejection_reasons),
    }


def _precondition_score(skill: SkillRecord, query: KnowledgeRetrievalQuery) -> float:
    query_text = _query_text(query)
    return max((_lexical_score(precondition, query_text) for precondition in skill.preconditions), default=0.0)


def _anti_condition_score(conditions: list[str], query: KnowledgeRetrievalQuery) -> float:
    text_tokens = _tokens(_query_text(query))
    scores = []
    for condition in conditions:
        condition_tokens = _tokens(condition)
        if condition_tokens:
            scores.append(len(condition_tokens & text_tokens) / len(condition_tokens))
    return max(scores, default=0.0)


def _portable_skill_match(skill: SkillRecord, query: KnowledgeRetrievalQuery, score: RetrievalScore, config: KnowledgeConfig) -> bool:
    if score.lexical_score >= config.skill.portable_minimum_semantic_score:
        return True
    if score.tag_score > 0:
        return True
    query_text = _query_text(query)
    return any(
        _lexical_score(precondition, query_text) >= config.skill.portable_minimum_semantic_score for precondition in skill.preconditions
    )


def _anti_condition_hit(conditions: list[str], query: KnowledgeRetrievalQuery, *, threshold: float = 0.75) -> bool:
    text_tokens = _tokens(_query_text(query))
    for condition in conditions:
        condition_tokens = _tokens(condition)
        if not condition_tokens:
            continue
        overlap = len(condition_tokens & text_tokens) / len(condition_tokens)
        if overlap >= threshold:
            return True
    return False


def _lexical_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tag_score(record_tags: list[str], query_tags: list[str]) -> float:
    left = set(record_tags)
    right = set(query_tags)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _scope_score(scope: MemoryScope, query: KnowledgeRetrievalQuery) -> float:
    if scope == MemoryScope.PORTABLE:
        return 0.8
    if scope == MemoryScope.REPOSITORY and query.repository_fingerprint:
        return 1.0
    if scope == MemoryScope.PROJECT and query.project_id:
        return 0.7
    if scope == MemoryScope.TASK:
        return 0.4
    return 0.0


def _freshness(updated_at: str) -> float:
    try:
        age_days = max(0.0, (datetime.now(datetime.fromisoformat(updated_at).tzinfo) - datetime.fromisoformat(updated_at)).days)
    except ValueError:
        age_days = 365
    return exp(-age_days / 90)


def _tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "when",
        "then",
        "that",
        "this",
        "must",
        "should",
        "task",
        "file",
        "files",
        "use",
        "using",
    }
    return {
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) > 2 and token not in stopwords
    }


def _query_text(query: KnowledgeRetrievalQuery) -> str:
    return " ".join([query.task_objective, *query.acceptance_criteria, query.blocker or "", *query.recent_error_signatures])
