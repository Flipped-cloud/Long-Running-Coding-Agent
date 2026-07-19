from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.renderer import render_bundle
from longrun_agent.knowledge.retrieval import retrieve_bundle
from longrun_agent.knowledge.schema import (
    KnowledgeRetrievalQuery,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    SkillRecord,
    SkillStatus,
    SkillStep,
)
from longrun_agent.knowledge.store import KnowledgeStore


def test_selected_memory_expands_low_lexical_skill_and_shadows_parent(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    skill = _skill(memory.memory_id)
    store.save_skill(skill)
    query = KnowledgeRetrievalQuery(task_objective="Reject whitespace-only command input", repository_fingerprint="repo-b")

    bundle, scores = retrieve_bundle(KnowledgeConfig(mode="memory_skill"), store, query)
    rendered, _tokens = render_bundle(bundle, KnowledgeConfig(mode="memory_skill"))

    assert [item.skill_id for item in bundle.skills] == [skill.skill_id]
    assert bundle.primary_skill_ids == [skill.skill_id]
    assert bundle.primary_memory_ids == []
    assert bundle.shadowed_memory_ids == [memory.memory_id]
    assert f"Derived from Memory ID: {memory.memory_id}" in rendered
    assert "Recommended actions:" not in rendered
    skill_score = next(score for score in scores if score.item_id == skill.skill_id)
    assert skill_score.lexical_score < 0.25
    audit = _last_retrieval(store)["retrieval_audit"]
    derived = next(item for item in audit if item["item_id"] == skill.skill_id)
    assert derived["origin"] == "derived_from_selected_memory"
    assert derived["parent_memory_ids"] == [memory.memory_id]
    assert derived["provenance_match"] is True
    assert derived["selected"] is True


def test_anti_condition_rejects_provenance_expanded_skill(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    skill = _skill(memory.memory_id)
    store.save_skill(skill)

    bundle, _scores = retrieve_bundle(
        KnowledgeConfig(mode="memory_skill"),
        store,
        KnowledgeRetrievalQuery(
            task_objective="Reject whitespace-only command input but preserve leading whitespace because indentation is semantically meaningful",
            repository_fingerprint="repo-b",
        ),
    )

    assert bundle.skills == []
    audit = next(item for item in _last_retrieval(store)["retrieval_audit"] if item["item_id"] == skill.skill_id)
    assert audit["anti_condition_result"] == "rejected"
    assert "anti_condition_matched" in audit["rejection_reasons"]


def _memory() -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title="Whitespace command validation",
        lesson="Reject command values containing spacing only.",
        recommended_actions=["Normalize command boundaries before validity checking."],
        applicability_conditions=["Command input is textual."],
        anti_conditions=["Preserve leading whitespace because indentation is semantically meaningful."],
        evidence_ids=["e1"],
        confidence=0.9,
    )


def _skill(memory_id: str) -> SkillRecord:
    return SkillRecord(
        status=SkillStatus.VALIDATED,
        scope=MemoryScope.PORTABLE,
        title="Orthogonal procedure vocabulary",
        summary="Canonicalize boundary representation before a predicate.",
        preconditions=["A scalar predicate accepts user-provided data."],
        anti_conditions=["Preserve leading whitespace because indentation is semantically meaningful."],
        procedure=[SkillStep(description="Canonicalize the value and apply the predicate.", tool_name="write_file")],
        verification=["python -m pytest -q"],
        source_memory_ids=[memory_id],
        evidence_ids=["e1"],
        confidence=0.9,
    )


def _last_retrieval(store: KnowledgeStore) -> dict:
    return json.loads(store.retrievals_path.read_text(encoding="utf-8").splitlines()[-1])
