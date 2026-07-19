from __future__ import annotations

from pathlib import Path

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.consolidator import KnowledgeConsolidator, KnowledgeSessionOutcome
from longrun_agent.knowledge.schema import (
    ExperienceEvidenceItem,
    ExperienceEvidencePack,
    KnowledgeUseType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    SkillRecord,
    SkillStatus,
    SkillStep,
)
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.fake import FakeModelProvider


def test_consolidator_uses_fixed_order_and_reuses_existing_derived_skill(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    skill = _skill(memory.memory_id)
    store.save_skill(skill)
    provider = FakeModelProvider([])

    result = KnowledgeConsolidator(_config(), store, provider).consolidate(_outcome(memory.memory_id))

    assert result.created_skill_ids == []
    assert result.reused_skill_ids == [skill.skill_id]
    assert provider.calls == 0
    steps = [event["step"] for event in store.read_jsonl(store.events_path) if event.get("event_type") == "knowledge_consolidation_step"]
    assert steps == [
        "persist_usage_attribution",
        "update_existing_utility",
        "consolidate_reflection",
        "query_existing_derived_skill",
        "induce_skill_if_allowed",
        "deterministic_derivation_upsert",
        "persist_consolidation_result",
    ]
    assert any(event.get("event_type") == "existing_derived_skill_reused" for event in store.read_jsonl(store.events_path))


def _config() -> KnowledgeConfig:
    return KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True})


def _memory() -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title="Blank input validation",
        lesson="Normalize textual input before emptiness checks.",
        recommended_actions=["Call strip() before checking emptiness."],
        applicability_conditions=["Textual input is validated."],
        anti_conditions=["Whitespace is semantically meaningful."],
        evidence_ids=["e1"],
        confidence=0.9,
    )


def _skill(memory_id: str) -> SkillRecord:
    return SkillRecord(
        status=SkillStatus.VALIDATED,
        scope=MemoryScope.PORTABLE,
        title="Parameterized input normalization",
        summary="Normalize textual boundaries before checking emptiness.",
        preconditions=["A textual validator rejects blank values."],
        anti_conditions=["Whitespace must be preserved."],
        procedure=[SkillStep(description="Apply a parameterized trim-before-emptiness check.", tool_name="write_file")],
        verification=["python -m pytest -q"],
        source_memory_ids=[memory_id],
        evidence_ids=["e1"],
        confidence=0.9,
    )


def _outcome(memory_id: str) -> KnowledgeSessionOutcome:
    pack = ExperienceEvidencePack(
        project_id="p1",
        task_id="t1",
        task_objective="Reject blank textual input",
        session_id="s1",
        run_id="s1",
        plan_version=1,
        repository_fingerprint="repo-b",
        ending_task_status="candidate_complete",
        successful_verifications=["python -m pytest -q"],
        evidence_items=[
            ExperienceEvidenceItem(
                evidence_id="e1",
                project_id="p1",
                task_id="t1",
                session_id="s1",
                run_id="s1",
                event_type="successful_verification",
                success=True,
                exit_code=0,
            )
        ],
    )
    return KnowledgeSessionOutcome(
        project_id="p1",
        task_id="t1",
        session_id="s1",
        repository_fingerprint="repo-b",
        referenced_memory_ids=[memory_id],
        attribution=KnowledgeUseType.HELPFUL,
        verification_passed=True,
        candidate_complete=True,
        experience_pack=pack,
    )
