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
from longrun_agent.protocol import ModelResponse, ToolCall


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


def test_consolidator_does_not_induce_skill_without_formal_verified_report(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    provider = FakeModelProvider([])
    outcome = _outcome(memory.memory_id)
    outcome.verified = False
    outcome.experience_pack.verification_report_id = "REPORT-partial"
    outcome.experience_pack.verification_verdict = "partial"

    result = KnowledgeConsolidator(_config(), store, provider).consolidate(outcome)

    assert result.created_skill_ids == []
    assert provider.calls == 0
    evaluated = [event for event in store.read_jsonl(store.events_path) if event.get("event_type") == "skill_candidate_evaluated"]
    assert evaluated[-1]["verification_passed"] is False
    assert "verification_failed" in evaluated[-1]["rejection_reasons"]


def test_consolidator_induces_skill_from_formal_verified_report(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    outcome = _outcome(memory.memory_id)
    outcome.verified = True
    outcome.experience_pack.verification_report_id = "REPORT-verified"
    outcome.experience_pack.verification_verdict = "verified"
    provider = FakeModelProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="skill",
                        name="submit_skill_candidate",
                        arguments={
                            "title": "Normalize textual validation input",
                            "summary": "Normalize text before evaluating whether it is blank.",
                            "scope": "portable",
                            "preconditions": ["A textual boundary rejects blank input."],
                            "anti_conditions": ["Whitespace is semantically meaningful."],
                            "procedure": [
                                {
                                    "description": "Normalize the input before the emptiness check.",
                                    "tool_name": "write_file",
                                }
                            ],
                            "verification": ["python -m pytest -q"],
                            "source_memory_ids": [memory.memory_id],
                            "evidence_ids": ["e1"],
                            "confidence": 0.9,
                        },
                    )
                ]
            )
        ]
    )

    result = KnowledgeConsolidator(_config(), store, provider).consolidate(outcome)

    assert len(result.created_skill_ids) == 1
    assert store.load_skill(result.created_skill_ids[0]).status.value == "validated"
    assert provider.calls == 1


def test_infrastructure_error_skips_implementation_reflection(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    outcome = _outcome(memory.memory_id)
    outcome.verified = False
    outcome.verification_passed = False
    outcome.experience_pack.verification_report_id = "REPORT-infra"
    outcome.experience_pack.verification_verdict = "infrastructure_error"
    outcome.experience_pack.infrastructure_error = "verification command could not start"
    outcome.experience_pack.failed_verifications = ["verification infrastructure unavailable"]
    provider = FakeModelProvider([])

    result = KnowledgeConsolidator(_config(), store, provider).consolidate(outcome)

    assert result.created_memory_ids == []
    assert result.created_skill_ids == []
    assert provider.calls == 0
    events = store.read_jsonl(store.events_path)
    skipped = [event for event in events if event.get("event_type") == "reflection_skipped"]
    assert skipped[-1]["reason"] == "verification infrastructure errors do not trigger implementation reflection"


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
