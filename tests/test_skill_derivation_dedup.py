from __future__ import annotations

from pathlib import Path

from longrun_agent.config import KnowledgeConfig
from longrun_agent.evals.experience_learning.fake_provider import METAMORPHIC_SKILL_TITLES, MetamorphicExperienceFakeProvider
from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCase
from longrun_agent.knowledge.schema import (
    ExperienceEvidenceItem,
    ExperienceEvidencePack,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    skill_derivation_key,
)
from longrun_agent.knowledge.skill_lifecycle import SkillCandidateGenerator, SkillLifecycleManager
from longrun_agent.knowledge.store import KnowledgeStore


def test_five_paraphrases_share_one_derivation_key_and_skill_id(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    memory = _memory()
    store.save_memory(memory)
    pack = _pack()
    manager = SkillLifecycleManager(
        KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True}),
        store,
    )
    case = ExperienceLearningCase(
        case_id="metamorphic",
        repository=tmp_path,
        task_file=Path("TASK.md"),
        reset_script=Path("reset.py"),
        role="same_repository_reuse",
        knowledge_phase="bootstrap_consolidation",
        initial_verification_should_pass=False,
        final_verification_should_pass=True,
    )
    ids = set()
    for variant in range(len(METAMORPHIC_SKILL_TITLES)):
        provider = MetamorphicExperienceFakeProvider(case, variant=variant)
        generation = SkillCandidateGenerator(provider, manager.config.skill).generate_result(
            pack,
            source_memory_ids=[memory.memory_id],
            source_memories=[memory],
        )
        assert generation.draft is not None
        skill = manager.candidate_from_success(
            pack,
            draft=generation.draft,
            source_memory_ids=[memory.memory_id],
        )
        assert skill is not None
        ids.add(skill.skill_id)

    skills = store.list_skills()
    assert len(ids) == 1
    assert len(skills) == 1
    assert skills[0].derivation_key == skill_derivation_key([memory.memory_id])
    assert skills[0].semantic_signature
    assert skills[0].source_memory_ids == [memory.memory_id]


def _memory() -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title="Whitespace input",
        lesson="Normalize textual boundaries.",
        applicability_conditions=["Text input"],
        anti_conditions=["Preserve whitespace"],
    )


def _pack() -> ExperienceEvidencePack:
    return ExperienceEvidencePack(
        project_id="p",
        task_id="t",
        task_objective="Reject whitespace-only values",
        session_id="s",
        run_id="s",
        plan_version=1,
        repository_fingerprint="repo-a",
        files_changed=["validator"],
        successful_verifications=["pytest -q"],
        evidence_items=[
            ExperienceEvidenceItem(
                evidence_id="e1",
                project_id="p",
                task_id="t",
                session_id="s",
                run_id="s",
                event_type="successful_verification",
                success=True,
                exit_code=0,
            )
        ],
    )
