from __future__ import annotations

from pathlib import Path

from longrun_agent.evals.experience_learning.collector import collect_case_result
from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCase, VerificationResult
from longrun_agent.knowledge.schema import MemoryKind, MemoryRecord, MemoryScope, MemoryStatus, SkillRecord, SkillStatus, SkillStep
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.state.store import ProjectStateStore


def test_collector_created_ids_are_case_level_deltas(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    knowledge_root = tmp_path / "knowledge"
    state_root = tmp_path / "projects"
    state_root.mkdir()
    store = KnowledgeStore(knowledge_root, workspace_root=workspace)

    existing_memory = _memory("Existing memory")
    created_memory = _memory("Created memory")
    existing_skill = _skill("Existing skill")
    created_skill = _skill("Created skill")
    for record in [existing_memory, created_memory]:
        store.save_memory(record)
    for record in [existing_skill, created_skill]:
        store.save_skill(record)

    case = ExperienceLearningCase(
        case_id="repo_a_reuse",
        repository=workspace,
        task_file=Path("TASK.md"),
        reset_script=Path("reset_repo.py"),
        role="same_repository_reuse",
        initial_verification_should_pass=False,
        final_verification_should_pass=True,
    )
    passed = VerificationResult(command=["python", "-m", "pytest", "-q"], exit_code=0, passed=True)
    ProjectStateStore(state_root, workspace_root=workspace).append_session(
        "p1",
        {
            "tool_call_count": 5,
            "unsupported_shell_syntax_count": 1,
            "protocol_error_count": 2,
            "provider_error_count": 0,
            "input_tokens_total": 11,
            "output_tokens_total": 7,
            "total_tokens": 18,
        },
    )

    result = collect_case_result(
        mode="memory_skill",
        repeat=1,
        case=case,
        project_id="p1",
        project_status="candidate_complete",
        reset_exit_code=0,
        initial_verification=passed,
        final_verification=passed,
        state_root=state_root,
        knowledge_root=knowledge_root,
        workspace_root=workspace,
        wall_time_seconds=0.1,
        before_memory_ids={existing_memory.memory_id},
        before_skill_ids={existing_skill.skill_id},
    )

    assert result.created_memory_ids == [created_memory.memory_id]
    assert result.created_skill_ids == [created_skill.skill_id]
    assert result.available_memory_ids == sorted([existing_memory.memory_id, created_memory.memory_id])
    assert result.available_skill_ids == sorted([existing_skill.skill_id, created_skill.skill_id])
    assert result.tool_call_count == 5
    assert result.unsupported_shell_syntax_count == 1
    assert result.protocol_error_count == 2
    assert result.provider_error_count == 0


def _memory(title: str) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title=title,
        lesson="strip whitespace before validation",
        recommended_actions=["strip input"],
        avoid_actions=["accept whitespace-only input"],
        applicability_conditions=["pytest validation"],
        evidence_ids=["e1"],
    )


def _skill(title: str) -> SkillRecord:
    return SkillRecord(
        status=SkillStatus.VALIDATED,
        scope=MemoryScope.PORTABLE,
        title=title,
        summary="strip whitespace before validation",
        preconditions=["validator rejects whitespace-only input"],
        anti_conditions=["preserve leading whitespace"],
        procedure=[SkillStep(description="strip input", tool_name="write_file")],
        verification=["python -m pytest -q"],
        evidence_ids=["e1"],
    )
