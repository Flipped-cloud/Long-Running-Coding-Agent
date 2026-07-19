from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCase, ExperienceLearningCaseResult, VerificationResult
from longrun_agent.knowledge.schema import KnowledgeUseType, MemoryScope, MemoryStatus, SkillStatus
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.state.store import ProjectStateStore


def collect_case_result(
    *,
    mode: str,
    repeat: int,
    case: ExperienceLearningCase,
    project_id: str,
    project_status: str,
    reset_exit_code: int,
    initial_verification: VerificationResult,
    final_verification: VerificationResult,
    state_root: Path,
    knowledge_root: Path,
    workspace_root: Path,
    wall_time_seconds: float,
    before_memory_ids: set[str] | None = None,
    before_skill_ids: set[str] | None = None,
) -> ExperienceLearningCaseResult:
    state_store = ProjectStateStore(state_root, workspace_root=workspace_root)
    sessions = state_store.read_sessions(project_id)
    events = _read_jsonl(knowledge_root / "events.jsonl")
    uses = _read_jsonl(knowledge_root / "uses.jsonl")
    project_events = [event for event in events if event.get("project_id") in {None, project_id} or event.get("project_id") == project_id]
    project_uses = [use for use in uses if use.get("project_id") == project_id]
    first_successful_verification = _operations_to_first_successful_verification(sessions)
    changed_files = {
        path for session in sessions for path in [*(session.get("changed_files") or []), *(session.get("files_touched") or [])] if path
    }
    exposed_memory_ids = _event_ids(events, "memory_exposed", "memory_id", project_id=project_id)
    exposed_skill_ids = _event_ids(events, "skill_exposed", "skill_id", project_id=project_id)
    referenced_memory_ids = _use_ids(project_uses, "memory_id", KnowledgeUseType.REFERENCED)
    referenced_skill_ids = _use_ids(project_uses, "skill_id", KnowledgeUseType.REFERENCED)
    helpful_memory_ids = _use_ids(project_uses, "memory_id", KnowledgeUseType.HELPFUL)
    harmful_memory_ids = _use_ids(project_uses, "memory_id", KnowledgeUseType.HARMFUL)
    helpful_skill_ids = _use_ids(project_uses, "skill_id", KnowledgeUseType.HELPFUL)
    harmful_skill_ids = _use_ids(project_uses, "skill_id", KnowledgeUseType.HARMFUL)
    episode_dir = state_root / project_id / "knowledge" / "episodes"
    episode_ids = sorted(path.stem for path in episode_dir.glob("*.json")) if episode_dir.exists() else []
    available_memory_ids = []
    available_skill_ids = []
    if mode != "disabled":
        store = _store(knowledge_root, workspace_root)
        available_memory_ids = sorted(memory.memory_id for memory in store.list_memories())
        skills = store.list_skills()
        available_skill_ids = sorted(skill.skill_id for skill in skills)
    else:
        skills = []
    created_memory_ids = sorted(set(available_memory_ids) - (before_memory_ids or set()))
    created_skill_ids = sorted(set(available_skill_ids) - (before_skill_ids or set()))
    skill_by_id = {skill.skill_id: skill for skill in skills}
    session_fingerprint = _session_repository_fingerprint(sessions)
    cross_repository_skill_reference_count = sum(
        1
        for skill_id in referenced_skill_ids
        if skill_id in skill_by_id
        and skill_by_id[skill_id].scope == MemoryScope.PORTABLE
        and str(skill_by_id[skill_id].repository_fingerprint or "") not in {"", session_fingerprint}
    )
    cross_repository_skill_helpful_count = sum(
        1
        for skill_id in helpful_skill_ids
        if skill_id in skill_by_id
        and skill_by_id[skill_id].scope == MemoryScope.PORTABLE
        and str(skill_by_id[skill_id].repository_fingerprint or "") not in {"", session_fingerprint}
    )
    rejection_reasons = [
        str(event.get("reason") or "")
        for event in project_events
        if event.get("event_type") == "reflection_evidence_rejected" and event.get("reason")
    ]
    return ExperienceLearningCaseResult(
        mode=mode,
        repeat=repeat,
        case_id=case.case_id,
        role=case.role,
        knowledge_phase=case.knowledge_phase,
        repository=str(case.repository),
        project_id=project_id,
        project_status=project_status,
        reset_exit_code=reset_exit_code,
        initial_verification=initial_verification,
        final_verification=final_verification,
        exposed_memory_ids=exposed_memory_ids,
        exposed_skill_ids=exposed_skill_ids,
        referenced_memory_ids=referenced_memory_ids,
        referenced_skill_ids=referenced_skill_ids,
        helpful_memory_ids=helpful_memory_ids,
        harmful_memory_ids=harmful_memory_ids,
        helpful_skill_ids=helpful_skill_ids,
        harmful_skill_ids=harmful_skill_ids,
        episode_ids=episode_ids,
        created_memory_ids=created_memory_ids,
        created_skill_ids=created_skill_ids,
        available_memory_ids=available_memory_ids,
        available_skill_ids=available_skill_ids,
        session_count=len(sessions),
        tool_call_count=sum(int(session.get("tool_call_count") or 0) for session in sessions),
        unsupported_shell_syntax_count=sum(int(session.get("unsupported_shell_syntax_count") or 0) for session in sessions),
        protocol_error_count=sum(int(session.get("protocol_error_count") or 0) for session in sessions),
        recoverable_protocol_error_count=sum(int(session.get("recoverable_protocol_error_count") or 0) for session in sessions),
        fatal_protocol_error_count=sum(int(session.get("fatal_protocol_error_count") or 0) for session in sessions),
        provider_error_count=sum(int(session.get("provider_error_count") or 0) for session in sessions),
        reflection_rejection_count=len(rejection_reasons),
        reflection_rejection_reasons=rejection_reasons,
        reflection_acceptance_count=sum(
            1
            for event in project_events
            if event.get("event_type") == "reflection_candidate_created" and event.get("status_after") == "accepted"
        ),
        portable_skill_count=sum(1 for skill in skills if skill.scope == MemoryScope.PORTABLE),
        repository_skill_count=sum(1 for skill in skills if skill.scope == MemoryScope.REPOSITORY),
        skills_created_without_source_memory_count=sum(
            1 for skill in skills if skill.skill_id in created_skill_ids and not skill.source_memory_ids
        ),
        cross_repository_skill_reference_count=cross_repository_skill_reference_count,
        cross_repository_skill_helpful_count=cross_repository_skill_helpful_count,
        input_tokens=sum(int(session.get("input_tokens_total") or 0) for session in sessions),
        output_tokens=sum(int(session.get("output_tokens_total") or 0) for session in sessions),
        total_tokens=sum(int(session.get("total_tokens") or 0) for session in sessions),
        wall_time_seconds=wall_time_seconds,
        knowledge_error=any(event.get("event_type") == "knowledge_error" for event in project_events),
        modified_after_run=bool(changed_files),
        operations_to_first_successful_verification=first_successful_verification,
        changed_file_count=len(changed_files),
        invalid_or_repeated_tool_call_count=sum(
            int(session.get("protocol_error_count") or 0) + len(session.get("repeated_tool_calls") or []) for session in sessions
        ),
    )


def _operations_to_first_successful_verification(sessions: list[dict]) -> int | None:
    operation_offset = 0
    for session in sessions:
        successful = [
            int(observation.get("operation_index") or 0)
            for observation in session.get("bash_observations") or []
            if observation.get("is_verification") and observation.get("exit_code") == 0
        ]
        if successful:
            return operation_offset + min(successful)
        operation_offset += int(session.get("tool_call_count") or 0)
    return None


def collect_store_counts(knowledge_root: Path, workspace_root: Path) -> dict[str, int]:
    store = _store(knowledge_root, workspace_root)
    events = _read_jsonl(knowledge_root / "events.jsonl")
    memories = store.list_memories()
    skills = store.list_skills()
    uses = _read_jsonl(knowledge_root / "uses.jsonl")
    return {
        "reflection_candidate_count": sum(1 for event in events if event.get("event_type") == "reflection_candidate_created"),
        "active_memory_count": sum(1 for memory in memories if memory.status == MemoryStatus.ACTIVE),
        "quarantined_memory_count": sum(1 for memory in memories if memory.status == MemoryStatus.QUARANTINED),
        "validated_skill_count": sum(1 for skill in skills if skill.status == SkillStatus.VALIDATED),
        "active_skill_count": sum(1 for skill in skills if skill.status == SkillStatus.ACTIVE),
        "portable_skill_count": sum(1 for skill in skills if skill.scope == MemoryScope.PORTABLE),
        "repository_skill_count": sum(1 for skill in skills if skill.scope == MemoryScope.REPOSITORY),
        "skills_created_without_source_memory_count": sum(1 for skill in skills if not skill.source_memory_ids),
        "memory_retrieval_count": sum(
            1 for use in uses if use.get("memory_id") and use.get("use_type") == KnowledgeUseType.RETRIEVED.value
        ),
        "memory_exposure_count": sum(1 for use in uses if use.get("memory_id") and use.get("use_type") == KnowledgeUseType.EXPOSED.value),
        "memory_reference_count": sum(
            1 for use in uses if use.get("memory_id") and use.get("use_type") == KnowledgeUseType.REFERENCED.value
        ),
        "memory_helpful_count": sum(1 for use in uses if use.get("memory_id") and use.get("use_type") == KnowledgeUseType.HELPFUL.value),
        "memory_harmful_count": sum(1 for use in uses if use.get("memory_id") and use.get("use_type") == KnowledgeUseType.HARMFUL.value),
        "skill_retrieval_count": sum(1 for use in uses if use.get("skill_id") and use.get("use_type") == KnowledgeUseType.RETRIEVED.value),
        "skill_exposure_count": sum(1 for use in uses if use.get("skill_id") and use.get("use_type") == KnowledgeUseType.EXPOSED.value),
        "skill_reference_count": sum(1 for use in uses if use.get("skill_id") and use.get("use_type") == KnowledgeUseType.REFERENCED.value),
        "skill_helpful_count": sum(1 for use in uses if use.get("skill_id") and use.get("use_type") == KnowledgeUseType.HELPFUL.value),
        "skill_harmful_count": sum(1 for use in uses if use.get("skill_id") and use.get("use_type") == KnowledgeUseType.HARMFUL.value),
        "knowledge_error_count": sum(1 for event in events if event.get("event_type") == "knowledge_error"),
        "reflection_rejection_count": sum(1 for event in events if event.get("event_type") == "reflection_evidence_rejected"),
        "reflection_rejection_reasons": [
            str(event.get("reason") or "")
            for event in events
            if event.get("event_type") == "reflection_evidence_rejected" and event.get("reason")
        ],
        "reflection_acceptance_count": sum(
            1 for event in events if event.get("event_type") == "reflection_candidate_created" and event.get("status_after") == "accepted"
        ),
    }


def _store(knowledge_root: Path, workspace_root: Path) -> KnowledgeStore:
    return KnowledgeStore(knowledge_root, workspace_root=workspace_root)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _event_ids(events: list[dict], event_type: str, field: str, *, project_id: str | None = None) -> list[str]:
    ids: list[str] = []
    for event in events:
        if event.get("event_type") != event_type:
            continue
        if project_id is not None and event.get("project_id") != project_id:
            continue
        raw = event.get(field)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if value and value not in ids:
                ids.append(value)
    return ids


def _use_ids(uses: list[dict], field: str, use_type: KnowledgeUseType) -> list[str]:
    ids: list[str] = []
    for use in uses:
        value = use.get(field)
        if value and use.get("use_type") == use_type.value and value not in ids:
            ids.append(value)
    return ids


def _session_repository_fingerprint(sessions: list[dict]) -> str:
    for session in sessions:
        candidate = session.get("repository_fingerprint")
        if candidate:
            return str(candidate)
    return ""
