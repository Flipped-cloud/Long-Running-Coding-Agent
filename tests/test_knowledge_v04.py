from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from longrun_agent.config import AppConfig, KnowledgeConfig, ModelConfig, load_config
from longrun_agent.context.assembler import render_current_instruction, render_task_anchor
from longrun_agent.context.schema import TaskContextSeed
from longrun_agent.exceptions import ConfigurationError, ProviderError
from longrun_agent.knowledge.conflict import MemoryConflictDetector
from longrun_agent.knowledge.evidence import build_experience_pack
from longrun_agent.knowledge.memory_lifecycle import MemoryLifecycleManager, ReflectionEvidenceGate
from longrun_agent.knowledge.reflection import ReflectionGenerator
from longrun_agent.knowledge.renderer import render_bundle
from longrun_agent.knowledge.retrieval import retrieve_bundle
from longrun_agent.knowledge.schema import (
    ExperienceEvidenceItem,
    ExperienceEvidencePack,
    KnowledgeConflictRecord,
    KnowledgeRetrievalQuery,
    KnowledgeUseType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    ReflectionCandidate,
    RetrievedKnowledgeBundle,
    SkillCandidateDraft,
    SkillRecord,
    SkillStatus,
    SkillStep,
)
from longrun_agent.knowledge.skill_lifecycle import SkillCandidateGenerator, SkillLifecycleManager, SkillValidator
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.knowledge.tools import KnowledgeUseChannel, ReportKnowledgeUseArgs, ReportKnowledgeUseTool
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus
from longrun_agent.tools.base import ToolContext


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths=['tests']\n", encoding="utf-8")
    (root / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def _store(tmp_path: Path, workspace: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "knowledge", workspace_root=workspace)


def _pack(tmp_path: Path, *, success: bool = True, failed: bool = False, blocker: bool = False) -> ExperienceEvidencePack:
    workspace = _workspace(tmp_path)
    session = {
        "session_id": "s1",
        "run_id": "s1",
        "run_status": "completed" if success else "max_steps_reached",
        "changed_files": [str(workspace / "service.py")] if success else [],
        "read_files": [str(workspace / "service.py")],
        "bash_commands": ["pytest -q"],
        "bash_exit_codes": [0 if success and not failed else 1],
        "bash_observations": [
            {
                "command": "pytest -q",
                "argv": ["pytest", "-q"],
                "exit_code": 0 if success and not failed else 1,
                "success": success and not failed,
                "is_verification": True,
                "output_excerpt": "AssertionError: assert True is False" if not success or failed else "1 passed",
                "error_type": None,
                "artifact_path": "tool-output.txt",
                "operation_index": 2,
            }
        ],
        "successful_test_commands": ["pytest -q"] if success and not failed else [],
        "terminal_signal": "blocker" if blocker else ("completion_request" if success else None),
    }
    return build_experience_pack(
        project_id="p1",
        task_id="t1",
        task_objective="fix pytest service validation",
        acceptance_criteria=["pytest passes"],
        session_record=session,
        plan_version=1,
        starting_task_status="in_progress",
        ending_task_status="candidate_complete" if success else "in_progress",
        workspace_root=workspace,
    )


def _candidate(pack: ExperienceEvidencePack, *, evidence_ids: list[str] | None = None, confidence: float = 0.8) -> ReflectionCandidate:
    failure_ids = [
        item.evidence_id
        for item in pack.evidence_items
        if (item.event_type == "test_command" and item.success is False) or (item.exit_code is not None and item.exit_code != 0)
    ]
    ids = evidence_ids or failure_ids or [pack.evidence_items[0].evidence_id]
    return ReflectionCandidate(
        episode_id=pack.episode_id,
        failure_category="implementation_error",
        root_cause="service validation used stale pytest evidence",
        recommended_actions=["rerun pytest before reporting completion"],
        avoid_actions=["avoid trusting stale pytest output"],
        applicability_conditions=["pytest task has validation uncertainty"],
        proposed_scope=MemoryScope.REPOSITORY,
        tags=["pytest"],
        evidence_ids=ids,
        confidence=confidence,
    )


def _memory(*, status: MemoryStatus = MemoryStatus.ACTIVE, project_id: str | None = "p1") -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PROJECT,
        status=status,
        title="Pytest validation",
        lesson="service validation should rerun pytest",
        recommended_actions=["rerun pytest before completion"],
        avoid_actions=["avoid stale pytest output"],
        applicability_conditions=["pytest validation task"],
        anti_conditions=["task is documentation only"],
        tags=["pytest", "python"],
        project_id=project_id,
        evidence_ids=["e1"],
        confidence=0.9,
    )


def _skill(*, status: SkillStatus = SkillStatus.ACTIVE) -> SkillRecord:
    return SkillRecord(
        status=status,
        title="Rerun pytest workflow",
        summary="Inspect changed files and rerun pytest before completion.",
        preconditions=["pytest validation task"],
        anti_conditions=["no tests are available"],
        procedure=[
            SkillStep(description="inspect file", tool_name="read_file"),
            SkillStep(description="run pytest", tool_name="bash", command="pytest -q"),
        ],
        verification=["pytest -q"],
        tags=["pytest", "python"],
        repository_fingerprint="repo-a",
        source_episode_ids=["ep1"],
        source_task_ids=["t1"],
        evidence_ids=["e1"],
        success_count=2,
        distinct_task_ids=["t1", "t2"],
        distinct_repository_fingerprints=["repo-a"],
    )


def _skill_args(pack: ExperienceEvidencePack, memory_id: str = "MEM-source") -> dict:
    return {
        "title": "Whitespace validation repair",
        "summary": "Repair validators by stripping textual input before boolean validation.",
        "scope": "portable",
        "preconditions": ["A validator accepts textual names or commands."],
        "anti_conditions": ["preserve leading whitespace"],
        "parameters": ["target validator"],
        "procedure": [
            {"description": "Inspect the target validator.", "tool_name": "read_file"},
            {"description": "Apply a minimal strip-before-bool change.", "tool_name": "write_file"},
            {"description": "Run pytest.", "tool_name": "bash", "command": "pytest -q"},
        ],
        "verification": ["pytest -q"],
        "stop_conditions": ["Stop if exact literal formatting must be preserved."],
        "tags": ["pytest"],
        "evidence_ids": [pack.evidence_items[0].evidence_id],
        "source_memory_ids": [memory_id],
        "confidence": 0.86,
    }


def _knowledge_session_record(workspace: Path, *, changed_files: list[str]) -> dict:
    return {
        "project_id": "p1",
        "task_id": "t1",
        "session_id": "s1",
        "run_id": "s1",
        "task_attempt": 1,
        "run_status": "completed",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "duration_seconds": 0.1,
        "repository_fingerprint": "",
        "steps": 1,
        "tool_call_count": 1,
        "total_tokens": 0,
        "terminal_signal": "completion_request",
        "files_touched": [],
        "read_files": [str(workspace / "service.py")],
        "written_files": [],
        "changed_files": changed_files,
        "bash_commands": ["pytest -q"],
        "bash_exit_codes": [0],
        "bash_observations": [
            {
                "command": "pytest -q",
                "argv": ["pytest", "-q"],
                "exit_code": 0,
                "success": True,
                "is_verification": True,
                "output_excerpt": "1 passed",
                "error_type": None,
                "artifact_path": "tool-output.txt",
                "operation_index": 1,
            }
        ],
        "successful_test_commands": ["pytest -q"],
        "successful_acceptance_commands": [],
        "repeated_tool_calls": [],
        "suppressed_tool_calls": [],
        "terminal_grace_turn_count": 0,
        "terminal_signal_recovered": False,
        "auto_completion_recovered": False,
        "completion_candidate": None,
        "unsupported_shell_syntax_count": 0,
        "protocol_error_count": 0,
        "recoverable_protocol_error_count": 0,
        "fatal_protocol_error_count": 0,
        "provider_error_count": 0,
        "tool_argument_protocol_retry_count": 0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "compactor_input_tokens": 0,
        "compactor_output_tokens": 0,
        "max_estimated_input_tokens": 0,
        "max_actual_input_tokens": 0,
        "max_context_usage_ratio": 0,
        "context_segment_count": 1,
        "context_reset_count": 0,
        "deterministic_prune_count": 0,
        "structured_compaction_count": 0,
        "pruned_item_count": 0,
        "stale_item_count": 0,
        "superseded_item_count": 0,
        "estimated_tokens_removed": 0,
        "context_budget_exhausted": False,
        "latest_context_handoff_id": None,
        "no_progress": False,
        "handoff_summary": None,
    }


def test_knowledge_config_modes_and_validation() -> None:
    for name in [
        "knowledge_disabled.yaml",
        "knowledge_raw_episode.yaml",
        "knowledge_reflection.yaml",
        "knowledge_verified_memory.yaml",
        "knowledge_memory_skill.yaml",
    ]:
        assert load_config(Path("configs") / name).knowledge.mode
    with pytest.raises(ValidationError, match="auto_execute"):
        KnowledgeConfig(skill={"auto_execute": True})
    with pytest.raises(ValidationError, match="positive knowledge retrieval weight"):
        KnowledgeConfig(
            retrieval={
                key: 0 for key in ["lexical_weight", "tag_weight", "scope_weight", "evidence_weight", "utility_weight", "freshness_weight"]
            }
        )


def test_evidence_pack_is_deterministic_and_sanitized(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=True)
    assert pack.files_changed == ["service.py"]
    assert pack.successful_verifications == ["pytest -q"]
    assert all(not Path(item.path or "").is_absolute() for item in pack.evidence_items)
    assert len({item.evidence_id for item in pack.evidence_items}) == len(pack.evidence_items)


def test_evidence_gate_accepts_supported_candidate_and_rejects_unknown_evidence(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    gate = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory", reflection={"minimum_confidence": 0.7}))
    accepted = gate.evaluate(_candidate(pack), pack)
    assert accepted.accepted
    rejected = gate.evaluate(_candidate(pack, evidence_ids=["missing"]), pack)
    assert rejected.decision == "rejected"


def test_reflection_mode_persists_candidate_without_active_promotion(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path)
    manager = MemoryLifecycleManager(KnowledgeConfig(mode="reflection"), store)
    record = manager.persist_candidate(_candidate(pack, evidence_ids=["missing"], confidence=0.1), pack)
    assert record is not None
    assert record.status == MemoryStatus.CANDIDATE


def test_verified_memory_gate_quarantines_low_confidence(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path, success=False)
    manager = MemoryLifecycleManager(KnowledgeConfig(mode="verified_memory", reflection={"minimum_confidence": 0.9}), store)
    record = manager.persist_candidate(_candidate(pack, confidence=0.6), pack)
    assert record is not None
    assert record.status == MemoryStatus.QUARANTINED


def test_store_rejects_workspace_internal_knowledge_root_and_saves_episode(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(ConfigurationError, match="inside the agent workspace"):
        KnowledgeStore(workspace / "knowledge", workspace_root=workspace)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path)
    path = store.save_episode(tmp_path / "project_state", pack)
    assert path.name == f"{pack.episode_id}.json"
    assert path.parent.name == "episodes"
    assert json.loads(path.read_text(encoding="utf-8"))["episode_id"] == pack.episode_id


def test_retrieval_filters_status_scope_anti_conditions_and_limits(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    store.save_memory(_memory())
    store.save_memory(_memory(status=MemoryStatus.CANDIDATE))
    store.save_memory(_memory(status=MemoryStatus.INVALIDATED))
    query = KnowledgeRetrievalQuery(task_objective="pytest validation service", project_id="p1", language_tags=["python"])
    bundle, scores = retrieve_bundle(KnowledgeConfig(mode="verified_memory", memory={"max_retrieved": 1}), store, query)
    assert len(bundle.memories) == 1
    assert bundle.memories[0].status == MemoryStatus.ACTIVE
    assert scores[0].total_score > 0
    blocked_query = query.model_copy(update={"task_objective": "task is documentation only"})
    blocked_bundle, _scores = retrieve_bundle(KnowledgeConfig(mode="verified_memory"), store, blocked_query)
    assert not blocked_bundle.memories


def test_rendered_context_stays_within_budget_and_contains_ids(tmp_path: Path) -> None:
    memory = _memory()
    skill = _skill()
    text, tokens = render_bundle(
        RetrievedKnowledgeBundle(memories=[memory], skills=[skill]),
        KnowledgeConfig(mode="memory_skill", memory={"max_context_tokens": 200}, skill={"max_context_tokens": 200}),
    )
    assert memory.memory_id in text
    assert skill.skill_id in text
    assert tokens <= 400


def test_skill_candidate_validation_promotion_and_deprecation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    manager = SkillLifecycleManager(
        KnowledgeConfig(mode="memory_skill", skill={"promotion_min_successes": 2, "promotion_min_distinct_tasks": 2}), store
    )
    pack = _pack(tmp_path)
    draft_args = _skill_args(pack)
    draft_args.update({"scope": "repository", "source_memory_ids": []})
    draft = SkillCandidateDraft.model_validate(draft_args)
    skill = manager.candidate_from_success(pack, draft=draft)
    assert skill is not None
    assert skill.status == SkillStatus.VALIDATED
    promoted = manager.record_success(skill.skill_id, task_id="t2", repository_fingerprint=skill.repository_fingerprint or "")
    assert promoted.status == SkillStatus.ACTIVE
    deprecated = manager.record_harmful_use(skill.skill_id, reason="negative transfer")
    assert deprecated.status == SkillStatus.DEPRECATED


def test_skill_validator_rejects_missing_verification_unknown_tool_and_dangerous_command(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    skill = _skill()
    skill.procedure[0].tool_name = "unknown_tool"
    valid, reasons = SkillValidator().validate(skill, pack)
    assert not valid
    assert "successful verification evidence required" in reasons
    assert "unknown tool in skill: unknown_tool" in reasons
    with pytest.raises(ValidationError, match="dangerous command"):
        SkillRecord(
            title="dangerous",
            summary="bad",
            preconditions=["x"],
            anti_conditions=["y"],
            procedure=[SkillStep(description="delete", tool_name="bash", command="rm -rf .")],
            verification=["pytest -q"],
        )


def test_report_knowledge_use_allows_only_exposed_ids(tmp_path: Path) -> None:
    channel = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=["s1"])
    context = ToolContext(tmp_path, knowledge_channel=channel)
    tool = ReportKnowledgeUseTool()
    ok = tool.execute(
        "call1", ReportKnowledgeUseArgs(memory_ids=["m1"], skill_ids=["s1"], use_type=KnowledgeUseType.REFERENCED, reason="used"), context
    )
    assert ok.success
    rejected_attribution = tool.execute(
        "call2", ReportKnowledgeUseArgs(memory_ids=["m1"], use_type=KnowledgeUseType.HELPFUL, reason="model judged helpful"), context
    )
    assert not rejected_attribution.success
    bad = tool.execute("call2", ReportKnowledgeUseArgs(memory_ids=["m2"], reason="unknown"), context)
    assert not bad.success
    assert len(channel.records) == 1
    assert channel.decision_recorded is True
    assert channel.referenced_memory_ids == {"m1"}
    assert channel.referenced_skill_ids == {"s1"}


def test_report_knowledge_use_records_reviewed_not_used_decision(tmp_path: Path) -> None:
    channel = KnowledgeUseChannel(exposed_memory_ids=["m1"], exposed_skill_ids=["s1"])
    result = ReportKnowledgeUseTool().execute(
        "call1",
        ReportKnowledgeUseArgs(memory_ids=[], skill_ids=[], reason="retrieved knowledge conflicts with literal formatting requirement"),
        ToolContext(tmp_path, knowledge_channel=channel),
    )
    assert result.success
    assert channel.decision_recorded is True
    assert channel.records == []
    assert channel.not_used_reason == "retrieved knowledge conflicts with literal formatting requirement"


def test_reviewed_not_used_decision_is_persisted_by_orchestrator(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    cfg = load_config(Path("configs/planning_static.yaml"))
    cfg.workspace.root = workspace
    cfg.telemetry.run_root = tmp_path / "runs"
    cfg.state.root = tmp_path / "projects"
    cfg.planning.initial_plan.min_tasks = 1
    cfg.planning.initial_plan.max_tasks = 1
    cfg.planning.execution.max_project_sessions = 1
    cfg.planning.execution.max_sessions_per_task = 1
    cfg.planning.execution.final_verification_command = []
    cfg.knowledge.mode = "verified_memory"
    cfg.knowledge.root = tmp_path / "knowledge"
    store = KnowledgeStore(cfg.knowledge.root, workspace_root=workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.PORTABLE
    memory.title = "Finish task guidance"
    memory.lesson = "finish task after checking evidence"
    memory.applicability_conditions = ["finish task"]
    memory.anti_conditions = ["task already has contradictory evidence"]
    store.save_memory(memory)
    provider = FakeModelProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="plan",
                        name="submit_plan",
                        arguments={
                            "project_summary": "summary",
                            "tasks": [
                                {
                                    "key": "T1",
                                    "title": "Finish",
                                    "objective": "finish task",
                                    "acceptance_criteria": ["done"],
                                    "depends_on_keys": [],
                                }
                            ],
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="k1",
                        name="report_knowledge_use",
                        arguments={"memory_ids": [], "skill_ids": [], "reason": "reviewed but not relevant"},
                    ),
                    ToolCall(
                        id="c1",
                        name="request_task_completion",
                        arguments={"summary": "done", "acceptance_criteria_addressed": ["done"]},
                    ),
                ]
            ),
        ]
    )

    ProjectOrchestrator(cfg, provider, project_id="p-review").start("finish task")

    events = store.read_jsonl(store.events_path)
    reviewed = [event for event in events if event.get("event_type") == "knowledge_reviewed_not_used"]
    assert reviewed
    assert reviewed[-1]["memory_ids"] == [memory.memory_id]
    assert reviewed[-1]["reason"] == "reviewed but not relevant"


def test_context_injection_mentions_verification_requirement() -> None:
    seed = TaskContextSeed(
        project_id="p1",
        project_objective="project",
        plan_version=1,
        task_id="t1",
        task_title="Fix pytest",
        task_objective="fix pytest",
        task_status="in_progress",
        acceptance_criteria=["pytest passes"],
        attempts=1,
        knowledge_context="- Memory MEM-1: rerun pytest",
    )
    assert "Retrieved knowledge:" in render_task_anchor(seed)
    instruction = render_current_instruction(seed, repeat_anchor=True)
    assert "verify with tools" in instruction
    assert "report_knowledge_use" in instruction


def test_duplicate_evidence_ids_are_rejected() -> None:
    item = ExperienceEvidenceItem(evidence_id="e1", project_id="p1", task_id="t1", session_id="s1", run_id="s1", event_type="bash")
    with pytest.raises(ValidationError, match="unique"):
        ExperienceEvidencePack(
            project_id="p1",
            task_id="t1",
            task_objective="x",
            session_id="s1",
            run_id="s1",
            plan_version=1,
            evidence_items=[item, item],
        )


def test_store_status_usage_filters_and_corrupt_records(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory()
    store.save_memory(memory)
    (store.memories_dir / "corrupt.json").write_text("{not-json", encoding="utf-8")
    assert [record.memory_id for record in store.list_memories({"status": "active"})] == [memory.memory_id]
    updated = store.update_memory_status(memory.memory_id, MemoryStatus.INVALIDATED, reason="stale")
    assert updated.status == MemoryStatus.INVALIDATED
    updated.status = MemoryStatus.ACTIVE
    store.save_memory(updated)
    for use_type in [
        KnowledgeUseType.RETRIEVED,
        KnowledgeUseType.EXPOSED,
        KnowledgeUseType.REFERENCED,
        KnowledgeUseType.HELPFUL,
        KnowledgeUseType.HARMFUL,
    ]:
        store.add_memory_usage(memory.memory_id, use_type)
    used = store.load_memory(memory.memory_id)
    assert used.retrieval_count == 1
    assert used.exposure_count == 1
    assert used.reference_count == 1
    assert used.helpful_count == 1
    assert used.harmful_count == 1
    assert len(store.read_jsonl(store.uses_path)) == 5
    store.uses_path.write_text(store.uses_path.read_text(encoding="utf-8") + "\n{bad", encoding="utf-8")
    assert len(store.read_jsonl(store.uses_path)) == 5

    skill = _skill(status=SkillStatus.VALIDATED)
    store.save_skill(skill)
    (store.skills_dir / "bad" / "skill.json").parent.mkdir(parents=True)
    (store.skills_dir / "bad" / "skill.json").write_text("{bad", encoding="utf-8")
    assert store.list_skills({"status": "validated"})[0].skill_id == skill.skill_id
    assert store.update_skill_status(skill.skill_id, SkillStatus.DEPRECATED).status == SkillStatus.DEPRECATED
    store.update_skill_status(skill.skill_id, SkillStatus.ACTIVE)
    for use_type in [KnowledgeUseType.RETRIEVED, KnowledgeUseType.EXPOSED, KnowledgeUseType.HELPFUL, KnowledgeUseType.HARMFUL]:
        store.add_skill_usage(skill.skill_id, use_type)
    loaded_skill = store.load_skill(skill.skill_id)
    assert loaded_skill.retrieval_count == 1
    assert loaded_skill.harmful_count == 1


def test_reflection_generator_retries_and_returns_none_on_protocol_failure(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    good_args = {
        "failure_category": "implementation_error",
        "root_cause": "pytest validation evidence was stale",
        "recommended_actions": ["rerun pytest before completion"],
        "avoid_actions": ["avoid stale pytest output"],
        "applicability_conditions": ["pytest validation task"],
        "evidence_ids": [pack.evidence_items[0].evidence_id],
        "confidence": 0.8,
    }
    model = FakeModelProvider(
        [
            ModelResponse(final_answer=FinalAnswer(content="no tool")),
            ModelResponse(tool_calls=[ToolCall(id="r1", name="submit_reflection_candidate", arguments=good_args)]),
        ]
    )
    candidate = ReflectionGenerator(model, KnowledgeConfig(mode="reflection").reflection).generate(pack, related_memory_ids=["m1"])
    assert candidate is not None
    assert candidate.episode_id == pack.episode_id
    assert model.calls == 2

    bad_model = FakeModelProvider([ModelResponse(tool_calls=[ToolCall(id="r2", name="submit_reflection_candidate", arguments={})])])
    assert (
        ReflectionGenerator(bad_model, KnowledgeConfig(mode="reflection", reflection={"max_protocol_retries": 0}).reflection).generate(pack)
        is None
    )


def test_evidence_gate_rejects_leaks_conflicts_and_success_as_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path)
    gate = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory"), store)

    leak = _candidate(pack)
    leak.proposed_scope = MemoryScope.PORTABLE
    leak.recommended_actions.append(r"inspect C:\Users\hp\secret.txt")
    assert gate.evaluate(leak, pack).decision == "rejected"

    success_as_failure = _candidate(pack)
    success_as_failure.root_cause = "pytest validation failed despite successful verification"
    success_as_failure.recommended_actions = ["pytest failed, inspect output"]
    assert gate.evaluate(success_as_failure, pack).decision == "rejected"

    existing = _memory()
    existing.avoid_actions = ["rerun pytest before reporting completion"]
    store.save_memory(existing)
    conflicted = gate.evaluate(_candidate(pack), pack)
    assert conflicted.decision == "rejected"
    assert conflicted.conflict_ids


def test_portable_memory_requires_anti_conditions_and_downgrades_source_specific_text(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    gate = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory"))
    no_anti = _candidate(pack)
    no_anti.proposed_scope = MemoryScope.PORTABLE
    no_anti.anti_conditions = []
    assert gate.evaluate(no_anti, pack).decision == "rejected"

    source_specific = _candidate(pack)
    source_specific.proposed_scope = MemoryScope.PORTABLE
    source_specific.anti_conditions = ["preserve leading whitespace"]
    source_specific.recommended_actions = ["Edit task_app.py and rerun pytest."]
    result = gate.evaluate(source_specific, pack)
    assert result.accepted
    assert result.final_scope == MemoryScope.REPOSITORY


def test_generic_strip_call_does_not_downgrade_portable_memory(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    candidate = _candidate(pack)
    candidate.proposed_scope = MemoryScope.PORTABLE
    candidate.anti_conditions = ["preserve leading whitespace"]
    candidate.recommended_actions = ["Call strip() before checking whether textual input is empty."]

    result = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory")).evaluate(candidate, pack)

    assert result.accepted
    assert result.final_scope == MemoryScope.PORTABLE


def test_repository_function_name_downgrades_portable_memory(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    candidate = _candidate(pack)
    candidate.proposed_scope = MemoryScope.PORTABLE
    candidate.anti_conditions = ["preserve leading whitespace"]
    candidate.recommended_actions = ["Call validate_task_name() before checking whether textual input is empty."]

    result = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory")).evaluate(candidate, pack)

    assert result.accepted
    assert result.final_scope == MemoryScope.REPOSITORY


def test_source_filename_downgrades_portable_memory(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    candidate = _candidate(pack)
    candidate.proposed_scope = MemoryScope.PORTABLE
    candidate.anti_conditions = ["preserve leading whitespace"]
    candidate.recommended_actions = ["Update task_app.py before checking whether textual input is empty."]

    result = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory")).evaluate(candidate, pack)

    assert result.accepted
    assert result.final_scope == MemoryScope.REPOSITORY


def test_evidence_gate_accepts_natural_root_cause_with_failed_evidence(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    gate = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory"))
    candidate = _candidate(pack)
    candidate.root_cause = "Blank-only input survives the truthiness check because normalization is omitted."
    candidate.proposed_scope = MemoryScope.PORTABLE
    candidate.anti_conditions = ["preserve leading whitespace"]

    result = gate.evaluate(candidate, pack)

    assert result.accepted


def test_evidence_gate_rejects_candidate_that_only_cites_read_evidence(tmp_path: Path) -> None:
    pack = _pack(tmp_path, success=False)
    gate = ReflectionEvidenceGate(KnowledgeConfig(mode="verified_memory"))
    read_id = next(item.evidence_id for item in pack.evidence_items if item.event_type == "read_file")
    candidate = _candidate(pack, evidence_ids=[read_id])

    result = gate.evaluate(candidate, pack)

    assert not result.accepted
    assert "candidate must cite at least one failure evidence item" in result.rejection_reasons


def test_rejected_reflection_writes_diagnostic_artifact(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path, success=False)
    read_id = next(item.evidence_id for item in pack.evidence_items if item.event_type == "read_file")
    candidate = _candidate(pack, evidence_ids=[read_id])

    assert MemoryLifecycleManager(KnowledgeConfig(mode="verified_memory"), store).persist_candidate(candidate, pack) is None

    path = store.rejections_dir / f"{candidate.candidate_id}.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == candidate.candidate_id
    assert payload["cited_evidence_ids"] == [read_id]
    assert "candidate must cite at least one failure evidence item" in payload["rejection_reasons"]


def test_memory_lifecycle_should_reflect_and_duplicate_merge(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    config = KnowledgeConfig(mode="verified_memory")
    manager = MemoryLifecycleManager(config, store)
    pack = _pack(tmp_path, success=False)
    assert manager.should_reflect(pack)[0]
    provider_error = pack.model_copy(update={"run_status": "provider_error"})
    assert manager.should_reflect(provider_error) == (False, "provider error does not trigger reflection")
    context_reset = pack.model_copy(
        update={"run_status": "completed", "no_progress": False, "context_reset_count": 1, "failed_verifications": []}
    )
    assert manager.should_reflect(context_reset) == (False, "context reset alone does not trigger reflection")
    blocked = pack.model_copy(update={"blockers": ["blocked"]})
    assert manager.should_reflect(blocked) == (True, "blocked")

    success_pack = _pack(tmp_path, success=False)
    existing = _memory()
    existing.lesson = "service validation used stale pytest evidence"
    existing.recommended_actions = ["rerun pytest before reporting completion"]
    store.save_memory(existing)
    merged = manager.persist_candidate(_candidate(success_pack), success_pack)
    assert merged is not None
    assert merged.memory_id == existing.memory_id
    assert success_pack.episode_id in merged.source_episode_ids


def test_retrieval_skills_repository_expiry_disabled_and_conflict_penalty(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.repository_fingerprint = "repo-a"
    memory.updated_at = "not-a-date"
    store.save_memory(memory)
    expired = _memory(project_id=None)
    expired.scope = MemoryScope.PORTABLE
    expired.expires_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.save_memory(expired)
    skill = _skill(status=SkillStatus.VALIDATED)
    store.save_skill(skill)
    query = KnowledgeRetrievalQuery(
        task_objective="pytest validation workflow",
        repository_fingerprint="repo-a",
        language_tags=["python"],
        tool_tags=["pytest"],
    )
    bundle, scores = retrieve_bundle(KnowledgeConfig(mode="memory_skill"), store, query)
    assert [record.memory_id for record in bundle.memories] == [memory.memory_id]
    assert [record.skill_id for record in bundle.skills] == [skill.skill_id]
    assert any(score.item_id == skill.skill_id for score in scores)
    assert retrieve_bundle(KnowledgeConfig(mode="disabled"), store, query)[0].memories == []
    assert retrieve_bundle(KnowledgeConfig(mode="memory_skill", skill={"enabled": False}), store, query)[0].skills == []

    store.append_conflict(
        KnowledgeConflictRecord(
            left_id=memory.memory_id,
            right_id="other",
            conflict_type="action_conflict",
            severity="medium",
            reason="conflicting guidance",
        )
    )
    penalized = MemoryConflictDetector(store).conflict_risk(memory)
    assert penalized > 0


def test_portable_skill_transfers_by_preconditions_and_respects_anti_conditions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    skill = _skill(status=SkillStatus.VALIDATED)
    skill.scope = MemoryScope.PORTABLE
    skill.repository_fingerprint = "repo-a"
    skill.preconditions = [
        "A validator accepts textual names or commands.",
        "Tests require whitespace-only input to be rejected.",
    ]
    skill.anti_conditions = ["preserve leading whitespace", "indentation is semantically meaningful"]
    store.save_skill(skill)

    config = KnowledgeConfig(mode="memory_skill", skill={"portable_minimum_semantic_score": 0.25})
    transfer_query = KnowledgeRetrievalQuery(
        task_objective="Fix shell command validation so blank commands are rejected after trimming.",
        acceptance_criteria=["pytest passes"],
        repository_fingerprint="repo-b",
        language_tags=["python"],
        tool_tags=["pytest"],
    )
    bundle, scores = retrieve_bundle(config, store, transfer_query)
    assert [record.skill_id for record in bundle.skills] == [skill.skill_id]
    assert next(score for score in scores if score.item_id == skill.skill_id).lexical_score < 0.25

    negative_query = transfer_query.model_copy(
        update={"task_objective": "Preserve leading whitespace because indentation is semantically meaningful."}
    )
    negative_bundle, _scores = retrieve_bundle(config, store, negative_query)
    assert negative_bundle.skills == []


def test_portable_skill_inherits_source_memory_anti_conditions_and_rejects_leaky_draft(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.PORTABLE
    memory.anti_conditions = ["preserve leading whitespace", "exact literal formatting is required"]
    store.save_memory(memory)
    pack = _pack(tmp_path)
    draft = SkillCandidateDraft(
        title="Edit task_app.py directly",
        summary="Use task_app.py to repair validators.",
        scope=MemoryScope.PORTABLE,
        preconditions=["validator rejects whitespace-only input"],
        anti_conditions=["indentation is semantically meaningful"],
        procedure=[SkillStep(description="Patch task_app.py", tool_name="write_file")],
        verification=["pytest -q"],
        evidence_ids=[pack.evidence_items[0].evidence_id],
        source_memory_ids=[memory.memory_id],
        confidence=0.9,
    )

    skill = SkillLifecycleManager(KnowledgeConfig(mode="memory_skill"), store).candidate_from_success(
        pack, draft=draft, source_memory_ids=[memory.memory_id]
    )

    assert skill is None


def test_skill_generator_accepts_tool_call_dict_arguments(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    args = _skill_args(pack)
    provider = FakeModelProvider([ModelResponse(tool_calls=[ToolCall(id="s1", name="submit_skill_candidate", arguments=args)])])

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.draft is not None
    assert result.generation_origin == "tool_call"
    assert result.draft.source_memory_ids == ["MEM-source"]


def test_skill_generator_accepts_tool_call_json_string_arguments(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    args = _skill_args(pack)
    call = ToolCall.model_construct(id="s1", name="submit_skill_candidate", arguments=json.dumps(args))
    provider = FakeModelProvider([ModelResponse(tool_calls=[call])])

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.draft is not None
    assert result.generation_origin == "deterministic_repair"
    assert "decoded_json_string_arguments" in result.repair_actions


def test_skill_generator_accepts_plain_json_content(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content=json.dumps(_skill_args(pack))))])

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.generation_origin == "content_json"


def test_skill_generator_accepts_fenced_json_content(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    content = "```json\n" + json.dumps(_skill_args(pack)) + "\n```"
    provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content=content))])

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.generation_origin == "deterministic_repair"
    assert "removed_markdown_fence" in result.repair_actions


def test_skill_generator_accepts_short_text_wrapped_json(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    content = "Here is the candidate:\n" + json.dumps(_skill_args(pack)) + "\nDone."
    provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content=content))])

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.generation_origin == "deterministic_repair"
    assert "extracted_json_object_from_text" in result.repair_actions


def test_skill_generator_retries_after_schema_error(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    bad_args = {"title": "missing required fields"}
    good_args = _skill_args(pack)
    provider = FakeModelProvider(
        [
            ModelResponse(tool_calls=[ToolCall(id="bad", name="submit_skill_candidate", arguments=bad_args)]),
            ModelResponse(tool_calls=[ToolCall(id="good", name="submit_skill_candidate", arguments=good_args)]),
        ]
    )

    result = SkillCandidateGenerator(provider, KnowledgeConfig().skill).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is True
    assert result.attempt_count == 2
    assert provider.calls == 2


def test_skill_generator_retries_after_portability_validation_error(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.PORTABLE
    bad_args = _skill_args(pack, memory.memory_id)
    bad_args["scope"] = "repository"
    bad_args["procedure"][0]["description"] = "Patch validate_task_name() in task_app.py"
    good_args = _skill_args(pack, memory.memory_id)
    provider = FakeModelProvider(
        [
            ModelResponse(tool_calls=[ToolCall(id="bad", name="submit_skill_candidate", arguments=bad_args)]),
            ModelResponse(tool_calls=[ToolCall(id="good", name="submit_skill_candidate", arguments=good_args)]),
        ]
    )

    config = KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True})
    result = SkillCandidateGenerator(provider, config.skill).generate_result(
        pack, source_memory_ids=[memory.memory_id], source_memories=[memory]
    )

    assert result.success is True
    assert result.attempt_count == 2
    assert result.draft is not None
    assert result.draft.scope == MemoryScope.PORTABLE


def test_skill_generator_reports_validation_errors_when_retries_exhaust(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    provider = FakeModelProvider([ModelResponse(tool_calls=[ToolCall(id="bad", name="submit_skill_candidate", arguments={})])])
    cfg = type("SkillCfg", (), {"max_protocol_retries": 0})()

    result = SkillCandidateGenerator(provider, cfg).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is False
    assert result.failure_stage == "protocol_exhausted"
    assert result.validation_errors
    assert "title" in " ".join(result.validation_errors)


def test_skill_generator_distinguishes_provider_error_from_parsing_error(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    cfg = type("SkillCfg", (), {"max_protocol_retries": 0})()
    provider_result = SkillCandidateGenerator(FakeModelProvider([]), cfg).generate_result(pack, source_memory_ids=["MEM-source"])
    parsing_provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content="no json here"))])
    parsing_result = SkillCandidateGenerator(parsing_provider, cfg).generate_result(pack, source_memory_ids=["MEM-source"])

    assert provider_result.failure_stage == "provider_request"
    assert provider_result.exception_type == ProviderError.__name__
    assert parsing_result.failure_stage == "json_extract"
    assert parsing_result.exception_type is None


def test_skill_generator_deterministic_repair_does_not_invent_missing_fields(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    args = _skill_args(pack)
    args["scope"] = "PORTABLE"
    del args["procedure"]
    provider = FakeModelProvider([ModelResponse(final_answer=FinalAnswer(content="```json\n" + json.dumps(args) + "\n```"))])
    cfg = type("SkillCfg", (), {"max_protocol_retries": 0})()

    result = SkillCandidateGenerator(provider, cfg).generate_result(pack, source_memory_ids=["MEM-source"])

    assert result.success is False
    assert result.failure_stage == "protocol_exhausted"
    assert "removed_markdown_fence" in result.repair_actions
    assert any("procedure" in item for item in result.validation_errors)


def test_skill_candidate_requires_active_helpful_source_when_configured(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    pack = _pack(tmp_path).model_copy(update={"files_changed": []})
    config = KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True})
    manager = SkillLifecycleManager(config, store)

    assert manager.candidate_from_success(pack, source_memory_ids=[]) is None
    events = store.read_jsonl(store.events_path)
    assert any(event.get("event_type") == "skill_candidate_skipped" for event in events)

    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.status = MemoryStatus.ACTIVE
    memory.anti_conditions = ["preserve leading whitespace"]
    store.save_memory(memory)
    draft = SkillCandidateDraft.model_validate(_skill_args(pack, memory.memory_id))
    skill = manager.candidate_from_success(pack, draft=draft, source_memory_ids=[memory.memory_id])
    assert skill is not None
    assert skill.scope == MemoryScope.PORTABLE
    assert skill.source_memory_ids == [memory.memory_id]
    assert skill.procedure[-1].command == "pytest -q"


def test_repository_memory_plus_source_specific_draft_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.status = MemoryStatus.ACTIVE
    memory.anti_conditions = ["preserve leading whitespace"]
    store.save_memory(memory)
    pack = _pack(tmp_path).model_copy(update={"files_changed": []})
    draft = SkillCandidateDraft(
        title="Repository-only repair",
        summary="Use task_app.py to repair the verified validation workflow.",
        scope=MemoryScope.PORTABLE,
        preconditions=["pytest validation task"],
        anti_conditions=["repository layout matches exactly"],
        procedure=[SkillStep(description="Run focused tests", tool_name="bash", command="pytest -q")],
        verification=["pytest -q"],
        evidence_ids=[pack.evidence_items[0].evidence_id],
        source_memory_ids=[memory.memory_id],
        confidence=0.9,
    )
    config = KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True})

    skill = SkillLifecycleManager(config, store).candidate_from_success(pack, draft=draft, source_memory_ids=[memory.memory_id])

    assert skill is None


def test_repository_helpful_memory_can_seed_portable_skill(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config = AppConfig(
        model=ModelConfig(provider="fake"),
        workspace={"root": workspace},
        state={"root": tmp_path / "state"},
        knowledge={
            "mode": "memory_skill",
            "root": tmp_path / "knowledge",
            "skill": {"require_active_helpful_source_for_candidate": True},
        },
    )
    skill_args = _skill_args(_pack(tmp_path))
    skill_args["source_memory_ids"] = []
    skill_args["evidence_ids"] = []
    orchestrator = ProjectOrchestrator(
        config,
        FakeModelProvider([ModelResponse(tool_calls=[ToolCall(id="s1", name="submit_skill_candidate", arguments=skill_args)])]),
        project_id="p1",
    )
    assert orchestrator.knowledge_store is not None
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.status = MemoryStatus.ACTIVE
    orchestrator.knowledge_store.save_memory(memory)
    task = TaskNode(
        id="t1",
        key="t1",
        title="Fix validation",
        objective="fix pytest service validation",
        acceptance_criteria=["pytest passes"],
        status=TaskStatus.CANDIDATE_COMPLETE,
    )
    state = ProjectState(project_id="p1", objective="project", tasks=[task])
    session_record = _knowledge_session_record(workspace, changed_files=[])
    channel = KnowledgeUseChannel(exposed_memory_ids=[memory.memory_id], exposed_skill_ids=[])
    channel.record(ReportKnowledgeUseArgs(memory_ids=[memory.memory_id], reason="memory guided the fix"))

    orchestrator._process_knowledge_after_session(
        state,
        task,
        session_record,
        starting_task_status="in_progress",
        knowledge_channel=channel,
        knowledge_bundle=RetrievedKnowledgeBundle(memories=[memory]),
    )

    skills = orchestrator.knowledge_store.list_skills()
    assert len(skills) == 1
    assert skills[0].scope == MemoryScope.PORTABLE
    assert skills[0].source_memory_ids == [memory.memory_id]
    assert session_record["memories_helpful"] == 1
    assert session_record["skills_created"] == 1
    events = orchestrator.knowledge_store.read_jsonl(orchestrator.knowledge_store.events_path)
    evaluated = [event for event in events if event.get("event_type") == "skill_candidate_evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0]["source_memory_scopes"] == {memory.memory_id: "repository"}
    assert evaluated[0]["candidate_created"] is True
    assert evaluated[0]["final_skill_scope"] == "portable"


def test_portable_skill_validator_accepts_active_repository_memory_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.status = MemoryStatus.ACTIVE
    store.save_memory(memory)
    skill = _skill()
    skill.scope = MemoryScope.PORTABLE
    skill.source_memory_ids = [memory.memory_id]
    skill.confidence = 0.9
    pack = _pack(tmp_path)
    skill.evidence_ids = [pack.evidence_items[0].evidence_id]

    valid, reasons = SkillValidator(store, KnowledgeConfig(mode="memory_skill")).validate(skill, pack)

    assert valid, reasons
    assert reasons == []


def test_inactive_memory_cannot_seed_portable_skill(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(status=MemoryStatus.CANDIDATE, project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    store.save_memory(memory)
    pack = _pack(tmp_path).model_copy(update={"files_changed": []})
    draft = SkillCandidateDraft.model_validate(_skill_args(pack, memory.memory_id))
    manager = SkillLifecycleManager(
        KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True}), store
    )

    skill = manager.candidate_from_success(pack, draft=draft, source_memory_ids=[memory.memory_id])

    assert skill is None


def test_repository_memory_plus_parameterized_draft_is_portable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.REPOSITORY
    memory.status = MemoryStatus.ACTIVE
    store.save_memory(memory)
    pack = _pack(tmp_path).model_copy(update={"files_changed": []})
    draft = SkillCandidateDraft.model_validate(_skill_args(pack, memory.memory_id))
    manager = SkillLifecycleManager(
        KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True}), store
    )

    skill = manager.candidate_from_success(pack, draft=draft, source_memory_ids=[memory.memory_id])

    assert skill is not None
    assert skill.scope == MemoryScope.PORTABLE
    assert skill.source_memory_ids == [memory.memory_id]


def test_orchestrator_does_not_create_skill_from_neutral_memory_reference(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config = AppConfig(
        model=ModelConfig(provider="fake"),
        workspace={"root": workspace},
        state={"root": tmp_path / "state"},
        knowledge={
            "mode": "memory_skill",
            "root": tmp_path / "knowledge",
            "skill": {"require_active_helpful_source_for_candidate": True},
        },
    )
    orchestrator = ProjectOrchestrator(config, FakeModelProvider([]), project_id="p1")
    assert orchestrator.knowledge_store is not None
    memory = _memory(project_id=None)
    memory.scope = MemoryScope.PORTABLE
    memory.status = MemoryStatus.ACTIVE
    orchestrator.knowledge_store.save_memory(memory)
    task = TaskNode(
        id="t1",
        key="t1",
        title="Still working",
        objective="fix pytest service validation",
        acceptance_criteria=["pytest passes"],
        status=TaskStatus.IN_PROGRESS,
    )
    state = ProjectState(project_id="p1", objective="project", tasks=[task], active_task_id="t1")
    session_record = _knowledge_session_record(workspace, changed_files=[])
    channel = KnowledgeUseChannel(exposed_memory_ids=[memory.memory_id], exposed_skill_ids=[])
    channel.record(ReportKnowledgeUseArgs(memory_ids=[memory.memory_id], reason="memory was inspected"))

    orchestrator._process_knowledge_after_session(
        state,
        task,
        session_record,
        starting_task_status="in_progress",
        knowledge_channel=channel,
        knowledge_bundle=RetrievedKnowledgeBundle(memories=[memory]),
    )

    assert orchestrator.knowledge_store.list_skills() == []
    assert session_record["memories_helpful"] == 0
    assert session_record["skills_created"] == 0
    events = orchestrator.knowledge_store.read_jsonl(orchestrator.knowledge_store.events_path)
    evaluated = [event for event in events if event.get("event_type") == "skill_candidate_evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0]["verification_passed"] is False
    assert evaluated[0]["rejection_reasons"] == ["verification_failed", "referenced_memory_not_helpful"]


def test_knowledge_use_records_context_and_is_idempotent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = _store(tmp_path, workspace)
    memory = _memory(project_id=None)
    store.save_memory(memory)

    first = store.add_memory_usage(
        memory.memory_id,
        KnowledgeUseType.REFERENCED,
        project_id="p1",
        task_id="t1",
        session_id="s1",
        retrieval_id="r1",
        reason="model referenced injected memory",
    )
    second = store.add_memory_usage(
        memory.memory_id,
        KnowledgeUseType.REFERENCED,
        project_id="p1",
        task_id="t1",
        session_id="s1",
        retrieval_id="r1",
        reason="model referenced injected memory",
    )

    assert first.created is True
    assert second.created is False
    assert first.record.project_id == "p1"
    assert first.record.task_id == "t1"
    assert first.record.session_id == "s1"
    assert first.record.retrieval_id == "r1"
    assert store.load_memory(memory.memory_id).reference_count == 1
    assert len(store.read_jsonl(store.uses_path)) == 1


def test_evidence_pack_failure_secret_and_context_reset_signals(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = {
        "session_id": "s2",
        "run_id": "s2",
        "run_status": "max_steps_reached",
        "read_files": [str(workspace / "service.py"), r"C:\outside\secrets.py"],
        "changed_files": [],
        "bash_commands": ["python -m pytest -q", "echo API_TOKEN=abc"],
        "bash_exit_codes": [1, 0],
        "successful_test_commands": [],
        "repeated_tool_calls": ["bash:pytest"],
        "context_reset_count": 1,
        "no_progress": True,
    }
    pack = build_experience_pack(
        project_id="p2",
        task_id="t2",
        task_objective="fix tests",
        acceptance_criteria=[],
        session_record=session,
        plan_version=2,
        starting_task_status="in_progress",
        ending_task_status="in_progress",
        workspace_root=workspace,
    )
    assert pack.failed_verifications == ["python -m pytest -q"]
    assert "[redacted secret-bearing text]" in pack.commands
    assert "secrets.py" in pack.files_read
    assert pack.context_reset_count == 1
    assert any(item.event_type == "context_reset_observed" for item in pack.evidence_items)
