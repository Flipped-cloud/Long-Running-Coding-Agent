from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from longrun_agent.config import AppConfig, ModelConfig, load_config
from longrun_agent.evals.experience_learning.collector import collect_case_result
from longrun_agent.evals.experience_learning.fake_provider import ProtocolAwareExperienceFakeProvider
from longrun_agent.evals.experience_learning.schemas import (
    ExperienceLearningCase,
    ExperienceLearningCaseResult,
    VerificationConfig,
    VerificationResult,
)
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator


def run_case(
    *,
    case: ExperienceLearningCase,
    mode: str,
    repeat: int,
    repeat_root: Path,
    verification: VerificationConfig,
    backend: str,
    fail_fast_on_knowledge_error: bool,
    model: ModelConfig | None = None,
    record_mutation_policy: str = "read_write",
    snapshot_skill_ids: list[str] | None = None,
) -> ExperienceLearningCaseResult:
    case_root = repeat_root / "cases" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    workspace = _prepare_case_workspace(case.repository, repeat_root / "workspaces" / case.case_id)
    working_case = case.model_copy(update={"repository": workspace})
    started = time.monotonic()
    reset = subprocess.run([_python(), str(working_case.reset_script)], cwd=workspace, check=False, capture_output=True, text=True)
    initial = _verify(workspace, verification)
    if initial.passed != case.initial_verification_should_pass:
        raise AssertionError(f"{case.case_id} initial verification expected {case.initial_verification_should_pass}, got {initial.passed}")
    working_case = working_case.model_copy(update={"task": _task_with_preflight(working_case.task, initial, workspace)})
    before_memory_ids, before_skill_ids = _knowledge_ids(repeat_root / "knowledge", workspace, mode)
    knowledge_store = KnowledgeStore(
        repeat_root / "knowledge",
        workspace_root=workspace,
        record_mutation_policy=record_mutation_policy,
    )
    records_hash_before = knowledge_store.records_hash()
    violations_before = sum(
        event.get("event_type") == "knowledge_mutation_violation" for event in knowledge_store.read_jsonl(knowledge_store.events_path)
    )
    config = _app_config(
        case=working_case,
        mode=mode,
        repeat_root=repeat_root,
        verification=verification,
        fail_fast=fail_fast_on_knowledge_error,
        backend=backend,
        model=model,
        record_mutation_policy=record_mutation_policy,
    )
    provider = ProtocolAwareExperienceFakeProvider(working_case) if backend == "fake" else OpenAICompatibleProvider(config.model)
    project_id = f"{mode}-r{repeat}-{case.case_id}"
    outcome = ProjectOrchestrator(config, provider, project_id=project_id).start(working_case.task)
    final = _verify(workspace, verification)
    if final.passed != case.final_verification_should_pass:
        raise AssertionError(f"{case.case_id} final verification expected {case.final_verification_should_pass}, got {final.passed}")
    result = collect_case_result(
        mode=mode,
        repeat=repeat,
        case=working_case,
        project_id=project_id,
        project_status=outcome.status,
        reset_exit_code=reset.returncode,
        initial_verification=initial,
        final_verification=final,
        state_root=repeat_root / "projects",
        knowledge_root=repeat_root / "knowledge",
        workspace_root=workspace,
        wall_time_seconds=time.monotonic() - started,
        before_memory_ids=before_memory_ids,
        before_skill_ids=before_skill_ids,
    )
    records_hash_after = knowledge_store.records_hash()
    violations_after = sum(
        event.get("event_type") == "knowledge_mutation_violation" for event in knowledge_store.read_jsonl(knowledge_store.events_path)
    )
    snapshot_skill_ids = snapshot_skill_ids or []
    result.knowledge_phase = case.knowledge_phase
    result.knowledge_records_hash_before = records_hash_before
    result.knowledge_records_hash_after = records_hash_after
    result.snapshot_skill_ids = list(snapshot_skill_ids)
    result.mutation_violation_count = violations_after - violations_before
    result.preexisting_skill_reference_count = len(set(result.referenced_skill_ids) & set(snapshot_skill_ids))
    result.preexisting_skill_helpful_count = len(set(result.helpful_skill_ids) & set(snapshot_skill_ids))
    result.knowledge_assisted_transfer_success = bool(
        case.knowledge_phase == "frozen_transfer"
        and result.final_verification.passed
        and result.preexisting_skill_reference_count
        and result.preexisting_skill_helpful_count
        and result.cross_repository_skill_reference_count
        and result.cross_repository_skill_helpful_count
        and not result.created_memory_ids
        and not result.created_skill_ids
        and records_hash_before == records_hash_after
    )
    result.derive_structural_readiness()
    (case_root / "case_result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _app_config(
    *,
    case: ExperienceLearningCase,
    mode: str,
    repeat_root: Path,
    verification: VerificationConfig,
    fail_fast: bool,
    backend: str,
    model: ModelConfig | None = None,
    record_mutation_policy: str = "read_write",
) -> AppConfig:
    config = load_config(Path("configs/planning_static.yaml"))
    if backend == "fake":
        config.model.provider = "fake"
        config.model.model_name = "experience-learning-fake"
    else:
        if model is None:
            raise ValueError("API backend requires top-level model configuration")
        config.model = model
    config.agent.max_steps = 10
    config.agent.terminal_grace_turns = 2
    config.workspace.root = case.repository
    config.telemetry.run_root = repeat_root / "telemetry"
    config.state.root = repeat_root / "projects"
    config.state.atomic_write = True
    config.planning.mode = "static"
    config.planning.initial_plan.min_tasks = 1
    config.planning.initial_plan.max_tasks = 1
    config.planning.execution.max_project_sessions = 1
    config.planning.execution.max_sessions_per_task = 1
    config.planning.execution.final_verification_command = verification.command
    config.planning.execution.final_verification_timeout_seconds = verification.timeout_seconds
    config.knowledge.mode = mode
    config.knowledge.root = repeat_root / "knowledge"
    config.knowledge.strict_errors = fail_fast
    config.knowledge.record_mutation_policy = record_mutation_policy
    config.knowledge.skill.require_active_helpful_source_for_candidate = True
    config.knowledge.skill.promotion_min_successes = 2
    config.knowledge.skill.promotion_min_distinct_tasks = 2
    config.knowledge.skill.promotion_min_distinct_repositories = 1
    return config


def _verify(repository: Path, verification: VerificationConfig) -> VerificationResult:
    result = subprocess.run(
        verification.command, cwd=repository, check=False, capture_output=True, text=True, timeout=verification.timeout_seconds
    )
    return VerificationResult(
        command=verification.command,
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
        passed=result.returncode == 0,
    )


def _task_with_preflight(task: str, initial: VerificationResult, workspace: Path) -> str:
    command = " ".join(initial.command)
    passed = "true" if initial.passed else "false"
    return "\n\n".join(
        [
            task,
            "Preflight verification evidence observed immediately before this project run:",
            "\n".join(
                [
                    f"- command: {command}",
                    f"- exit_code: {initial.exit_code}",
                    f"- passed: {passed}",
                    "- stdout:",
                    _sanitize_preflight_stream(initial.stdout, workspace),
                    "- stderr:",
                    _sanitize_preflight_stream(initial.stderr, workspace),
                ]
            ),
            "\n".join(
                [
                    "Execution policy for this evidence:",
                    "- Treat this as observed verification evidence from immediately before the project run.",
                    "- If preflight already passes and the acceptance criteria preserve existing behavior, do not edit without contradictory evidence.",
                    "- Prefer confirming implementation and completion without edits when the passing evidence satisfies the criteria.",
                    "- If preflight fails, diagnose the failure and apply the minimal verified fix.",
                ]
            ),
        ]
    )


def _sanitize_preflight_stream(value: str, workspace: Path) -> str:
    text = (value or "")[-2000:]
    text = text.replace(str(workspace), "<workspace>")
    lowered = text.lower()
    if any(marker in lowered for marker in ("api_key", "apikey", "token", "secret", "password", "authorization", "bearer ")):
        return "[redacted secret-bearing output]"
    return text


def _python() -> str:
    import sys

    return sys.executable


def _prepare_case_workspace(template: Path, workspace: Path) -> Path:
    resolved_workspace = workspace.resolve()
    workspaces_root = workspace.parent.resolve()
    if resolved_workspace.parent != workspaces_root:
        raise ValueError(f"unexpected workspace path: {workspace}")
    if resolved_workspace.exists():
        shutil.rmtree(resolved_workspace)
    shutil.copytree(template, resolved_workspace, ignore=_copy_ignore)
    return resolved_workspace


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".runs"}
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _knowledge_ids(knowledge_root: Path, workspace: Path, mode: str) -> tuple[set[str], set[str]]:
    if mode == "disabled":
        return set(), set()
    store = KnowledgeStore(knowledge_root, workspace_root=workspace)
    return {memory.memory_id for memory in store.list_memories()}, {skill.skill_id for skill in store.list_skills()}
