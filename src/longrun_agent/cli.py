from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import load_config
from longrun_agent.exceptions import ConfigurationError
from longrun_agent.knowledge.evidence import RepositoryProfiler
from longrun_agent.knowledge.renderer import render_bundle
from longrun_agent.knowledge.retrieval import retrieve_bundle
from longrun_agent.knowledge.schema import KnowledgeRetrievalQuery, MemoryStatus, SkillStatus
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.fake import FakeModelProvider, default_calculator_script
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall
from longrun_agent.state.aggregation import project_statistics
from longrun_agent.state.schema import ProjectState, TaskNode
from longrun_agent.state.store import ProjectStateStore

app = typer.Typer(add_completion=False)
project_app = typer.Typer(add_completion=False)
context_app = typer.Typer(add_completion=False)
verify_app = typer.Typer(add_completion=False)
verify_contract_app = typer.Typer(add_completion=False)
knowledge_app = typer.Typer(add_completion=False)
knowledge_memories_app = typer.Typer(add_completion=False)
knowledge_skills_app = typer.Typer(add_completion=False)
knowledge_retrieval_app = typer.Typer(add_completion=False)
sandbox_app = typer.Typer(add_completion=False)
app.add_typer(project_app, name="project")
app.add_typer(context_app, name="context")
app.add_typer(verify_app, name="verify")
verify_app.add_typer(verify_contract_app, name="contract")
app.add_typer(knowledge_app, name="knowledge")
knowledge_app.add_typer(knowledge_memories_app, name="memories")
knowledge_app.add_typer(knowledge_skills_app, name="skills")
knowledge_app.add_typer(knowledge_retrieval_app, name="retrieval")
app.add_typer(sandbox_app, name="sandbox")
console = Console()


def _load_scripted_responses(path: Path | None) -> list[ModelResponse]:
    if path is None:
        return default_calculator_script()
    raw = json.loads(path.read_text(encoding="utf-8"))
    responses: list[ModelResponse] = []
    for item in raw:
        if item.get("final_answer") is not None:
            responses.append(ModelResponse(final_answer=FinalAnswer(content=item["final_answer"])))
        else:
            calls = [ToolCall.model_validate(call) for call in item.get("tool_calls", [])]
            responses.append(ModelResponse(tool_calls=calls))
    return responses


def _provider(app_config, scripted_responses: Path | None = None):
    return (
        FakeModelProvider(_load_scripted_responses(scripted_responses))
        if app_config.model.provider == "fake"
        else OpenAICompatibleProvider(app_config.model)
    )


@app.command()
def run(
    config: Path = typer.Option(Path("configs/baseline.yaml"), exists=True, file_okay=True, dir_okay=False),
    task: str = typer.Option("Fix the implementation bug in calculator.py so that all tests pass."),
    workspace: Path | None = typer.Option(None, file_okay=False, dir_okay=True),
    model: str | None = typer.Option(None),
    max_steps: int | None = typer.Option(None),
    fake_provider: bool = typer.Option(False),
    scripted_responses: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    app_config = load_config(config)
    if workspace is not None:
        app_config.workspace.root = workspace.resolve()
    if model is not None:
        app_config.model.model_name = model
    if max_steps is not None:
        app_config.agent.max_steps = max_steps
    if fake_provider:
        app_config.model.provider = "fake"

    def on_event(event_type: str, payload: dict) -> None:
        if event_type == "run_started":
            console.print(f"run_id: {payload['run_id']}")
            console.print(f"log_dir: {payload['run_dir']}")
        elif event_type == "model_request":
            console.print(f"step {payload['step']}: model request")
        elif event_type == "model_response":
            console.print(f"step {payload['step']}: model {payload['kind']}")
        elif event_type == "tool_started":
            console.print(f"step {payload['step']}: tool {payload['tool']}")
        elif event_type == "tool_finished":
            exit_code = payload.get("metadata", {}).get("exit_code")
            suffix = f", exit_code={exit_code}" if exit_code is not None else ""
            console.print(f"step {payload['step']}: {payload['summary']}{suffix}")
        elif event_type == "run_finished":
            console.print(f"finished: {payload['status']}")

    provider = _provider(app_config, scripted_responses)
    result = AgentLoop(app_config, provider, on_event=on_event).run(app_config.workspace.root, task)
    if result.final_answer:
        console.print(result.final_answer)
    raise typer.Exit(code=0 if result.status == "completed" else 1)


@app.command()
def tools(
    config: Path = typer.Option(Path("configs/baseline.yaml"), exists=True, file_okay=True, dir_okay=False),
) -> None:
    load_config(config)
    for schema in default_router().schemas():
        console.print_json(json.dumps(schema))


@sandbox_app.command("doctor")
def sandbox_doctor(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    from longrun_agent.tools.sandbox import (
        EvaluationSandboxRuntimeUnavailable,
        EvaluationSandboxUnavailable,
        build_subprocess_sandbox,
    )
    from longrun_agent.tools.workspace_policy import WorkspaceAccessPolicy

    app_config = load_config(config)
    denied_roots = [app_config.state.root, app_config.telemetry.run_root]
    if app_config.verification.contract.path is not None:
        denied_roots.append(app_config.verification.contract.path.parent)
    policy = WorkspaceAccessPolicy.for_workspace(
        app_config.workspace.root,
        evaluation_isolation_enabled=True,
        denied_roots=denied_roots,
    )
    try:
        report = build_subprocess_sandbox(policy).preflight()
    except EvaluationSandboxRuntimeUnavailable:
        console.print_json(json.dumps({"status": "NO_GO", "error_code": "EVALUATION_SANDBOX_RUNTIME_UNAVAILABLE"}))
        raise typer.Exit(code=2) from None
    except EvaluationSandboxUnavailable:
        console.print_json(json.dumps({"status": "NO_GO", "error_code": "EVALUATION_SANDBOX_UNAVAILABLE"}))
        raise typer.Exit(code=2) from None
    console.print_json(json.dumps(report))


def _task_text(task: str | None, task_file: Path | None) -> str:
    if task_file is not None:
        return task_file.read_text(encoding="utf-8")
    if task:
        return task
    raise typer.BadParameter("provide --task or --task-file")


@project_app.command("start")
def project_start(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    task: str | None = typer.Option(None),
    task_file: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    project_id: str | None = typer.Option(None),
    scripted_responses: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    app_config = load_config(config)
    objective = _task_text(task, task_file)
    outcome = ProjectOrchestrator(app_config, _provider(app_config, scripted_responses), project_id=project_id).start(objective)
    console.print(f"project_id: {outcome.project_id}")
    console.print(f"status: {outcome.status}")
    console.print(f"state_path: {outcome.state_path}")
    raise typer.Exit(code=0 if outcome.status in {"candidate_complete", "verified"} else 1)


@project_app.command("resume")
def project_resume(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
    scripted_responses: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    app_config = _load_project_resume_config(config, project_id)
    outcome = ProjectOrchestrator(app_config, _provider(app_config, scripted_responses)).resume(project_id)
    console.print(f"project_id: {outcome.project_id}")
    console.print(f"status: {outcome.status}")
    console.print(f"state_path: {outcome.state_path}")
    raise typer.Exit(code=0 if outcome.status in {"candidate_complete", "verified"} else 1)


def _load_project_resume_config(config_path: Path, project_id: str):
    app_config = load_config(
        config_path,
        allow_missing_env={"LONGRUN_WORKSPACE", "LONGRUN_PLAN_FILE"},
    )
    bootstrap_store = ProjectStateStore(app_config.state.root)
    state = bootstrap_store.load(project_id)
    runtime_paths_migrated = state.workspace_root is None
    workspace = Path(state.workspace_root).resolve() if state.workspace_root else None
    plan_file = Path(state.initial_plan_file).resolve() if state.initial_plan_file else None

    if workspace is None or plan_file is None:
        legacy_workspace, legacy_plan_file = _legacy_project_runtime_paths(app_config, state)
        workspace = workspace or legacy_workspace
        plan_file = plan_file or legacy_plan_file

    if workspace is None or not workspace.is_dir():
        raise ConfigurationError(
            f"cannot recover the original workspace for project {project_id}; set LONGRUN_WORKSPACE to the existing project workspace"
        )

    app_config.workspace.root = workspace
    state.workspace_root = str(workspace)
    if plan_file is not None and plan_file.is_file():
        app_config.planning.initial_plan.plan_file = plan_file
        state.initial_plan_file = str(plan_file)
    if runtime_paths_migrated:
        correction = (
            f"Runtime path recovery: resume restored the original workspace at {workspace}. "
            "Ignore file-not-found or repository-layout observations from Sessions that used a different workspace. "
            "Continue from the files and tests in this restored workspace."
        )
        for task in state.tasks:
            if task.status.value in {"ready", "in_progress", "failed", "reopened"}:
                task.progress_notes.append(correction)
                task.last_handoff_summary = correction
                task.consecutive_no_progress_sessions = 0
    bootstrap_store.save(state)
    return app_config


def _legacy_project_runtime_paths(app_config, state: ProjectState) -> tuple[Path | None, Path | None]:
    session_ids = [session_id for task in state.tasks for session_id in task.session_ids]
    if not session_ids:
        session_ids = [f"{state.project_id}-s1"]
    for session_id in session_ids:
        events_path = app_config.telemetry.run_root / session_id / "events.jsonl"
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "run_started":
                continue
            payload = event.get("payload") or {}
            sanitized_config = payload.get("config") or {}
            raw_workspace = payload.get("workspace") or (sanitized_config.get("workspace") or {}).get("root")
            planning = sanitized_config.get("planning") or {}
            raw_plan_file = (planning.get("initial_plan") or {}).get("plan_file")
            workspace = Path(raw_workspace).resolve() if isinstance(raw_workspace, str) and raw_workspace else None
            plan_file = Path(raw_plan_file).resolve() if isinstance(raw_plan_file, str) and raw_plan_file else None
            return workspace, plan_file
    return None, None


def _store(config_path: Path) -> tuple[ProjectStateStore, object]:
    app_config = load_config(config_path)
    return ProjectStateStore(app_config.state.root, workspace_root=app_config.workspace.root), app_config


def _knowledge_store(config_path: Path) -> tuple[KnowledgeStore, object]:
    app_config = load_config(config_path)
    return (
        KnowledgeStore(
            app_config.knowledge.root,
            workspace_root=app_config.workspace.root,
            atomic_write=app_config.state.atomic_write,
            record_mutation_policy=app_config.knowledge.record_mutation_policy,
        ),
        app_config,
    )


def _memory_status_filter(status: str | None) -> dict[str, object] | None:
    if status is None:
        return None
    try:
        parsed_status = MemoryStatus(status)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MemoryStatus)
        raise typer.BadParameter(f"invalid memory status '{status}'; expected one of: {allowed}") from exc
    return {"status": parsed_status.value}


def _skill_status_filter(status: str | None) -> dict[str, object] | None:
    if status is None:
        return None
    try:
        parsed_status = SkillStatus(status)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SkillStatus)
        raise typer.BadParameter(f"invalid skill status '{status}'; expected one of: {allowed}") from exc
    return {"status": parsed_status.value}


@project_app.command("status")
def project_status(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    store, app_config = _store(config)
    state = store.load(project_id)
    console.print(f"Project objective: {state.objective}")
    console.print(f"Project status: {state.status.value}")
    console.print(f"plan version: {state.plan_version}")
    console.print(f"active task: {state.active_task_id}")
    for task in state.tasks:
        console.print(
            f"- {task.id} depth={task.depth} status={task.status.value} attempts={task.attempts} deps={task.dependencies} blocker={task.blocker}"
        )
    console.print(f"session count: {state.session_count}")
    console.print(f"Project verification status: {state.status.value}")
    console.print(f"latest verification report: {state.latest_project_verification_report_id}")
    console.print(f"verified tasks: {sum(task.status.value == 'verified' for task in state.tasks)}")
    console.print(f"reopened tasks: {sum(task.reopen_count > 0 for task in state.tasks)}")
    if state.latest_project_verification_report_id and app_config.verification.store_root:
        from longrun_agent.verification.store import VerificationStore

        verification_store = VerificationStore(
            app_config.verification.store_root,
            project_id,
            workspace_root=app_config.workspace.root,
        )
        report = verification_store.load_report(state.latest_project_verification_report_id)
        console.print(f"F2P: {report.summary.resolution_passed}/{report.summary.resolution_total}")
        console.print(f"P2P: {report.summary.regression_passed}/{report.summary.regression_total}")
        console.print(f"Integrity passed: {report.summary.integrity_passed}")
    console.print(f"state file: {store.state_path(project_id)}")


def _print_task_tree(state: ProjectState, task: TaskNode, indent: int = 0) -> None:
    console.print(f"{'  ' * indent}- {task.key} [{task.status.value}] {task.title}")
    for child in [candidate for candidate in state.tasks if candidate.parent_id == task.id]:
        _print_task_tree(state, child, indent + 1)


@project_app.command("tree")
def project_tree(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    store, app_config = _store(config)
    state = store.load(project_id)
    console.print(f"Project {state.project_id}: {state.status.value}")
    for task in [candidate for candidate in state.tasks if candidate.parent_id is None]:
        _print_task_tree(state, task)


@project_app.command("metrics")
def project_metrics(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    store, _app_config = _store(config)
    path = store.metrics_path(project_id)
    if not path.exists():
        state = store.load(project_id)
        store.write_metrics(project_id, project_statistics(state, store.read_sessions(project_id)))
    console.print_json(path.read_text(encoding="utf-8"))


@knowledge_memories_app.command("list")
def knowledge_memories_list(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    status: str | None = typer.Option(None, help="Filter by memory status."),
) -> None:
    store, _app_config = _knowledge_store(config)
    records = store.list_memories(_memory_status_filter(status))
    payload = [
        {
            "memory_id": record.memory_id,
            "status": record.status.value,
            "scope": record.scope.value,
            "kind": record.kind.value,
            "title": record.title,
            "confidence": record.confidence,
            "source_episode_ids": record.source_episode_ids,
        }
        for record in records
    ]
    console.print_json(json.dumps(payload))


@knowledge_memories_app.command("show")
def knowledge_memories_show(
    memory_id: str,
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    store, _app_config = _knowledge_store(config)
    try:
        record = store.load_memory(memory_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"memory not found: {memory_id}", param_hint="memory_id") from exc
    console.print_json(record.model_dump_json())


@knowledge_memories_app.command("invalidate")
def knowledge_memories_invalidate(
    memory_id: str,
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    reason: str = typer.Option("manually updated through CLI", help="Reason for the status change."),
) -> None:
    store, _app_config = _knowledge_store(config)
    try:
        record = store.update_memory_status(memory_id, MemoryStatus.INVALIDATED, reason=reason)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"memory not found: {memory_id}", param_hint="memory_id") from exc
    console.print_json(record.model_dump_json())


@knowledge_skills_app.command("list")
def knowledge_skills_list(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    status: str | None = typer.Option(None, help="Filter by skill status."),
) -> None:
    store, _app_config = _knowledge_store(config)
    records = store.list_skills(_skill_status_filter(status))
    payload = [
        {
            "skill_id": record.skill_id,
            "status": record.status.value,
            "title": record.title,
            "version": record.version,
            "success_count": record.success_count,
            "failure_count": record.failure_count,
            "source_episode_ids": record.source_episode_ids,
            "source_task_ids": record.source_task_ids,
        }
        for record in records
    ]
    console.print_json(json.dumps(payload))


@knowledge_skills_app.command("show")
def knowledge_skills_show(
    skill_id: str,
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    store, _app_config = _knowledge_store(config)
    try:
        record = store.load_skill(skill_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"skill not found: {skill_id}", param_hint="skill_id") from exc
    console.print_json(record.model_dump_json())


@knowledge_skills_app.command("deprecate")
def knowledge_skills_deprecate(
    skill_id: str,
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    reason: str = typer.Option("manually updated through CLI", help="Reason for the status change."),
) -> None:
    store, _app_config = _knowledge_store(config)
    try:
        record = store.update_skill_status(skill_id, SkillStatus.DEPRECATED, reason=reason)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"skill not found: {skill_id}", param_hint="skill_id") from exc
    console.print_json(record.model_dump_json())


@knowledge_retrieval_app.command("explain")
def knowledge_retrieval_explain(
    task: str = typer.Option(..., help="Task objective to explain retrieval for."),
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str | None = typer.Option(None),
) -> None:
    if not task.strip():
        raise typer.BadParameter("task must not be empty")
    store, app_config = _knowledge_store(config)
    profile = RepositoryProfiler(app_config.workspace.root).profile()
    query = KnowledgeRetrievalQuery(
        task_objective=task,
        acceptance_criteria=[],
        repository_fingerprint=profile.repository_fingerprint,
        language_tags=profile.language_tags,
        framework_tags=profile.framework_tags,
        tool_tags=profile.tool_tags,
        project_id=project_id,
    )
    bundle, scores = retrieve_bundle(app_config.knowledge, store, query)
    rendered, estimated_tokens = render_bundle(bundle, app_config.knowledge)
    payload = {
        "retrieval_id": bundle.retrieval_id,
        "estimated_tokens": estimated_tokens,
        "memory_ids": [record.memory_id for record in bundle.memories],
        "skill_ids": [record.skill_id for record in bundle.skills],
        "scores": [score.model_dump(mode="json") for score in scores],
        "rendered_context": rendered,
    }
    console.print_json(json.dumps(payload))


@context_app.command("inspect")
def context_inspect(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
    session_id: str | None = typer.Option(None),
) -> None:
    store, app_config = _store(config)
    sessions = store.read_sessions(project_id)
    if session_id:
        sessions = [session for session in sessions if session.get("session_id") == session_id]
    segments = store.read_context_segments(project_id)
    events = store.read_context_events(project_id)
    latest = sessions[-1] if sessions else {}
    payload = {
        "context_mode": app_config.context.mode,
        "segment_count": len(segments) or int(latest.get("context_segment_count") or 0),
        "current_segment": latest.get("context_segment_count"),
        "estimated_token_usage": latest.get("max_estimated_input_tokens"),
        "actual_max_input_token": latest.get("max_actual_input_tokens"),
        "reset_count": latest.get("context_reset_count", 0),
        "prune_count": latest.get("deterministic_prune_count", 0),
        "latest_handoff": latest.get("latest_context_handoff_id"),
        "stale_item_count": latest.get("stale_item_count", 0),
        "artifact_paths": {
            "segments": str(store.context_segments_path(project_id)),
            "events": str(store.context_events_path(project_id)),
            "handoffs": str(store.handoffs_dir(project_id)),
        },
        "context_event_count": len(events),
    }
    console.print_json(json.dumps(payload))


@context_app.command("handoff")
def context_handoff(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
    handoff_id: str = typer.Option(...),
) -> None:
    store, _app_config = _store(config)
    record = store.load_handoff(project_id, handoff_id)
    console.print_json(record.model_dump_json())


@verify_contract_app.command("validate")
def verify_contract_validate(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    from longrun_agent.verification.contract import load_contract

    app_config = load_config(config)
    if app_config.verification.contract.path is None:
        raise typer.BadParameter("verification.contract.path is required")
    contract = load_contract(app_config.verification.contract.path, workspace_root=app_config.workspace.root).freeze()
    console.print_json(
        json.dumps(
            {
                "valid": True,
                "contract_id": contract.contract_id,
                "contract_hash": contract.contract_hash,
                "scope": contract.scope,
                "public_checks": sum(check.visibility.value == "public" for check in contract.checks),
                "hidden_checks": sum(check.visibility.value == "hidden" for check in contract.checks),
            }
        )
    )


@verify_contract_app.command("show")
def verify_contract_show(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    contract_id: str = typer.Option(...),
    include_hidden: bool = typer.Option(False, help="Show hidden details. Intended only for harness operators."),
) -> None:
    from longrun_agent.verification.renderer import render_contract
    from longrun_agent.verification.store import VerificationStore

    app_config = load_config(config)
    if app_config.verification.store_root is None:
        raise typer.BadParameter("verification.store_root is required")
    project_ids = ProjectStateStore(app_config.state.root, workspace_root=app_config.workspace.root).list_projects()
    matching = []
    for project_id in project_ids:
        store = VerificationStore(app_config.verification.store_root, project_id, workspace_root=app_config.workspace.root)
        try:
            matching.append(store.load_contract(contract_id))
        except FileNotFoundError:
            continue
    if not matching:
        raise typer.BadParameter(f"contract not found: {contract_id}")
    console.print_json(json.dumps(render_contract(matching[-1], include_hidden=include_hidden)))


@verify_app.command("run")
def verify_run(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    app_config = load_config(config)
    outcome = ProjectOrchestrator(app_config, _provider(app_config)).resume(project_id)
    console.print_json(
        json.dumps(
            {
                "project_id": project_id,
                "status": outcome.status,
                "verification_verdict": outcome.verification_verdict,
                "report_id": outcome.verification_report_id,
            }
        )
    )
    raise typer.Exit(code=0 if outcome.status == "verified" else 1)


@verify_app.command("report")
def verify_report(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
    report_id: str = typer.Option(...),
) -> None:
    from longrun_agent.verification.renderer import render_agent_feedback
    from longrun_agent.verification.store import VerificationStore

    app_config = load_config(config)
    assert app_config.verification.store_root is not None
    report = VerificationStore(
        app_config.verification.store_root,
        project_id,
        workspace_root=app_config.workspace.root,
    ).load_report(report_id)
    payload = {
        "report_id": report.report_id,
        "verdict": report.verdict.value,
        "summary": report.summary.model_dump(mode="json"),
        "feedback": render_agent_feedback(report),
    }
    console.print_json(json.dumps(payload))


if __name__ == "__main__":
    app()
