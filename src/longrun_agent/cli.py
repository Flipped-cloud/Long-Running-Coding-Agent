from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import load_config
from longrun_agent.model.fake import FakeModelProvider, default_calculator_script
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall
from longrun_agent.state.aggregation import project_statistics
from longrun_agent.state.schema import ProjectState, TaskNode
from longrun_agent.state.store import ProjectStateStore

app = typer.Typer(add_completion=False)
project_app = typer.Typer(add_completion=False)
app.add_typer(project_app, name="project")
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


def _task_text(task: str | None, task_file: Path | None) -> str:
    if task_file is not None:
        return task_file.read_text(encoding="utf-8")
    if task:
        return task
    raise typer.BadParameter("provide --task or --task-file")


@project_app.command("start")
def project_start(
    config: Path = typer.Option(Path("configs/planning_static.yaml"), exists=True, file_okay=True, dir_okay=False),
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
    raise typer.Exit(code=0 if outcome.status == "candidate_complete" else 1)


@project_app.command("resume")
def project_resume(
    config: Path = typer.Option(Path("configs/planning_static.yaml"), exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
    scripted_responses: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    app_config = load_config(config)
    outcome = ProjectOrchestrator(app_config, _provider(app_config, scripted_responses)).resume(project_id)
    console.print(f"project_id: {outcome.project_id}")
    console.print(f"status: {outcome.status}")
    console.print(f"state_path: {outcome.state_path}")
    raise typer.Exit(code=0 if outcome.status == "candidate_complete" else 1)


def _store(config_path: Path) -> tuple[ProjectStateStore, object]:
    app_config = load_config(config_path)
    return ProjectStateStore(app_config.state.root, workspace_root=app_config.workspace.root), app_config


@project_app.command("status")
def project_status(
    config: Path = typer.Option(Path("configs/planning_static.yaml"), exists=True, file_okay=True, dir_okay=False),
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
    console.print(f"state file: {store.state_path(project_id)}")


def _print_task_tree(state: ProjectState, task: TaskNode, indent: int = 0) -> None:
    console.print(f"{'  ' * indent}- {task.key} [{task.status.value}] {task.title}")
    for child in [candidate for candidate in state.tasks if candidate.parent_id == task.id]:
        _print_task_tree(state, child, indent + 1)


@project_app.command("tree")
def project_tree(
    config: Path = typer.Option(Path("configs/planning_static.yaml"), exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    store, app_config = _store(config)
    state = store.load(project_id)
    console.print(f"Project {state.project_id}: {state.status.value}")
    for task in [candidate for candidate in state.tasks if candidate.parent_id is None]:
        _print_task_tree(state, task)


@project_app.command("metrics")
def project_metrics(
    config: Path = typer.Option(Path("configs/planning_static.yaml"), exists=True, file_okay=True, dir_okay=False),
    project_id: str = typer.Option(...),
) -> None:
    store, _app_config = _store(config)
    path = store.metrics_path(project_id)
    if not path.exists():
        state = store.load(project_id)
        store.write_metrics(project_id, project_statistics(state, store.read_sessions(project_id)))
    console.print_json(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
