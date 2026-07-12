from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import load_config
from longrun_agent.model.fake import FakeModelProvider, default_calculator_script
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall

app = typer.Typer(add_completion=False)
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

    provider = (
        FakeModelProvider(_load_scripted_responses(scripted_responses))
        if app_config.model.provider == "fake"
        else OpenAICompatibleProvider(app_config.model)
    )
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


if __name__ == "__main__":
    app()
