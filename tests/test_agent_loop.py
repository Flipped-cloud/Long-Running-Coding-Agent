import json
from pathlib import Path

import pytest

from longrun_agent.agent.loop import AgentLoop
from longrun_agent.config import AgentConfig, AppConfig, BashConfig, ModelConfig, TelemetryConfig, ToolsConfig, WorkspaceConfig
from longrun_agent.exceptions import ProviderError
from longrun_agent.model.base import ModelProvider
from longrun_agent.model.fake import FakeModelProvider, default_calculator_script
from longrun_agent.protocol import FinalAnswer, ModelResponse, RunStatus, ToolCall


class RaisingProvider(ModelProvider):
    def generate(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        raise ProviderError("temporary API failure")


def config(workspace: Path, run_root: Path, max_steps: int = 10) -> AppConfig:
    return AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        agent=AgentConfig(max_steps=max_steps, max_consecutive_errors=2),
        workspace=WorkspaceConfig(root=workspace),
        tools=ToolsConfig(bash=BashConfig(timeout_seconds=10, max_output_chars=20000, shell=False)),
        telemetry=TelemetryConfig(run_root=run_root, save_prompts=True, save_full_tool_outputs=True),
    )


def make_calculator_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def divide(a: float, b: float) -> float:\n    return a * b\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        """
import pytest
from calculator import divide

def test_divide_positive_numbers():
    assert divide(8, 2) == 4

def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)
""",
        encoding="utf-8",
    )
    return repo


def test_agent_loop_full_fake_provider_trace_repairs_calculator(tmp_path: Path):
    repo = make_calculator_repo(tmp_path)
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(default_calculator_script()), run_id="run1").run(
        repo,
        "Fix the implementation bug in calculator.py so that all tests pass.",
    )
    assert result.status == RunStatus.COMPLETED
    assert result.tool_call_count == 3
    assert "return a / b" in (repo / "calculator.py").read_text(encoding="utf-8")
    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert {"read_file", "write_file", "bash"}.issubset({event.get("tool_name") for event in events})
    assert "final_answer" in {event["event_type"] for event in events}
    bash_events = [event for event in events if event.get("tool_name") == "bash"]
    assert bash_events[-1]["exit_code"] == 0


def test_agent_loop_multiple_tool_calls_execute_in_order(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [
        ModelResponse(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "1"}),
                ToolCall(id="w2", name="write_file", arguments={"path": "b.txt", "content": "2"}),
            ]
        ),
        ModelResponse(final_answer=FinalAnswer(content="done")),
    ]
    result = AgentLoop(config(repo, tmp_path / ".runs"), FakeModelProvider(responses), run_id="run2").run(repo, "write files")
    assert result.status == RunStatus.COMPLETED
    assert (repo / "a.txt").read_text(encoding="utf-8") == "1"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "2"


def test_agent_loop_stops_at_max_steps(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    responses = [
        ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "missing.txt"})]),
        ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "missing.txt"})]),
    ]
    result = AgentLoop(config(repo, tmp_path / ".runs", max_steps=1), FakeModelProvider(responses), run_id="run3").run(repo, "loop")
    assert result.status == RunStatus.MAX_STEPS_REACHED


def test_agent_loop_provider_exception_status(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = AgentLoop(config(repo, tmp_path / ".runs"), RaisingProvider(), run_id="run4").run(repo, "task")
    assert result.status == RunStatus.PROVIDER_ERROR


def test_fake_provider_response_exhaustion():
    provider = FakeModelProvider([])
    with pytest.raises(ProviderError):
        provider.generate([], [])
