from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from longrun_agent.agent.loop import AgentLoop
from longrun_agent.config import AgentConfig, AppConfig, BashConfig, ModelConfig, TelemetryConfig, ToolsConfig, WorkspaceConfig
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.protocol import ErrorType, FinalAnswer, ModelResponse, RunStatus, ToolCall
from longrun_agent.tools.arguments import ToolArgumentError, normalize_command_argv, render_command
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.bash import BashArgs, BashTool
from longrun_agent.tools.router import ToolRouter


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        config=ToolsConfig(bash=BashConfig(timeout_seconds=10, max_output_chars=20000, shell=False)),
    )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        agent=AgentConfig(max_steps=4, max_consecutive_errors=2),
        workspace=WorkspaceConfig(root=tmp_path),
        tools=ToolsConfig(bash=BashConfig(timeout_seconds=10, max_output_chars=20000, shell=False)),
        telemetry=TelemetryConfig(run_root=tmp_path / "telemetry"),
    )


def test_normalize_command_argv_safe_json_scalars() -> None:
    argv, records = normalize_command_argv(["find", ".", "-maxdepth", 3, 1.5, True, False, "-q"])

    assert argv == ["find", ".", "-maxdepth", "3", "1.5", "true", "false", "-q"]
    assert [record.original_type for record in records] == ["int", "float", "bool", "bool"]
    assert all(record.normalized_type == "str" for record in records)
    assert all(record.reason == "json_scalar_to_string" for record in records)
    assert records[0].field == "argv"
    assert records[0].index == 3


@pytest.mark.parametrize(
    "value, received",
    [
        (None, "null"),
        ({"x": 1}, "object"),
        (["nested"], "array"),
        (("nested",), "array"),
        ({"nested"}, "array"),
        (b"secret", "bytes"),
        (object(), "object"),
    ],
)
def test_normalize_command_argv_rejects_non_json_scalars(value, received: str) -> None:
    with pytest.raises(ToolArgumentError) as raised:
        normalize_command_argv(["echo", value])

    message = str(raised.value)
    assert "argv[1]" in message
    assert f"received {received}" in message
    assert "allowed types" in message


@pytest.mark.parametrize("argv", [[], [""], ["   "]])
def test_normalize_command_argv_rejects_empty_command(argv) -> None:
    with pytest.raises(ToolArgumentError):
        normalize_command_argv(argv)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_normalize_command_argv_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ToolArgumentError, match=r"argv\[1\].*non-finite number"):
        normalize_command_argv(["echo", value])


def test_bash_schema_and_renderer_use_normalized_argv() -> None:
    arguments = BashArgs.model_validate({"argv": ["find", ".", "-maxdepth", 3]})

    assert arguments.argv == ["find", ".", "-maxdepth", "3"]
    assert render_command(arguments.argv) == "find . -maxdepth 3"
    with pytest.raises(ToolArgumentError, match=r"argv\[1\]"):
        render_command(["echo", 3])  # type: ignore[list-item]


def test_router_rejects_invalid_argv_without_starting_subprocess(tmp_path: Path, monkeypatch) -> None:
    started = False

    def fail_if_started(*args, **kwargs):
        nonlocal started
        started = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr(subprocess, "Popen", fail_if_started)
    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="bad-call", name="bash", arguments={"argv": ["echo", {"secret": "not-logged"}]}),
        _context(tmp_path),
    )

    assert started is False
    assert result.tool_call_id == "bad-call"
    assert result.success is False
    assert result.error_type == ErrorType.INVALID_TOOL_ARGUMENTS
    assert result.error_type.value == "invalid_tool_arguments"
    assert result.retryable is True
    assert result.metadata["failure_code"] == "TOOL_INVALID_ARGUMENT"
    assert "not-logged" not in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["echo", None],
        ["echo", {"x": 1}],
        ["echo", ["nested"]],
        [],
        [""],
        ["echo", math.nan],
        ["echo", math.inf],
    ],
)
def test_router_maps_unsafe_argv_to_retryable_tool_result(tmp_path: Path, argv) -> None:
    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="invalid", name="bash", arguments={"argv": argv}),
        _context(tmp_path),
    )

    assert result.success is False
    assert result.error_type == ErrorType.INVALID_TOOL_ARGUMENTS
    assert result.retryable is True
    assert result.metadata["failure_code"] == "TOOL_INVALID_ARGUMENT"


def test_router_executes_only_normalized_strings_and_records_conversion(tmp_path: Path, monkeypatch) -> None:
    received_argv: list[str] = []

    class Process:
        returncode = 0
        pid = 1

        def communicate(self, timeout=None):
            return "ok\n", ""

    def capture(argv, **kwargs):
        received_argv.extend(argv)
        return Process()

    monkeypatch.setattr(subprocess, "Popen", capture)
    result = ToolRouter([BashTool()]).execute(
        ToolCall(
            id="real-regression",
            name="bash",
            arguments={"argv": ["find", ".", "-type", "f", "-maxdepth", 3], "cwd": "."},
        ),
        _context(tmp_path),
    )

    assert result.success is True
    assert received_argv == ["find", ".", "-type", "f", "-maxdepth", "3"]
    assert all(isinstance(item, str) for item in received_argv)
    assert result.metadata["command"] == "find . -type f -maxdepth 3"
    assert result.metadata["normalization_code"] == "TOOL_ARGUMENT_NORMALIZED"
    assert result.metadata["argument_normalizations"] == [
        {
            "field": "argv",
            "index": 5,
            "original_type": "int",
            "normalized_type": "str",
            "reason": "json_scalar_to_string",
        }
    ]


class CapturingFakeProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]):
        super().__init__(responses)
        self.message_batches: list[list[dict]] = []

    def generate(self, messages: list[dict], tools: list[dict], tool_choice=None) -> ModelResponse:
        self.message_batches.append(messages)
        return super().generate(messages, tools, tool_choice)


def test_agent_loop_isolates_bad_call_and_recovers_with_later_calls(tmp_path: Path) -> None:
    provider = CapturingFakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="bad", name="bash", arguments={"argv": ["echo", {"bad": 3}]}),
                    ToolCall(id="write", name="write_file", arguments={"path": "continued.txt", "content": "continued"}),
                ]
            ),
            ModelResponse(tool_calls=[ToolCall(id="fixed", name="bash", arguments={"argv": ["python", "-c", "print('ok')", 3]})]),
            ModelResponse(final_answer=FinalAnswer(content="done")),
        ]
    )

    result = AgentLoop(_config(tmp_path), provider, run_id="argument-recovery").run(tmp_path, "recover")

    assert result.status == RunStatus.COMPLETED
    assert (tmp_path / "continued.txt").read_text(encoding="utf-8") == "continued"
    tool_messages = [message for message in provider.message_batches[1] if message.get("role") == "tool"]
    assert {message["tool_call_id"] for message in tool_messages} == {"bad", "write"}
    bad_result = json.loads(next(message["content"] for message in tool_messages if message["tool_call_id"] == "bad"))
    assert bad_result["error_type"] == "invalid_tool_arguments"
    assert bad_result["retryable"] is True

    events = [json.loads(line) for line in Path(result.event_log_path).read_text(encoding="utf-8").splitlines()]
    assert any(event["event_type"] == "tool_arguments_normalized" for event in events)
    normalized = next(event for event in events if event["event_type"] == "tool_arguments_normalized")
    assert normalized["tool_call_id"] == "fixed"
    assert normalized["payload"]["index"] == 3
    assert normalized["payload"]["original_type"] == "int"
    assert normalized["payload"]["normalized_type"] == "str"
