from __future__ import annotations

from pathlib import Path

from longrun_agent.orchestration.session_trace import READ_ONLY_STREAK_LIMIT, SessionTrace
from longrun_agent.protocol import ErrorType, ToolCall, ToolResult


def test_bash_observation_records_failed_output_and_sanitizes_secret(tmp_path: Path) -> None:
    trace = SessionTrace()
    artifact = tmp_path / "tool-output.txt"
    artifact.write_text("FULL SECRET_TOKEN=abc\n", encoding="utf-8")
    output = "\n".join(
        [
            "STDOUT:",
            "tests/test_task_app.py::test_validate_task_name_rejects_empty FAILED",
            "AssertionError: assert True is False",
            "Authorization: Bearer hidden",
        ]
    )
    trace.record(
        ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]}),
        ToolResult(
            tool_call_id="b1",
            tool_name="bash",
            success=True,
            summary="bash finished with exit code 1",
            output=output,
            metadata={
                "command": "python -m pytest -q",
                "argv": ["python", "-m", "pytest", "-q"],
                "exit_code": 1,
                "output_artifact": str(artifact),
            },
            artifact_path=str(artifact),
        ),
    )

    observation = trace.bash_observations[0]
    assert observation.command == "python -m pytest -q"
    assert observation.argv == ["python", "-m", "pytest", "-q"]
    assert observation.exit_code == 1
    assert observation.is_verification is True
    assert observation.artifact_path == "tool-output.txt"
    assert "AssertionError" in observation.output_excerpt
    assert "Bearer hidden" not in observation.output_excerpt
    assert "[redacted credential line]" in observation.output_excerpt


def test_bash_observation_excerpt_preserves_head_and_tail() -> None:
    trace = SessionTrace()
    long_output = "HEAD\n" + ("x" * 5000) + "\nTAIL AssertionError"
    trace.record(
        ToolCall(id="b1", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q"]}),
        ToolResult(
            tool_call_id="b1",
            tool_name="bash",
            success=True,
            summary="bash finished with exit code 1",
            output=long_output,
            metadata={"command": "python -m pytest -q", "argv": ["python", "-m", "pytest", "-q"], "exit_code": 1},
        ),
    )

    excerpt = trace.bash_observations[0].output_excerpt
    assert excerpt.startswith("HEAD")
    assert excerpt.endswith("TAIL AssertionError")
    assert "...[truncated]..." in excerpt
    assert len(excerpt) < len(long_output)


def test_unsupported_shell_syntax_sets_recoverable_action_message() -> None:
    trace = SessionTrace()
    trace.record(
        ToolCall(id="b1", name="bash", arguments={"command": "cd repo && pytest -q"}),
        ToolResult(
            tool_call_id="b1",
            tool_name="bash",
            success=False,
            summary="unsupported_shell_syntax",
            error_type=ErrorType.PROTOCOL,
            error_message="unsupported_shell_syntax",
            metadata={"command": "cd repo && pytest -q", "unsupported_shell_syntax": True},
        ),
    )

    assert trace.unsupported_shell_syntax_count == 1
    assert "retry the same intended command once using argv" in (trace.action_required_message or "")


def test_invalid_raw_bash_argv_does_not_break_trace() -> None:
    trace = SessionTrace()
    call = ToolCall(id="bad", name="bash", arguments={"argv": ["echo", {"bad": 3}]})

    trace.record(
        call,
        ToolResult(
            tool_call_id="bad",
            tool_name="bash",
            success=False,
            summary="invalid arguments for bash",
            error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
            retryable=True,
        ),
    )

    assert trace.bash_observations[0].command == ""
    assert trace.bash_observations[0].argv == []


def test_pytest_collect_only_is_inspection_not_verification_progress() -> None:
    trace = SessionTrace()
    trace.record(
        ToolCall(
            id="collect",
            name="bash",
            arguments={"argv": ["python", "-m", "pytest", "--collect-only", "-q"]},
        ),
        ToolResult(
            tool_call_id="collect",
            tool_name="bash",
            success=True,
            summary="bash finished with exit code 0",
            metadata={
                "command": "python -m pytest --collect-only -q",
                "argv": ["python", "-m", "pytest", "--collect-only", "-q"],
                "exit_code": 0,
            },
        ),
    )

    assert trace.successful_test_commands == []
    assert trace.successful_acceptance_commands == []
    assert trace.bash_observations[0].is_verification is False
    assert trace.read_only_streak == 1
    assert trace.no_progress(progress_count=0, terminal_signal=None) is True


def test_policy_gate_does_not_turn_read_only_loop_into_progress() -> None:
    trace = SessionTrace()
    trace.record_policy_gate(
        ToolResult(
            tool_call_id="gate",
            tool_name="read_file",
            success=False,
            summary="repeated_tool_call_suppressed",
            error_type=ErrorType.POLICY_GATE,
        )
    )

    assert trace.no_progress(progress_count=0, terminal_signal=None) is True


def test_bash_call_key_uses_normalized_argv() -> None:
    trace = SessionTrace()

    numeric = trace.call_key(ToolCall(id="numeric", name="bash", arguments={"argv": ["find", ".", 3]}))
    string = trace.call_key(ToolCall(id="string", name="bash", arguments={"argv": ["find", ".", "3"]}))

    assert numeric == string


def test_repeated_read_is_suppressed_until_a_file_changes() -> None:
    trace = SessionTrace()
    read_a = ToolCall(id="a1", name="read_file", arguments={"path": "a.py"})
    read_b = ToolCall(id="b1", name="read_file", arguments={"path": "b.py"})

    for call in (read_a, read_b):
        trace.record(
            call,
            ToolResult(
                tool_call_id=call.id,
                tool_name="read_file",
                success=True,
                summary="read succeeded",
                metadata={"path": call.arguments["path"]},
            ),
        )

    assert trace.should_suppress(ToolCall(id="a2", name="read_file", arguments={"path": "a.py"}))
    trace.record_suppressed(ToolCall(id="a2", name="read_file", arguments={"path": "a.py"}))
    assert "next call must edit, run focused tests, complete, or report a blocker" in (trace.action_required_message or "")

    trace.record(
        ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "VALUE = 2\n"}),
        ToolResult(
            tool_call_id="w1",
            tool_name="write_file",
            success=True,
            summary="write succeeded",
            metadata={"path": "a.py", "status": "updated"},
        ),
    )
    assert not trace.should_suppress(ToolCall(id="a3", name="read_file", arguments={"path": "a.py"}))


def test_read_only_limit_blocks_new_files_and_bash_inspection() -> None:
    trace = SessionTrace()
    paths = tuple(f"{index}.py" for index in range(READ_ONLY_STREAK_LIMIT))
    for index, path in enumerate(paths, start=1):
        trace.record(
            ToolCall(id=f"r{index}", name="read_file", arguments={"path": path}),
            ToolResult(
                tool_call_id=f"r{index}",
                tool_name="read_file",
                success=True,
                summary="read succeeded",
                metadata={"path": path},
            ),
        )

    assert trace.should_suppress(ToolCall(id="r-next", name="read_file", arguments={"path": "new.py"}))
    assert trace.should_suppress(ToolCall(id="b1", name="bash", arguments={"argv": ["cat", "-n", "new.py"]}))
    assert not trace.should_suppress(
        ToolCall(id="b2", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q", "tests/test_new.py"]})
    )
