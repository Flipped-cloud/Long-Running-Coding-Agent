from __future__ import annotations

from pathlib import Path

from longrun_agent.orchestration.session_trace import SessionTrace
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
                "verification_kind": "pytest",
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
            metadata={
                "command": "python -m pytest -q",
                "argv": ["python", "-m", "pytest", "-q"],
                "exit_code": 1,
                "verification_kind": "pytest",
            },
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
                "verification_kind": None,
            },
        ),
    )

    assert trace.successful_test_commands == []
    assert trace.successful_acceptance_commands == []
    assert trace.bash_observations[0].is_verification is False
    assert trace.no_progress(progress_count=0, terminal_signal=None) is True


def test_file_listing_with_pytest_path_is_not_verification() -> None:
    trace = SessionTrace()
    command = "find . -not -path './.pytest_cache/*'"
    trace.record(
        ToolCall(id="find", name="bash", arguments={"argv": ["find", "."]}),
        ToolResult(
            tool_call_id="find",
            tool_name="bash",
            success=True,
            summary="bash finished with exit code 0",
            metadata={"command": command, "exit_code": 0, "verification_kind": None},
        ),
    )

    assert trace.successful_test_commands == []
    assert trace.successful_acceptance_commands == []
    assert trace.bash_observations[0].is_verification is False


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


def test_repeated_reads_are_recorded_as_telemetry() -> None:
    trace = SessionTrace()
    call = ToolCall(id="a1", name="read_file", arguments={"path": "a.py"})
    result = ToolResult(
        tool_call_id=call.id,
        tool_name="read_file",
        success=True,
        summary="read succeeded",
        metadata={"path": "a.py"},
    )

    trace.record(call, result)
    trace.record(call.model_copy(update={"id": "a2"}), result.model_copy(update={"tool_call_id": "a2"}))

    assert trace.repeated_tool_calls
    assert trace.suppressed_tool_calls == []


def test_context_segment_restarts_repeat_tracking() -> None:
    trace = SessionTrace()
    call = ToolCall(id="a1", name="read_file", arguments={"path": "a.py"})
    result = ToolResult(tool_call_id="a1", tool_name="read_file", success=True, summary="read succeeded")

    trace.record(call, result)
    trace.start_context_segment()
    trace.record(call.model_copy(update={"id": "a2"}), result.model_copy(update={"tool_call_id": "a2"}))

    assert trace.repeated_tool_calls == []
