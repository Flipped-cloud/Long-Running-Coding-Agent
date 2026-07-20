from longrun_agent.protocol import AgentToolCall, EventRecord, ModelResponse, ToolResult


def test_protocol_serialization_roundtrip():
    call = AgentToolCall(call_id="c1", tool_name="read_file", arguments={"path": "a.py"})
    response = ModelResponse(kind="tool_calls", tool_calls=[call], usage={"total_tokens": 3})
    restored = ModelResponse.model_validate_json(response.model_dump_json())
    assert restored.tool_calls[0].tool_name == "read_file"

    result = ToolResult(tool_call_id="c1", tool_name="read_file", success=True, summary="ok")
    assert ToolResult.model_validate(result.model_dump()).summary == "ok"

    retryable = ToolResult(
        tool_call_id="c2",
        tool_name="bash",
        success=False,
        summary="invalid",
        error_type="invalid_tool_arguments",
        retryable=True,
    )
    assert ToolResult.model_validate_json(retryable.model_dump_json()).retryable is True

    event = EventRecord(
        event_id="e1",
        run_id="r1",
        step=1,
        event_type="run_started",
        timestamp="2026-01-01T00:00:00+00:00",
        elapsed_seconds=0,
    )
    assert EventRecord.model_validate_json(event.model_dump_json()).run_id == "r1"
