import pytest

from longrun_agent.context.buffer import ContextBuffer


def test_context_buffer_exports_full_history_messages():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    buffer.add_user_reminder("remember")
    buffer.add_assistant_final({"role": "assistant", "content": "done"})

    assert buffer.export_messages() == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "remember"},
        {"role": "assistant", "content": "done"},
    ]


def test_context_buffer_keeps_multi_tool_call_turn_atomic():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ],
    }

    buffer.add_assistant_tool_turn(assistant, step=1)
    buffer.add_tool_result({"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "read"})
    buffer.add_tool_result({"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "bash"})
    buffer.finalize_turn()

    messages = buffer.export_messages()
    assert messages[2] == assistant
    assert [message["tool_call_id"] for message in messages[3:]] == ["c1", "c2"]
    assert buffer.recent_turns(1)[0].tool_call_ids == ["c1", "c2"]


def test_context_buffer_rejects_orphan_tool_result():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")

    with pytest.raises(ValueError, match="without an active assistant tool turn"):
        buffer.add_tool_result({"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "read"})


def test_context_buffer_rejects_incomplete_tool_turn_export():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    buffer.add_assistant_tool_turn(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        step=1,
    )

    with pytest.raises(ValueError, match="active tool interaction turn"):
        buffer.export_messages()


def test_context_buffer_rejects_mismatched_tool_result():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    buffer.add_assistant_tool_turn(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        step=1,
    )

    with pytest.raises(ValueError, match="does not match"):
        buffer.add_tool_result({"role": "tool", "tool_call_id": "wrong", "name": "read_file", "content": "read"})


def test_context_buffer_segment_counter_does_not_drop_history_yet():
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    assert buffer.start_new_segment() == 2
    snapshot = buffer.snapshot()
    assert snapshot["context_segment_id"] == 2
    assert snapshot["context_reset_count"] == 1
    assert snapshot["message_count"] == 2
