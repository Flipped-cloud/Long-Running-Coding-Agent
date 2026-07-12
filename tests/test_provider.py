from types import SimpleNamespace

from longrun_agent.model.openai_compatible import OpenAICompatibleProvider


def response_with_message(message):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_provider_returns_tool_call():
    tool_call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    result = OpenAICompatibleProvider._convert_response(response_with_message(message))
    assert result.kind == "tool_calls"
    assert result.tool_calls[0].arguments["path"] == "a.py"
    assert result.usage["total_tokens"] == 3


def test_provider_returns_final_answer():
    message = SimpleNamespace(content="done", tool_calls=None)
    result = OpenAICompatibleProvider._convert_response(response_with_message(message))
    assert result.kind == "final_answer"
    assert result.final_answer.content == "done"
