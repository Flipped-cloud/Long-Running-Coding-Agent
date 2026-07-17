from types import SimpleNamespace

import pytest
from tenacity import wait_none

import longrun_agent.model.openai_compatible as provider_module
from longrun_agent.config import ModelConfig
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider, _is_retryable


class StatusError(Exception):
    def __init__(self, status_code: int, message: str | None = None):
        super().__init__(message or f"status {status_code}")
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def config(max_api_retries: int = 3) -> ModelConfig:
    return ModelConfig(
        provider="openai_compatible",
        model_name="test-model",
        api_key_env="MISSING_API_KEY",
        temperature=0.3,
        max_output_tokens=123,
        request_timeout_seconds=1,
        max_api_retries=max_api_retries,
    )


def deepseek_config(max_api_retries: int = 3) -> ModelConfig:
    cfg = config(max_api_retries=max_api_retries)
    cfg.model_name = "deepseek-v4-flash"
    return cfg


def response_with_message(message, *, usage=None, response_id="resp-1"):
    return SimpleNamespace(id=response_id, choices=[SimpleNamespace(message=message)], usage=usage)


def usage(prompt_tokens=100, completion_tokens=20, total_tokens=120):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)


def text_message(content="done"):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_generate_passes_expected_request_parameters(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    response = response_with_message(text_message("done"))
    client = FakeClient([response])
    provider = OpenAICompatibleProvider(config(), client=client)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    provider.generate(messages, tools)

    assert len(client.completions.calls) == 1
    kwargs = client.completions.calls[0]
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == messages
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 123


def test_provider_returns_final_answer_and_usage():
    message = text_message("done")
    result = OpenAICompatibleProvider._convert_response(response_with_message(message, usage=usage()))
    assert result.kind == "final_answer"
    assert result.final_answer.content == "done"
    assert result.provider_request_id == "resp-1"
    assert result.usage == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def test_provider_returns_one_tool_call_with_raw_metadata():
    message = SimpleNamespace(content=None, tool_calls=[tool_call("c1", "read_file", '{"path":"calculator.py"}')])
    result = OpenAICompatibleProvider._convert_response(response_with_message(message, usage=usage(1, 2, 3), response_id="resp-tools"))
    assert result.kind == "tool_calls"
    assert result.provider_request_id == "resp-tools"
    assert result.tool_calls[0].id == "c1"
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "calculator.py"}
    raw_call = result.raw_metadata["message"]["tool_calls"][0]
    assert raw_call["id"] == "c1"
    assert raw_call["function"]["name"] == "read_file"
    assert raw_call["function"]["arguments"] == '{"path":"calculator.py"}'
    assert result.usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


def test_provider_returns_multiple_tool_calls():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            tool_call("c1", "read_file", '{"path":"calculator.py"}'),
            tool_call("c2", "bash", '{"command":"python -m pytest -q"}'),
        ],
    )
    result = OpenAICompatibleProvider._convert_response(response_with_message(message))
    assert [call.id for call in result.tool_calls] == ["c1", "c2"]
    assert [call.name for call in result.tool_calls] == ["read_file", "bash"]
    assert result.tool_calls[1].arguments["command"] == "python -m pytest -q"


def test_provider_invalid_tool_call_json_raises_clear_error():
    message = SimpleNamespace(content=None, tool_calls=[tool_call("c1", "read_file", "{path: calculator.py}")])
    with pytest.raises(ToolArgumentsProtocolError, match="invalid tool call JSON"):
        OpenAICompatibleProvider._convert_response(response_with_message(message))


def test_provider_handles_missing_usage():
    result = OpenAICompatibleProvider._convert_response(response_with_message(text_message("done"), usage=None))
    assert result.kind == "final_answer"
    assert result.usage == {}


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_is_retryable_status_codes(status_code, expected):
    assert _is_retryable(StatusError(status_code)) is expected


def test_generate_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    client = FakeClient([StatusError(429), response_with_message(text_message("done"))])
    result = OpenAICompatibleProvider(config(max_api_retries=2), client=client).generate([], [])
    assert result.final_answer.content == "done"
    assert len(client.completions.calls) == 2


def test_generate_falls_back_to_auto_when_named_tool_choice_is_unsupported(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    unsupported = StatusError(400, "InvalidParameter: The tool_choice parameter does not support being set to object")
    client = FakeClient([unsupported, response_with_message(text_message("done"))])
    provider = OpenAICompatibleProvider(config(max_api_retries=1), client=client)
    named = {"type": "function", "function": {"name": "read_file"}}

    result = provider.generate([], [{"type": "function", "function": {"name": "read_file"}}], tool_choice=named)

    assert result.final_answer.content == "done"
    assert [call["tool_choice"] for call in client.completions.calls] == [named, "auto"]


def test_generate_disables_deepseek_v4_thinking_for_named_tool_choice(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    client = FakeClient([response_with_message(text_message("done"))])
    provider = OpenAICompatibleProvider(deepseek_config(max_api_retries=1), client=client)
    named = {"type": "function", "function": {"name": "read_file"}}

    result = provider.generate([], [{"type": "function", "function": {"name": "read_file"}}], tool_choice=named)

    assert result.final_answer.content == "done"
    assert client.completions.calls[0]["tool_choice"] == named
    assert client.completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_generate_retries_500_then_succeeds(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    client = FakeClient([StatusError(500), response_with_message(text_message("done"))])
    result = OpenAICompatibleProvider(config(max_api_retries=2), client=client).generate([], [])
    assert result.final_answer.content == "done"
    assert len(client.completions.calls) == 2


def test_generate_does_not_retry_400(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    client = FakeClient([StatusError(400), response_with_message(text_message("done"))])
    with pytest.raises(StatusError):
        OpenAICompatibleProvider(config(max_api_retries=3), client=client).generate([], [])
    assert len(client.completions.calls) == 1


def test_generate_reraises_after_max_retries(monkeypatch):
    monkeypatch.setattr(provider_module, "wait_exponential", lambda *args, **kwargs: wait_none())
    client = FakeClient([StatusError(503), StatusError(503), StatusError(503)])
    with pytest.raises(StatusError):
        OpenAICompatibleProvider(config(max_api_retries=3), client=client).generate([], [])
    assert len(client.completions.calls) == 3
