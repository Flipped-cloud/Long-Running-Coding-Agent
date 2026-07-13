from __future__ import annotations

import json
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from longrun_agent.config import ModelConfig, get_api_key
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 429, 500, 502, 503, 504}
    return True


class OpenAICompatibleProvider(ModelProvider):
    """OpenAI SDK based provider for OpenAI-compatible chat completions."""

    def __init__(self, config: ModelConfig, client: Any | None = None):
        self.config = config
        self.client = client
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is required for real API calls") from exc
            kwargs: dict[str, Any] = {"api_key": get_api_key(config), "timeout": config.request_timeout_seconds}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self.client = OpenAI(**kwargs)

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(self.config.max_api_retries),
            reraise=True,
        )
        def call_api():
            return self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )

        return self._convert_response(call_api())

    @staticmethod
    def _convert_response(response: Any) -> ModelResponse:
        choice = response.choices[0]
        message = choice.message
        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            calls: list[ToolCall] = []
            raw_tool_calls = []
            for tool_call in tool_calls:
                function = tool_call.function
                try:
                    arguments = json.loads(function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ToolArgumentsProtocolError(function.name, str(exc), function.arguments or "") from exc
                calls.append(ToolCall(id=tool_call.id, name=function.name, arguments=arguments))
                raw_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {"name": function.name, "arguments": function.arguments or "{}"},
                    }
                )
            return ModelResponse(
                tool_calls=calls,
                usage=usage,
                provider_request_id=getattr(response, "id", None),
                raw_metadata={"message": {"role": "assistant", "content": getattr(message, "content", None), "tool_calls": raw_tool_calls}},
            )
        return ModelResponse(
            final_answer=FinalAnswer(content=getattr(message, "content", "") or ""),
            usage=usage,
            provider_request_id=getattr(response, "id", None),
            raw_metadata={"message": {"role": "assistant", "content": getattr(message, "content", "") or ""}},
        )
