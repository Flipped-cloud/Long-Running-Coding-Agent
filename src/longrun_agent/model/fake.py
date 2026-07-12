from __future__ import annotations

from collections import deque

from longrun_agent.exceptions import ProviderError
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall


class FakeModelProvider(ModelProvider):
    """Deterministic provider that returns scripted responses for tests and demos."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = deque(responses)
        self.calls = 0

    def generate(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        if not self._responses:
            raise ProviderError("FakeModelProvider scripted responses exhausted")
        return self._responses.popleft()


def default_calculator_script() -> list[ModelResponse]:
    fixed = """def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError(\"division by zero\")
    return a / b
"""
    return [
        ModelResponse(tool_calls=[ToolCall(id="read-1", name="read_file", arguments={"path": "calculator.py"})]),
        ModelResponse(tool_calls=[ToolCall(id="write-1", name="write_file", arguments={"path": "calculator.py", "content": fixed})]),
        ModelResponse(tool_calls=[ToolCall(id="bash-1", name="bash", arguments={"command": "python -m pytest -q"})]),
        ModelResponse(final_answer=FinalAnswer(content="Fixed calculator.py and verified the test suite with python -m pytest -q.")),
    ]
