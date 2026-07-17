from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from longrun_agent.context_probes.schemas import ProbeName
from longrun_agent.protocol import ModelResponse


class PositionAnswer(BaseModel):
    constraints: list[str] = Field(default_factory=list)


class MultiConstraintAnswer(BaseModel):
    constraints: list[str] = Field(default_factory=list)


class StateTrackingAnswer(BaseModel):
    plan_version: int
    file_hash: str
    active_task: str
    verification_state: str
    latest_valid_test: str | None = None


class AggregationAnswer(BaseModel):
    current_test: str
    unresolved_failures: list[str] = Field(default_factory=list)
    stale_results_rejected: bool
    modified_files: list[str] = Field(default_factory=list)
    code_epoch: int


ANSWER_MODELS: dict[ProbeName, type[BaseModel]] = {
    "position": PositionAnswer,
    "multi_constraint": MultiConstraintAnswer,
    "state_tracking": StateTrackingAnswer,
    "aggregation": AggregationAnswer,
}

ANSWER_TOOL_NAMES: dict[ProbeName, str] = {
    "position": "submit_position_answer",
    "multi_constraint": "submit_multi_constraint_answer",
    "state_tracking": "submit_state_tracking_answer",
    "aggregation": "submit_aggregation_answer",
}


def answer_tool_schema(probe: ProbeName) -> dict[str, Any]:
    model = ANSWER_MODELS[probe]
    return {
        "type": "function",
        "function": {
            "name": ANSWER_TOOL_NAMES[probe],
            "description": f"Submit the structured answer for the {probe} context lifecycle probe.",
            "parameters": model.model_json_schema(),
        },
    }


def parse_probe_answer(probe: ProbeName, response: ModelResponse) -> tuple[dict[str, Any] | None, str | None]:
    tool_name = ANSWER_TOOL_NAMES[probe]
    if len(response.tool_calls) != 1:
        if not response.tool_calls:
            return None, f"missing required tool call: {tool_name}"
        return None, f"expected exactly one tool call: {tool_name}"
    call = response.tool_calls[0]
    if call.name != tool_name:
        return None, f"missing required tool call: {tool_name}"
    try:
        answer = ANSWER_MODELS[probe].model_validate(call.arguments)
    except ValidationError as exc:
        return None, f"invalid {tool_name} arguments: {exc}"
    return answer.model_dump(), None
