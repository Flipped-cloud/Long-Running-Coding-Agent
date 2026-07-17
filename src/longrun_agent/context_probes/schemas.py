from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProbeName = Literal["position", "multi_constraint", "state_tracking", "aggregation"]
ProbeMode = Literal["full_history", "recent_window", "deterministic_prune", "structured_reset"]
ProbePosition = Literal["beginning", "middle", "end"]


class ProbeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    token_role: str = Field(default="filler")
    is_relevant: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    probe: ProbeName
    target_length_tokens: int = Field(ge=1)
    estimated_length_tokens: int = Field(ge=0)
    position: ProbePosition | None = None
    events: list[ProbeEvent] = Field(min_length=1)
    final_query: str = Field(min_length=1)
    expected: dict[str, Any]
    seed: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    probe: ProbeName
    mode: ProbeMode
    predicted: dict[str, Any] | None = None
    raw_final_text: str | None = None
    protocol_valid: bool = False
    error_type: str | None = None
    provider_error: str | None = None
    protocol_error: str | None = None
    estimated_input_tokens: int = 0
    actual_input_tokens: int = 0
    output_tokens: int = 0
    compactor_input_tokens: int = 0
    compactor_output_tokens: int = 0
    latency_seconds: float = 0.0
    context_reset_count: int = 0
    recent_window_trim_count: int = 0
    deterministic_prune_count: int = 0
    structured_reset_count: int = 0
    structured_compaction_count: int = 0
    stale_item_count: int = 0
    superseded_item_count: int = 0
    estimated_tokens_removed: int = 0
    context_budget_exhausted: bool = False
    answer_provider_calls: int = 0
    protocol_retry_count: int = 0
    required_answer_tool: str | None = None
    compactor_provider_calls: int = 0
    runtime_context_config: dict[str, Any] = Field(default_factory=dict)
    prompt_messages: list[dict[str, Any]] | None = None


class ProbeCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: ProbeCase
    prediction: ProbePrediction
    exact_match: bool
    metrics: dict[str, float]
    errors: list[str] = Field(default_factory=list)


class ProbeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    exact: int
    recall: float
    precision: float
    stale_rejection: float = 0.0
