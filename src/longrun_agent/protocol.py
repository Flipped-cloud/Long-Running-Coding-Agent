from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ErrorType(StrEnum):
    """Runtime error categories surfaced to the agent harness."""

    PROVIDER = "provider_error"
    PROTOCOL = "protocol_error"
    TOOL = "tool_error"
    ENVIRONMENT = "environment_observation"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    TIME_LIMIT_REACHED = "time_limit_reached"
    FINAL_WITHOUT_TERMINAL_SIGNAL = "final_without_terminal_signal"
    TERMINAL_SIGNAL_MISSING = "terminal_signal_missing"
    ABORTED = "aborted"
    PROVIDER_ERROR = "provider_error"
    PROTOCOL_ERROR = "protocol_error"
    CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"


class ToolCall(BaseModel):
    """A normalized native tool call returned by a model provider."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1, validation_alias=AliasChoices("id", "call_id"))
    name: str = Field(min_length=1, validation_alias=AliasChoices("name", "tool_name"))
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def call_id(self) -> str:
        return self.id

    @property
    def tool_name(self) -> str:
        return self.name


class FinalAnswer(BaseModel):
    """A final model answer with no further tool calls."""

    content: str


class AgentAction(BaseModel):
    """One of the two actions the runtime accepts from the model."""

    type: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    final_answer: FinalAnswer | None = None


class ToolResult(BaseModel):
    """Structured observation returned by a tool."""

    tool_call_id: str
    tool_name: str
    success: bool
    summary: str
    output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None


class ModelResponse(BaseModel):
    """Provider response normalized for the agent loop."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: FinalAnswer | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    provider_request_id: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def kind(self) -> Literal["tool_calls", "final_answer"]:
        return "tool_calls" if self.tool_calls else "final_answer"


class RunResult(BaseModel):
    """Final runtime result and aggregate counters."""

    run_id: str
    status: RunStatus
    final_answer: str | None = None
    steps: int
    consecutive_errors: int
    started_at: str
    finished_at: str
    event_log_path: str
    run_json_path: str
    tool_call_count: int = 0
    total_tokens: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    compactor_input_tokens: int = 0
    compactor_output_tokens: int = 0
    max_estimated_input_tokens: int = 0
    max_actual_input_tokens: int = 0
    max_context_usage_ratio: float = 0.0
    context_segment_count: int = 1
    context_reset_count: int = 0
    deterministic_prune_count: int = 0
    structured_compaction_count: int = 0
    pruned_item_count: int = 0
    stale_item_count: int = 0
    superseded_item_count: int = 0
    estimated_tokens_removed: int = 0
    context_budget_exhausted: bool = False
    latest_context_handoff_id: str | None = None
    terminal_grace_turn_count: int = 0
    terminal_signal_recovered: bool = False
    tool_argument_protocol_retry_count: int = 0


class EventRecord(BaseModel):
    """One JSONL telemetry event."""

    run_id: str
    step: int
    timestamp: str
    event_type: str
    model_name: str | None = None
    action_type: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    sanitized_arguments: dict[str, Any] | None = None
    success: bool | None = None
    summary: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    artifact_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


# Backward-compatible aliases used by earlier tests and local scripts.
AgentToolCall = ToolCall
AgentFinalAnswer = FinalAnswer
AgentRunResult = RunResult
