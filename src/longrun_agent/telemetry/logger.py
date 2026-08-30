from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.protocol import EventRecord, RunResult

SENSITIVE_KEY_NAMES = {
    "api_key",
    "openai_api_key",
    "model_api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "authorization",
    "password",
    "client_secret",
    "secret",
}
SENSITIVE_KEY_SUFFIXES = ("_api_key", "_access_token", "_refresh_token", "_id_token", "_password", "_secret")


def is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in SENSITIVE_KEY_NAMES or normalized.endswith(SENSITIVE_KEY_SUFFIXES)


def redact_sensitive_values(text: str, sensitive_values: tuple[str, ...] = ()) -> str:
    for value in sensitive_values:
        if value:
            text = text.replace(value, "[redacted]")
    for name, value in os.environ.items():
        if value and len(value) >= 8 and is_sensitive_key(name):
            text = text.replace(value, "[redacted]")
    return text


def sanitize_payload(value: Any, sensitive_values: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(key):
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = sanitize_payload(item, sensitive_values)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item, sensitive_values) for item in value]
    if isinstance(value, str):
        value = redact_sensitive_values(value, sensitive_values)
        if len(value) > 4000:
            return value[:2000] + "\n...[truncated]...\n" + value[-2000:]
    return value


class EventLogger:
    """Flush-on-write JSONL event logger for a single run."""

    def __init__(self, run_id: str, run_dir: Path, model_name: str, sensitive_values: tuple[str, ...] = ()):
        self.run_id = run_id
        self.run_dir = run_dir
        self.model_name = model_name
        self.sensitive_values = sensitive_values
        self.events_path = run_dir / "events.jsonl"
        self.run_json_path = run_dir / "run.json"
        self.prompts_dir = run_dir / "prompts"
        self.tool_outputs_dir = run_dir / "tool_outputs"
        self.diffs_dir = run_dir / "diffs"
        self.started = time.monotonic()
        for directory in [self.run_dir, self.prompts_dir, self.tool_outputs_dir, self.diffs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def log(
        self,
        step: int,
        event_type: str,
        *,
        action_type: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        sanitized_arguments: dict[str, Any] | None = None,
        success: bool | None = None,
        summary: str | None = None,
        duration_seconds: float | None = None,
        exit_code: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        artifact_path: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        retryable: bool | None = None,
        sanitized_message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        record = EventRecord(
            run_id=self.run_id,
            step=step,
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            model_name=self.model_name,
            action_type=action_type,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            sanitized_arguments=sanitize_payload(sanitized_arguments, self.sensitive_values) if sanitized_arguments else None,
            success=success,
            summary=sanitize_message(summary, self.sensitive_values),
            duration_seconds=duration_seconds,
            exit_code=exit_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            artifact_path=artifact_path,
            error_type=error_type,
            error_message=sanitize_message(error_message, self.sensitive_values),
            retryable=retryable,
            sanitized_message=sanitize_message(sanitized_message, self.sensitive_values),
            payload=sanitize_payload(payload or {}, self.sensitive_values),
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
        return record

    def save_prompt(self, step: int, data: dict[str, Any]) -> Path:
        path = self.prompts_dir / f"model-turn-{step:04d}.json"
        path.write_text(json.dumps(sanitize_payload(data, self.sensitive_values), indent=2), encoding="utf-8")
        return path

    def save_run(self, result: RunResult, stats: dict[str, Any] | None = None) -> None:
        payload = sanitize_payload(result.model_dump(mode="json"), self.sensitive_values)
        payload["stats"] = sanitize_payload(stats or {}, self.sensitive_values)
        self.run_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sanitize_message(message: str | None, sensitive_values: tuple[str, ...] = ()) -> str | None:
    if message is None:
        return None
    text = message[:1000]
    lowered = text.lower()
    if any(marker in lowered for marker in ("api_key=", "authorization:", "bearer ")):
        return "[redacted sensitive error message]"
    return redact_sensitive_values(text, sensitive_values)
