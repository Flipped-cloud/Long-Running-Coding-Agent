from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from longrun_agent.evaluation.schema import TrajectoryFeatures

EVENT_FEATURES = {
    "protocol_error": "first_protocol_error",
    "fatal_protocol_error": "first_protocol_error",
    "invalid_tool_call": "first_invalid_tool_call",
    "tool_argument_error": "first_invalid_tool_call",
    "repeated_tool_call_suppressed": "first_repeated_action",
    "no_progress_session": "first_no_progress_session",
    "context_stale_observation": "first_stale_context_use",
    "memory_harmful": "first_harmful_knowledge_use",
    "skill_harmful": "first_harmful_knowledge_use",
    "task_completion_requested": "first_completion_request",
    "verification_check_failed": "first_failed_required_check",
    "integrity_violation_detected": "first_integrity_violation",
    "task_reopened": "first_reopen",
}


class TrajectoryFeatureExtractor:
    def extract(self, event_streams: Iterable[Iterable[dict[str, Any]]]) -> TrajectoryFeatures:
        events = [event for stream in event_streams for event in stream]
        events.sort(key=lambda event: (str(event.get("timestamp") or ""), int(event.get("step") or 0)))
        values: dict[str, str | None] = {}
        valid_verification = None
        for event in events:
            event_type = str(event.get("event_type") or "")
            event_id = _event_id(event)
            feature = EVENT_FEATURES.get(event_type)
            if feature and feature not in values:
                values[feature] = event_id
            if event_type in {"verification_report_created", "project_verified", "task_verified"} and event.get("verdict") in {
                None,
                "verified",
            }:
                valid_verification = event_id
            if event_type == "verification_transition_computed" and event.get("verdict") not in {"F2P", "P2P", "X2P"}:
                values.setdefault("first_failed_required_check", event_id)
            if event_type in {"task_completion_requested", "completion_request"}:
                values.setdefault("first_completion_request", event_id)
                values["latest_valid_verification_before_completion"] = valid_verification
            if event_type in {
                "context_budget_exhausted",
                "project_session_limit_reached",
                "project_time_limit_reached",
                "token_budget_exhausted",
            }:
                values.setdefault("budget_exhaustion_event", event_id)
            if event_type in {"run_finished", "project_verified", "project_failed", "trial_finished"}:
                values["final_outcome_event"] = event_id
        symptom_candidates = [
            values.get("first_protocol_error"),
            values.get("first_invalid_tool_call"),
            values.get("first_failed_required_check"),
            values.get("first_integrity_violation"),
            values.get("budget_exhaustion_event"),
        ]
        values["first_observable_symptom"] = next((item for item in symptom_candidates if item), None)
        divergence_candidates = [
            values.get("first_integrity_violation"),
            values.get("first_protocol_error"),
            values.get("first_harmful_knowledge_use"),
            values.get("first_repeated_action"),
            values.get("first_failed_required_check"),
        ]
        values["first_causal_divergence"] = next((item for item in divergence_candidates if item), None)
        return TrajectoryFeatures.model_validate(values)


def _event_id(event: dict[str, Any]) -> str:
    return str(event.get("event_id") or event.get("report_id") or event.get("run_id") or event.get("timestamp") or "unknown-event")
