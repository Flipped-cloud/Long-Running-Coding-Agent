from __future__ import annotations

from typing import Any

from longrun_agent.evaluation.failure_taxonomy import FAILURE_LAYERS, FailureCode
from longrun_agent.evaluation.schema import FailureAttribution, FailureLayer, TerminationReason, TrajectoryFeatures


class FailureAttributor:
    def attribute(
        self,
        *,
        case_id: str,
        trial_id: str,
        termination_reason: TerminationReason,
        features: TrajectoryFeatures,
        events: list[dict[str, Any]],
    ) -> FailureAttribution:
        codes: list[FailureCode] = []
        if termination_reason == TerminationReason.CONTEXT_LIMIT:
            codes.append(FailureCode.CONTEXT_LIMIT_EXCEEDED)
        if termination_reason == TerminationReason.SESSION_LIMIT:
            codes.append(FailureCode.RESOURCE_SESSION_EXHAUSTED)
        if termination_reason == TerminationReason.TIME_LIMIT:
            codes.append(FailureCode.RESOURCE_TIME_EXHAUSTED)
        event_types = {str(item.get("event_type") or "") for item in events}
        oracle_event = next(
            (item for item in reversed(events) if item.get("event_type") == "oracle_verification_finished"),
            None,
        )
        oracle_verdict = str(oracle_event.get("oracle_verdict")) if oracle_event else None
        runtime_verified = any(
            item.get("event_type") == "verification_report_created" and item.get("verdict") == "verified" for item in events
        )
        completion_claimed = bool(features.first_completion_request or runtime_verified)
        verification_succeeded = "project_verified" in event_types or any(
            item.get("event_type") == "verification_report_created" and item.get("verdict") == "verified" for item in events
        )
        if completion_claimed and oracle_verdict is not None and oracle_verdict != "verified":
            codes[0:0] = [FailureCode.COMPLETION_PREMATURE_REQUEST, FailureCode.VERIFICATION_FALSE_POSITIVE]
        if features.first_protocol_error:
            codes.append(FailureCode.PROTOCOL_INVALID_FORMAT)
        if features.first_invalid_tool_call:
            codes.append(FailureCode.TOOL_INVALID_ARGUMENT)
        if (
            features.first_integrity_violation
            or "integrity_violation_detected" in event_types
            or (oracle_event is not None and oracle_event.get("integrity_passed") is False)
        ):
            codes.append(FailureCode.VERIFICATION_TEST_TAMPERING)
        if {"verification_contract_mismatch", "verification_baseline_mismatch", "oracle_baseline_mismatch"} & event_types:
            codes.append(FailureCode.VERIFICATION_BASELINE_MISMATCH)
        if "verification_infrastructure_error" in event_types or oracle_verdict == "infrastructure_error":
            codes.append(FailureCode.VERIFICATION_INFRA_FAILURE)
        if not codes and features.first_completion_request and features.first_failed_required_check:
            codes.extend([FailureCode.COMPLETION_PREMATURE_REQUEST, FailureCode.VERIFICATION_FALSE_POSITIVE])
        elif (
            features.first_completion_request
            and features.latest_valid_verification_before_completion is None
            and not verification_succeeded
            and oracle_verdict != "verified"
        ):
            codes.append(FailureCode.COMPLETION_PREMATURE_REQUEST)
        if features.first_harmful_knowledge_use:
            codes.append(FailureCode.KNOWLEDGE_NEGATIVE_TRANSFER)
        if features.first_repeated_action:
            codes.append(FailureCode.RECOVERY_REPEATED_WORK)
        if "verification_transition_p2f" in event_types or (
            oracle_event is not None and float(oracle_event.get("f2p_rate") or 0) == 1.0 and float(oracle_event.get("p2p_rate") or 0) < 1.0
        ):
            codes.append(FailureCode.IMPLEMENTATION_REGRESSION)
        codes = list(dict.fromkeys(codes)) or [FailureCode.UNKNOWN]
        primary = codes[0]
        evidence = [
            item for item in [features.first_observable_symptom, features.first_causal_divergence, features.final_outcome_event] if item
        ]
        uncertain = primary in {FailureCode.UNKNOWN, FailureCode.KNOWLEDGE_NEGATIVE_TRANSFER}
        return FailureAttribution(
            case_id=case_id,
            trial_id=trial_id,
            termination_reason=termination_reason,
            primary_layer=FAILURE_LAYERS.get(primary, FailureLayer.UNKNOWN),
            primary_code=primary.value,
            secondary_codes=[item.value for item in codes[1:]],
            first_symptom_event_id=features.first_observable_symptom,
            first_divergence_event_id=features.first_causal_divergence,
            evidence_event_ids=list(dict.fromkeys(evidence)),
            explanation=f"Deterministic trajectory rules selected {primary.value}.",
            confidence=0.45 if uncertain else 0.9,
            deterministic=True,
            needs_human_review=uncertain,
        )
