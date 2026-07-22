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
        oracle_verification_verdict: str | None = None,
        integrity_passed: bool | None = None,
        runtime_verification_verdict: str | None = None,
    ) -> FailureAttribution | None:
        codes: list[FailureCode] = []
        event_types = {str(item.get("event_type") or "") for item in events}
        oracle_event = next(
            (item for item in reversed(events) if item.get("event_type") == "oracle_verification_finished"),
            None,
        )
        oracle_verdict = oracle_verification_verdict
        if oracle_verdict is None and oracle_event:
            oracle_verdict = str(oracle_event.get("oracle_verdict"))
        oracle_integrity = integrity_passed
        if oracle_integrity is None and oracle_event:
            oracle_integrity = oracle_event.get("integrity_passed")
        if oracle_verdict == "verified" and oracle_integrity is True:
            return None
        runtime_verified = runtime_verification_verdict == "verified" or any(
            item.get("event_type") == "verification_report_created" and item.get("verdict") == "verified" for item in events
        )
        if features.first_integrity_violation or "integrity_violation_detected" in event_types or oracle_integrity is False:
            codes.append(FailureCode.VERIFICATION_TEST_TAMPERING)
        if runtime_verified and oracle_verdict not in {None, "verified"}:
            codes.extend([FailureCode.VERIFICATION_FALSE_POSITIVE, FailureCode.RUNTIME_ORACLE_DISAGREEMENT])
        if termination_reason == TerminationReason.CONTEXT_LIMIT:
            codes.append(FailureCode.CONTEXT_LIMIT_EXCEEDED)
        if termination_reason == TerminationReason.SESSION_LIMIT:
            codes.append(FailureCode.RESOURCE_SESSION_EXHAUSTED)
        if termination_reason == TerminationReason.TIME_LIMIT:
            codes.append(FailureCode.EXECUTION_TIME_LIMIT)
        if termination_reason == TerminationReason.TASK_LIMIT:
            codes.append(FailureCode.EXECUTION_STEP_LIMIT)
        generated_requirement_event = next(
            (item for item in reversed(events) if item.get("event_type") == "generated_test_requirement_unmet"),
            None,
        )
        if generated_requirement_event is not None:
            workflow = generated_requirement_event.get("payload") or {}
            registered = int(workflow.get("registered_candidates") or 0)
            valid = int(workflow.get("valid_candidates") or 0)
            attempts = int(workflow.get("registration_attempts") or 0)
            if registered == 0 and attempts == 0:
                codes.insert(0, FailureCode.GENERATED_TEST_NOT_ATTEMPTED)
            elif registered > 0 and valid == 0:
                codes.insert(0, FailureCode.GENERATED_TEST_INVALID_CANDIDATE)
            else:
                codes.insert(0, FailureCode.GENERATED_TEST_REQUIREMENT_UNMET)
        completion_claimed = bool(features.first_completion_request or runtime_verified)
        verification_succeeded = "project_verified" in event_types or any(
            item.get("event_type") == "verification_report_created" and item.get("verdict") == "verified" for item in events
        )
        if (
            completion_claimed
            and oracle_verdict is not None
            and oracle_verdict != "verified"
            and not runtime_verified
            and oracle_integrity is not False
        ):
            codes[0:0] = [FailureCode.COMPLETION_PREMATURE_REQUEST, FailureCode.VERIFICATION_FALSE_POSITIVE]
        if features.first_protocol_error:
            codes.append(FailureCode.PROTOCOL_INVALID_FORMAT)
        if features.first_invalid_tool_call:
            codes.append(FailureCode.TOOL_INVALID_ARGUMENT)
        if {"verification_contract_mismatch", "verification_baseline_mismatch", "oracle_baseline_mismatch"} & event_types:
            codes.append(FailureCode.VERIFICATION_BASELINE_MISMATCH)
        if "verification_infrastructure_error" in event_types or oracle_verdict == "infrastructure_error":
            codes.append(FailureCode.VERIFICATION_INFRA_FAILURE)
        if oracle_event is not None and oracle_verdict not in {"verified", "infrastructure_error", "contract_invalid"}:
            if int(oracle_event.get("required_checks_failed") or 0) > 0:
                codes.append(FailureCode.VERIFICATION_REQUIREMENT_FAILED)
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
        if (
            termination_reason == TerminationReason.TASK_LIMIT
            and features.first_completion_request is None
            and generated_requirement_event is None
        ):
            codes.append(FailureCode.COMPLETION_MISSING_REQUEST)
        codes = list(dict.fromkeys(codes)) or [FailureCode.UNCLASSIFIED_FAILURE]
        primary = codes[0]
        evidence = [
            item for item in [features.first_observable_symptom, features.first_causal_divergence, features.final_outcome_event] if item
        ]
        uncertain = primary in {FailureCode.UNCLASSIFIED_FAILURE, FailureCode.KNOWLEDGE_NEGATIVE_TRANSFER}
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

    def attribute_error(self, *, case_id: str, trial_id: str, explanation: str) -> FailureAttribution:
        return FailureAttribution(
            case_id=case_id,
            trial_id=trial_id,
            termination_reason=TerminationReason.ENVIRONMENT_ERROR,
            primary_layer=FAILURE_LAYERS[FailureCode.HARNESS_ERROR],
            primary_code=FailureCode.HARNESS_ERROR,
            explanation=explanation,
            confidence=1.0,
            deterministic=True,
        )
