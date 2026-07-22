from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from longrun_agent.verification.schema import TestCandidate, VerificationContract


class ControlSignalType(StrEnum):
    PROGRESS = "progress"
    BLOCKER = "blocker"
    COMPLETION_REQUEST = "completion_request"
    DECOMPOSITION_REQUEST = "decomposition_request"


class ControlSignal(BaseModel):
    type: ControlSignalType
    summary: str | None = None
    files_touched: list[str] = Field(default_factory=list)
    reason: str | None = None
    attempted_actions: list[str] = Field(default_factory=list)
    decomposition_recommended: bool = False
    acceptance_criteria_addressed: list[str] = Field(default_factory=list)


class GeneratedTestRequirementError(ValueError):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(str(payload["next_action"]))


class TaskControlChannel:
    def __init__(
        self,
        *,
        workspace: Path | None = None,
        task_id: str = "",
        session_id: str = "",
        verification_contract: VerificationContract | None = None,
        max_test_candidates: int = 0,
        require_test_candidate_before_completion: bool = False,
        minimum_registered_candidates: int = 1,
        minimum_valid_candidates: int = 1,
        max_registration_attempts: int = 3,
        candidate_validator: Callable[[TestCandidate], TestCandidate] | None = None,
    ):
        self.signals: list[ControlSignal] = []
        self.workspace = workspace
        self.task_id = task_id
        self.session_id = session_id
        self.verification_contract = verification_contract
        self.max_test_candidates = max_test_candidates
        self.require_test_candidate_before_completion = require_test_candidate_before_completion
        self.minimum_registered_candidates = minimum_registered_candidates
        self.minimum_valid_candidates = minimum_valid_candidates
        self.max_registration_attempts = max_registration_attempts
        self.candidate_validator = candidate_validator
        self.test_candidates: list[TestCandidate] = []
        self.registration_attempt_count = 0
        self.completion_request_count = 0

    @property
    def terminal_signal(self) -> ControlSignal | None:
        for signal in self.signals:
            if signal.type != ControlSignalType.PROGRESS:
                return signal
        return None

    @property
    def progress_signals(self) -> list[ControlSignal]:
        return [signal for signal in self.signals if signal.type == ControlSignalType.PROGRESS]

    def record(self, signal: ControlSignal) -> None:
        if signal.type == ControlSignalType.COMPLETION_REQUEST:
            self.completion_request_count += 1
            error = self.generated_test_requirement_error()
            if error is not None:
                raise error
        if signal.type != ControlSignalType.PROGRESS and self.terminal_signal is not None:
            raise ValueError("terminal control signal already exists")
        self.signals.append(signal)

    @property
    def valid_test_candidate_count(self) -> int:
        return sum(
            candidate.valid
            and candidate.transition is not None
            and candidate.baseline_result is not None
            and candidate.candidate_result is not None
            for candidate in self.test_candidates
        )

    def workflow_state(self) -> dict[str, int]:
        return {
            "registered_candidates": len(self.test_candidates),
            "valid_candidates": self.valid_test_candidate_count,
            "registration_attempts": self.registration_attempt_count,
            "completion_requests": self.completion_request_count,
        }

    def generated_test_requirement_error(self) -> GeneratedTestRequirementError | None:
        if not self.require_test_candidate_before_completion:
            return None
        state = self.workflow_state()
        if (
            state["registered_candidates"] >= self.minimum_registered_candidates
            and state["valid_candidates"] >= self.minimum_valid_candidates
        ):
            return None
        return GeneratedTestRequirementError(
            {
                "success": False,
                "error_type": "generated_test_requirement_unmet",
                "retryable": self.registration_attempt_count < self.max_registration_attempts,
                **state,
                "required_registered": self.minimum_registered_candidates,
                "required_valid": self.minimum_valid_candidates,
                "next_action": "Add, run, and register an issue-reproduction test.",
            }
        )

    def register_test_candidate(
        self,
        *,
        paths: list[str],
        command_argv: list[str],
        issue_behavior: str,
        expected_failure_reason: str,
    ) -> TestCandidate:
        from longrun_agent.verification.generated_tests import register_test_candidate

        if self.workspace is None or not self.task_id or not self.session_id:
            raise ValueError("test candidate registration is not configured")
        if self.registration_attempt_count >= self.max_registration_attempts:
            raise ValueError("maximum test candidate registration attempts reached")
        self.registration_attempt_count += 1
        if len(self.test_candidates) >= self.max_test_candidates:
            raise ValueError("maximum test candidates reached for task session")
        candidate = register_test_candidate(
            workspace=self.workspace,
            task_id=self.task_id,
            session_id=self.session_id,
            paths=paths,
            command_argv=command_argv,
            issue_behavior=issue_behavior,
            expected_failure_reason=expected_failure_reason,
            contract=self.verification_contract,
        )
        if self.candidate_validator is not None:
            candidate = self.candidate_validator(candidate)
        self.test_candidates.append(candidate)
        return candidate
