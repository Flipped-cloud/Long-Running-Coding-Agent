from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

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


class TaskControlChannel:
    def __init__(
        self,
        *,
        workspace: Path | None = None,
        task_id: str = "",
        session_id: str = "",
        verification_contract: VerificationContract | None = None,
        max_test_candidates: int = 0,
    ):
        self.signals: list[ControlSignal] = []
        self.workspace = workspace
        self.task_id = task_id
        self.session_id = session_id
        self.verification_contract = verification_contract
        self.max_test_candidates = max_test_candidates
        self.test_candidates: list[TestCandidate] = []

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
        if signal.type != ControlSignalType.PROGRESS and self.terminal_signal is not None:
            raise ValueError("terminal control signal already exists")
        self.signals.append(signal)

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
        self.test_candidates.append(candidate)
        return candidate
