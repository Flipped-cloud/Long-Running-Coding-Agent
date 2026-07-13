from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


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
    def __init__(self):
        self.signals: list[ControlSignal] = []

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
