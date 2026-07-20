from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from longrun_agent.evaluation.schema import AdapterVerificationResult, EvaluationOutcome, EvaluationTaskCase, TrialDescriptor


class TaskAdapter(Protocol):
    def prepare(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None: ...

    def reset(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None: ...

    def objective(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> str: ...

    def workspace(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> Path: ...

    def verification_contract(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> Path | None: ...

    def run_agent(self, case: EvaluationTaskCase, config_path: Path, seed: int, descriptor: TrialDescriptor) -> Any: ...

    def verify(
        self,
        case: EvaluationTaskCase,
        outcome: Any,
        descriptor: TrialDescriptor,
    ) -> AdapterVerificationResult: ...

    def collect_artifacts(
        self,
        case: EvaluationTaskCase,
        outcome: Any,
        verification: AdapterVerificationResult,
        descriptor: TrialDescriptor,
    ) -> EvaluationOutcome: ...

    def cleanup(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None: ...
