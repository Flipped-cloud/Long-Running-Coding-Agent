from __future__ import annotations

from longrun_agent.config import BoundedSearchConfig, DecompositionConfig
from longrun_agent.model.base import ModelProvider
from longrun_agent.planning.prompts import RECOVERY_EVALUATOR_PROMPT
from longrun_agent.planning.protocol import RecoveryCandidate, RecoverySelection
from longrun_agent.planning.validator import PlanValidator
from longrun_agent.state.schema import TaskNode

SELECT_RECOVERY_CANDIDATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "select_recovery_candidate",
        "description": "Select one recovery candidate.",
        "parameters": RecoverySelection.model_json_schema(),
    },
}


class RecoveryCandidateEvaluator:
    def __init__(self, model: ModelProvider, config: BoundedSearchConfig, decomposition_config: DecompositionConfig):
        self.model = model
        self.config = config
        self.decomposition_config = decomposition_config
        self.validator = PlanValidator()
        self.rejections: dict[str, str] = {}

    def filter_candidates(self, task: TaskNode, candidates: list[RecoveryCandidate]) -> list[RecoveryCandidate]:
        valid: list[RecoveryCandidate] = []
        self.rejections = {}
        for candidate in candidates:
            try:
                if candidate.kind == "decompose":
                    self.validator.validate_children(
                        task,
                        candidate.child_tasks,
                        min_children=self.decomposition_config.min_children,
                        max_children=self.decomposition_config.max_children,
                        max_depth=self.decomposition_config.max_depth,
                    )
                elif candidate.kind == "retry_with_guidance" and not candidate.description.strip():
                    raise ValueError("retry candidate needs new guidance")
                valid.append(candidate)
            except Exception as exc:
                self.rejections[candidate.id] = str(exc)
        return valid

    def select(self, candidates: list[RecoveryCandidate]) -> RecoverySelection:
        valid_ids = {candidate.id for candidate in candidates}
        if not valid_ids:
            raise ValueError("no valid recovery candidates")
        messages = [{"role": "system", "content": RECOVERY_EVALUATOR_PROMPT}, {"role": "user", "content": "\n".join(valid_ids)}]
        last_error: Exception | None = None
        for _ in range(self.config.max_protocol_retries):
            response = self.model.generate(messages, [SELECT_RECOVERY_CANDIDATE_SCHEMA])
            try:
                calls = [call for call in response.tool_calls if call.name == "select_recovery_candidate"]
                if len(calls) != 1:
                    raise ValueError("evaluator must call select_recovery_candidate exactly once")
                selection = RecoverySelection.model_validate(calls[0].arguments)
                if selection.selected_candidate_id not in valid_ids:
                    raise ValueError("evaluator selected a rejected or unknown candidate")
                return selection
            except Exception as exc:
                last_error = exc
                messages.append({"role": "user", "content": f"Selection protocol error: {exc}. Try again."})
        raise ValueError(f"recovery candidate evaluation failed: {last_error}")
