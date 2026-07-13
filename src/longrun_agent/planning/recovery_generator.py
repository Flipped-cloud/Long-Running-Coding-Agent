from __future__ import annotations

from longrun_agent.config import BoundedSearchConfig
from longrun_agent.model.base import ModelProvider
from longrun_agent.planning.prompts import RECOVERY_GENERATOR_PROMPT
from longrun_agent.planning.protocol import RecoveryCandidate
from longrun_agent.state.schema import TaskNode

SUBMIT_RECOVERY_CANDIDATES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_recovery_candidates",
        "description": "Submit bounded recovery candidates.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "candidates": {"type": "array", "items": RecoveryCandidate.model_json_schema()},
            },
            "required": ["task_id", "candidates"],
        },
    },
}


class RecoveryCandidateGenerator:
    def __init__(self, model: ModelProvider, config: BoundedSearchConfig):
        self.model = model
        self.config = config

    def generate(self, task: TaskNode, reason: str) -> list[RecoveryCandidate]:
        messages = [
            {"role": "system", "content": RECOVERY_GENERATOR_PROMPT},
            {"role": "user", "content": f"Task {task.id}: {task.objective}\nReason: {reason}"},
        ]
        last_error: Exception | None = None
        for _ in range(self.config.max_protocol_retries):
            response = self.model.generate(messages, [SUBMIT_RECOVERY_CANDIDATES_SCHEMA])
            try:
                calls = [call for call in response.tool_calls if call.name == "submit_recovery_candidates"]
                if len(calls) != 1:
                    raise ValueError("generator must call submit_recovery_candidates exactly once")
                if calls[0].arguments.get("task_id") not in {task.id, task.key}:
                    raise ValueError("candidate task_id mismatch")
                candidates = [RecoveryCandidate.model_validate(item) for item in calls[0].arguments.get("candidates", [])]
                if not candidates:
                    raise ValueError("recovery candidates must not be empty")
                return candidates[: self.config.candidate_count]
            except Exception as exc:
                last_error = exc
                messages.append({"role": "user", "content": f"Candidate protocol error: {exc}. Try again."})
        raise ValueError(f"recovery candidate generation failed: {last_error}")
