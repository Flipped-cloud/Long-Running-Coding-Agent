from __future__ import annotations

from longrun_agent.config import DecompositionConfig
from longrun_agent.model.base import ModelProvider
from longrun_agent.planning.prompts import DECOMPOSER_PROMPT
from longrun_agent.planning.protocol import ChildTaskDraft
from longrun_agent.planning.validator import PlanValidator
from longrun_agent.state.schema import TaskNode

SUBMIT_DECOMPOSITION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_decomposition",
        "description": "Submit child tasks for a parent task.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_task_id": {"type": "string"},
                "reason": {"type": "string"},
                "children": {"type": "array", "items": ChildTaskDraft.model_json_schema()},
            },
            "required": ["parent_task_id", "reason", "children"],
        },
    },
}


class AsNeededDecomposer:
    def __init__(self, model: ModelProvider, config: DecompositionConfig, validator: PlanValidator | None = None):
        self.model = model
        self.config = config
        self.validator = validator or PlanValidator()

    def decompose(self, parent: TaskNode, reason: str) -> list[TaskNode]:
        messages = [
            {"role": "system", "content": DECOMPOSER_PROMPT},
            {"role": "user", "content": f"Parent task {parent.id}: {parent.objective}\nReason: {reason}"},
        ]
        last_error: Exception | None = None
        for _ in range(self.config.max_protocol_retries):
            response = self.model.generate(messages, [SUBMIT_DECOMPOSITION_SCHEMA])
            try:
                calls = [call for call in response.tool_calls if call.name == "submit_decomposition"]
                if len(calls) != 1:
                    raise ValueError("decomposer must call submit_decomposition exactly once")
                if calls[0].arguments.get("parent_task_id") not in {parent.id, parent.key}:
                    raise ValueError("decomposition parent_task_id mismatch")
                drafts = [ChildTaskDraft.model_validate(item) for item in calls[0].arguments.get("children", [])]
                self.validator.validate_children(
                    parent,
                    drafts,
                    min_children=self.config.min_children,
                    max_children=self.config.max_children,
                    max_depth=self.config.max_depth,
                )
                return self.validator.children_to_task_nodes(parent, drafts)
            except Exception as exc:
                last_error = exc
                messages.append({"role": "user", "content": f"Decomposition protocol error: {exc}. Try again."})
        raise ValueError(f"decomposer failed after retries: {last_error}")
