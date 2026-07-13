from __future__ import annotations

import json

from longrun_agent.config import InitialPlanConfig
from longrun_agent.exceptions import ConfigurationError
from longrun_agent.model.base import ModelProvider
from longrun_agent.planning.prompts import initial_planner_prompt
from longrun_agent.planning.protocol import TaskDraft
from longrun_agent.planning.validator import PlanValidator
from longrun_agent.protocol import ModelResponse
from longrun_agent.state.schema import TaskNode

SUBMIT_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit a coarse task plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_summary": {"type": "string"},
                "tasks": {"type": "array", "items": TaskDraft.model_json_schema()},
            },
            "required": ["project_summary", "tasks"],
        },
    },
}


class InitialPlanner:
    def __init__(self, model: ModelProvider, config: InitialPlanConfig, validator: PlanValidator | None = None):
        self.model = model
        self.config = config
        self.validator = validator or PlanValidator()
        self.raw_responses: list[ModelResponse] = []

    def plan(self, *, project_id: str, objective: str) -> list[TaskNode]:
        messages = [
            {"role": "system", "content": initial_planner_prompt(self.config.min_tasks, self.config.max_tasks)},
            {"role": "user", "content": objective},
        ]
        last_error: Exception | None = None
        for _ in range(self.config.max_protocol_retries):
            response = self.model.generate(messages, [SUBMIT_PLAN_SCHEMA])
            self.raw_responses.append(response)
            try:
                calls = [call for call in response.tool_calls if call.name == "submit_plan"]
                if len(calls) != 1:
                    raise ValueError("planner must call submit_plan exactly once")
                drafts = [TaskDraft.model_validate(item) for item in calls[0].arguments.get("tasks", [])]
                task_count = len(drafts)
                self.validator.validate_initial(drafts, min_tasks=self.config.min_tasks, max_tasks=self.config.max_tasks)
                return self.validator.to_task_nodes(project_id, drafts)
            except Exception as exc:
                last_error = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Planner protocol error: {exc}. "
                            f"Submitted task count: {locals().get('task_count', 'unknown')}; "
                            f"allowed range: {self.config.min_tasks}-{self.config.max_tasks}. "
                            "Call submit_plan again with a completely corrected plan. "
                            "Every item in tasks must include key, title, objective, acceptance_criteria, and depends_on_keys. "
                            "Do not emit dependency-only objects such as {'key': 'task-key'}."
                        ),
                    }
                )
        raise ValueError(f"initial planner failed after retries: {last_error}")

    def load_from_file(self, *, project_id: str) -> list[TaskNode]:
        if self.config.plan_file is None:
            raise ConfigurationError("planning.initial_plan.plan_file is required when source=file")
        try:
            raw = json.loads(self.config.plan_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"initial plan file does not exist: {self.config.plan_file}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"initial plan file is not valid JSON: {self.config.plan_file}: {exc}") from exc
        try:
            if not isinstance(raw, dict):
                raise ValueError("initial plan file must contain a JSON object")
            if not isinstance(raw.get("project_summary"), str) or not raw["project_summary"].strip():
                raise ValueError("initial plan file must contain project_summary")
            if not isinstance(raw.get("tasks"), list):
                raise ValueError("initial plan file must contain tasks list")
            drafts = [TaskDraft.model_validate(item) for item in raw.get("tasks", [])]
            self.validator.validate_initial(drafts, min_tasks=self.config.min_tasks, max_tasks=self.config.max_tasks)
            return self.validator.to_task_nodes(project_id, drafts)
        except Exception as exc:
            raise ConfigurationError(f"initial plan file failed validation: {self.config.plan_file}: {exc}") from exc
