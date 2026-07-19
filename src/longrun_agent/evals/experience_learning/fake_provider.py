from __future__ import annotations

import re

from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCase
from longrun_agent.knowledge.schema import MemoryScope
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall


class ProtocolAwareExperienceFakeProvider(ModelProvider):
    def __init__(self, case: ExperienceLearningCase):
        self.case = case
        self.calls = 0
        self.session_step = 0

    def generate(self, messages: list[dict], tools: list[dict], tool_choice: dict | str | None = None) -> ModelResponse:
        self.calls += 1
        tool_names = {_tool_name(tool) for tool in tools}
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        if "submit_plan" in tool_names:
            return _response("plan", "submit_plan", _plan_arguments(self.case))
        if _forced_tool(tool_choice) == "submit_reflection_candidate":
            return _response("reflection", "submit_reflection_candidate", _reflection_arguments(prompt))
        if _forced_tool(tool_choice) == "submit_skill_candidate":
            return _response("skill", "submit_skill_candidate", _skill_arguments(prompt))
        return self._session_response(prompt)

    def _session_response(self, prompt: str) -> ModelResponse:
        self.session_step += 1
        role = self.case.role
        if role == "learning_probe":
            sequence = [
                ("read_file", {"path": "task_app.py"}),
                ("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
                (
                    "report_blocker",
                    {
                        "reason": "Whitespace-only task names are accepted; pytest shows the validation failure.",
                        "attempted_actions": ["read task_app.py", "python -m pytest -q"],
                        "decomposition_recommended": False,
                    },
                ),
            ]
            return _sequence_response(sequence, self.session_step)
        if role == "same_repository_reuse":
            memory_id = _first_id(prompt, "MEM")
            prefix = (
                [
                    (
                        "report_knowledge_use",
                        {"memory_ids": [memory_id], "skill_ids": [], "reason": "Applied the exposed whitespace validation memory."},
                    )
                ]
                if memory_id
                else []
            )
            sequence = [
                *prefix,
                ("read_file", {"path": "task_app.py"}),
                (
                    "write_file",
                    {
                        "path": "task_app.py",
                        "content": "def validate_task_name(name: str) -> bool:\n    return bool(name and name.strip())\n",
                    },
                ),
                ("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
                (
                    "request_task_completion",
                    {
                        "summary": "Whitespace-only task names are rejected and pytest passes.",
                        "acceptance_criteria_addressed": ["pytest passes", "whitespace-only task names are rejected"],
                    },
                ),
            ]
            return _sequence_response(sequence, self.session_step)
        if role == "positive_transfer":
            skill_id = _first_id(prompt, "SKILL")
            memory_id = _first_id(prompt, "MEM")
            prefix = (
                [
                    (
                        "report_knowledge_use",
                        {"memory_ids": [], "skill_ids": [skill_id], "reason": "Applied the exposed portable validation skill."},
                    )
                ]
                if skill_id
                else [
                    (
                        "report_knowledge_use",
                        {"memory_ids": [memory_id], "skill_ids": [], "reason": "Applied the exposed whitespace validation memory."},
                    )
                ]
                if memory_id
                else []
            )
            sequence = [
                *prefix,
                ("read_file", {"path": "command_app.py"}),
                (
                    "write_file",
                    {
                        "path": "command_app.py",
                        "content": (
                            "def normalize_command(command: str) -> str:\n"
                            "    return command.strip()\n\n\n"
                            "def is_valid_command(command: str) -> bool:\n"
                            "    return bool(normalize_command(command))\n"
                        ),
                    },
                ),
                ("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
                (
                    "request_task_completion",
                    {
                        "summary": "Command whitespace validation is fixed and pytest passes.",
                        "acceptance_criteria_addressed": ["pytest passes", "whitespace-only commands are rejected"],
                    },
                ),
            ]
            return _sequence_response(sequence, self.session_step)
        knowledge_present = _first_id(prompt, "MEM") or _first_id(prompt, "SKILL")
        prefix = (
            [
                (
                    "report_knowledge_use",
                    {
                        "memory_ids": [],
                        "skill_ids": [],
                        "reason": "Reviewed retrieved whitespace-normalization knowledge but rejected it because this task preserves literal formatting.",
                    },
                )
            ]
            if knowledge_present
            else []
        )
        sequence = [
            *prefix,
            ("read_file", {"path": "format_app.py"}),
            ("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
            (
                "request_task_completion",
                {
                    "summary": "Formatting preservation already satisfies the tests.",
                    "acceptance_criteria_addressed": ["pytest passes", "leading whitespace is preserved"],
                },
            ),
        ]
        return _sequence_response(sequence, self.session_step)


METAMORPHIC_SKILL_TITLES = [
    "Whitespace validation repair",
    "Blank textual input guard",
    "Trim-before-emptiness check",
    "Normalized command validation",
    "Reject space-only values",
]

METAMORPHIC_TRANSFER_TASKS = [
    "reject whitespace-only commands",
    "disallow commands containing only spacing",
    "normalize command boundaries before validity checking",
]


class MetamorphicExperienceFakeProvider(ProtocolAwareExperienceFakeProvider):
    def __init__(self, case: ExperienceLearningCase, *, variant: int):
        super().__init__(case)
        self.variant = variant % len(METAMORPHIC_SKILL_TITLES)

    def generate(self, messages: list[dict], tools: list[dict], tool_choice: dict | str | None = None) -> ModelResponse:
        if _forced_tool(tool_choice) != "submit_skill_candidate":
            return super().generate(messages, tools, tool_choice)
        self.calls += 1
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        arguments = _skill_arguments(prompt)
        arguments["title"] = METAMORPHIC_SKILL_TITLES[self.variant]
        arguments["summary"] = [
            "Strip textual input before evaluating emptiness.",
            "Guard blank user text after boundary normalization.",
            "Apply trimming before an empty-value predicate.",
            "Normalize command text before validating it.",
            "Reject values that contain spacing and no content.",
        ][self.variant]
        return _response(f"skill-{self.variant}", "submit_skill_candidate", arguments)


def _tool_name(tool: dict) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _forced_tool(tool_choice) -> str | None:
    if isinstance(tool_choice, dict):
        return tool_choice.get("function", {}).get("name")
    return None


def _response(call_id: str, name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)], usage={"input_tokens": 1, "output_tokens": 1})


def _sequence_response(sequence: list[tuple[str, dict]], step: int) -> ModelResponse:
    index = min(step - 1, len(sequence) - 1)
    name, arguments = sequence[index]
    return _response(f"{name}-{step}", name, arguments)


def _plan_arguments(case: ExperienceLearningCase) -> dict:
    return {
        "project_summary": f"Experience learning case {case.case_id}",
        "tasks": [
            {
                "key": "main",
                "title": case.case_id,
                "objective": case.task,
                "acceptance_criteria": ["Run the requested verification command and report the correct terminal signal."],
                "depends_on_keys": [],
            }
        ],
    }


def _reflection_arguments(prompt: str) -> dict:
    evidence_ids = _evidence_ids(prompt)
    return {
        "failure_category": "implementation_error",
        "root_cause": "Blank-only input survives the truthiness check because normalization is omitted.",
        "incorrect_assumptions": ["Non-empty strings were assumed valid without trimming whitespace."],
        "recommended_actions": ["Call strip() before checking whether the input is empty."],
        "avoid_actions": ["Do not treat whitespace-only input as valid."],
        "applicability_conditions": ["Validation accepts textual names or commands and pytest reports whitespace-only input failure."],
        "anti_conditions": [
            "preserve leading whitespace",
            "indentation is semantically meaningful",
            "exact formatting must be retained",
        ],
        "proposed_scope": MemoryScope.PORTABLE.value,
        "tags": ["pytest", "python", "whitespace", "validation"],
        "evidence_ids": evidence_ids[:2],
        "confidence": 0.86,
    }


def _skill_arguments(prompt: str) -> dict:
    evidence_ids = _evidence_ids(prompt)
    memory_ids = sorted(set(re.findall(r"\bMEM-[A-Za-z0-9]+\b", prompt)))
    return {
        "title": "Whitespace validation repair",
        "summary": "Repair validators that must reject whitespace-only textual input by stripping before boolean validation.",
        "scope": MemoryScope.PORTABLE.value,
        "preconditions": [
            "A validator accepts textual names or commands.",
            "Tests require whitespace-only input to be rejected.",
            "The repository uses pytest for verification.",
        ],
        "anti_conditions": [
            "preserve leading whitespace",
            "indentation is semantically meaningful",
            "exact formatting must be retained",
        ],
        "parameters": ["target validation function", "focused pytest command"],
        "procedure": [
            {"description": "Inspect the relevant validation function.", "tool_name": "read_file"},
            {"description": "Apply a minimal strip-before-bool validation change.", "tool_name": "write_file"},
            {"description": "Run pytest verification.", "tool_name": "bash", "command": "python -m pytest -q"},
        ],
        "verification": ["python -m pytest -q"],
        "stop_conditions": ["Stop if formatting preservation is required instead of normalization."],
        "tags": ["pytest", "python", "whitespace", "validation"],
        "evidence_ids": evidence_ids[:3],
        "source_memory_ids": memory_ids,
        "confidence": 0.86,
    }


def _first_id(prompt: str, prefix: str) -> str | None:
    match = re.search(rf"\b{prefix}-[A-Za-z0-9]+\b", prompt)
    return match.group(0) if match else None


def _evidence_ids(prompt: str) -> list[str]:
    ids = re.findall(r'"evidence_id":\s*"([^"]+)"', prompt)
    return ids or re.findall(r"\bs\d+:e\d+\b", prompt)
