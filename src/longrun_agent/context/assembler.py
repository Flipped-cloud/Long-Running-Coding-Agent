from __future__ import annotations

import json

from longrun_agent.agent.prompt import SYSTEM_PROMPT
from longrun_agent.config import ContextConfig
from longrun_agent.context.schema import HandoffRecord, TaskContextSeed


class ContextAssembler:
    def __init__(self, config: ContextConfig):
        self.config = config

    def system_message(self) -> dict[str, str]:
        return {"role": "system", "content": SYSTEM_PROMPT}

    def task_anchor_message(self, seed: TaskContextSeed) -> dict[str, str]:
        return {"role": "user", "content": render_task_anchor(seed)}

    def current_instruction_message(self, seed: TaskContextSeed) -> dict[str, str]:
        return {"role": "user", "content": render_current_instruction(seed, repeat_anchor=self.config.repeat_task_anchor_at_end)}

    def handoff_message(self, handoff: HandoffRecord | None) -> dict[str, str] | None:
        if handoff is None:
            return None
        return {"role": "user", "content": render_handoff(handoff)}


def render_task_anchor(seed: TaskContextSeed) -> str:
    lines = [
        f"Project objective: {seed.project_objective}",
        f"Project ID: {seed.project_id or 'unknown'}",
        f"Plan version: {seed.plan_version}",
        f"Current task ID: {seed.task_id}",
        f"Current task title: {seed.task_title}",
        f"Task objective: {seed.task_objective}",
        f"Authoritative task status: {seed.task_status}",
        "Acceptance criteria:",
        *[f"- {criterion}" for criterion in seed.acceptance_criteria],
        "Completed dependency summary:",
        *(seed.dependency_summaries or ["- none"]),
        f"Blocker: {seed.blocker or 'none'}",
        f"Attempts: {seed.attempts}",
    ]
    if seed.progress_summary:
        lines.extend(["Recent progress summary:", seed.progress_summary])
    if seed.files_touched:
        lines.extend(["Current touched files:", *[f"- {path}" for path in seed.files_touched[-10:]]])
    return "\n".join(lines)


def render_current_instruction(seed: TaskContextSeed, *, repeat_anchor: bool) -> str:
    lines = [
        "Current instruction:",
        "- Continue by using tools; do not invent file contents, command results, or verification results.",
        "- Use request_task_completion only when the acceptance criteria are satisfied by observed evidence.",
        "- Use report_blocker when blocked and cite the remaining issue.",
        "Control tools: report_progress, report_blocker, request_task_completion, request_decomposition.",
        "Bash tool protocol:",
        '- Prefer argv, for example {"argv": ["python", "-m", "pytest", "-q"], "cwd": "."}.',
        "- Commands already run inside the workspace.",
        "- Do not use cd, &&, ||, pipes, redirection, semicolons, or shell built-ins.",
    ]
    if repeat_anchor:
        lines.extend(
            [
                "Current task:",
                seed.task_objective,
                "Critical acceptance criteria:",
                *[f"- {criterion}" for criterion in seed.acceptance_criteria[:5]],
            ]
        )
    return "\n".join(lines)


def render_handoff(handoff: HandoffRecord) -> str:
    payload = handoff.model_dump(mode="json")
    return "Structured context handoff:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
