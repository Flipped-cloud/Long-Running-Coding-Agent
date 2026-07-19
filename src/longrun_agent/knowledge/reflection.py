from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from longrun_agent.config import KnowledgeReflectionConfig
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.knowledge.schema import ExperienceEvidencePack, ReflectionCandidate, ReflectionCandidateDraft
from longrun_agent.model.base import ModelProvider

REFLECTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_reflection_candidate",
        "description": "Submit one evidence-grounded reflection candidate.",
        "parameters": ReflectionCandidateDraft.model_json_schema(),
    },
}


class ReflectionGenerator:
    def __init__(self, model: ModelProvider, config: KnowledgeReflectionConfig):
        self.model = model
        self.config = config

    def generate(self, pack: ExperienceEvidencePack, *, related_memory_ids: list[str] | None = None) -> ReflectionCandidate | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a conservative reflection candidate using only supplied evidence IDs. "
                    "Do not claim task completion. Use unknown when evidence is insufficient. "
                    "Recommended actions must be concrete and parameterized. Portable memories must be abstract: "
                    "do not include project IDs, file names, function names, absolute paths, or repository-specific symbols. "
                    "Normalization or string-stripping lessons must include anti-conditions such as leading whitespace must be "
                    "preserved, indentation is meaningful, and exact literal formatting is required. "
                    "Project-specific paths must not become portable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(_reflection_payload(pack, related_memory_ids or []), ensure_ascii=False, sort_keys=True),
            },
        ]
        tool_choice = {"type": "function", "function": {"name": "submit_reflection_candidate"}}
        for retry in range(self.config.max_protocol_retries + 1):
            try:
                response = self.model.generate(messages, [REFLECTION_TOOL_SCHEMA], tool_choice=tool_choice)
            except ToolArgumentsProtocolError:
                if retry >= self.config.max_protocol_retries:
                    return None
                messages.append({"role": "user", "content": "Call submit_reflection_candidate with valid JSON arguments."})
                continue
            calls = [call for call in response.tool_calls if call.name == "submit_reflection_candidate"]
            if len(calls) != 1:
                if retry >= self.config.max_protocol_retries:
                    return None
                messages.append({"role": "user", "content": "Call submit_reflection_candidate exactly once. Do not answer in prose."})
                continue
            try:
                draft = ReflectionCandidateDraft.model_validate(calls[0].arguments)
            except ValidationError:
                if retry >= self.config.max_protocol_retries:
                    return None
                messages.append({"role": "user", "content": "Repair the candidate so it matches the provided schema."})
                continue
            return ReflectionCandidate(**draft.model_dump(mode="json"), episode_id=pack.episode_id)
        return None


def _reflection_payload(pack: ExperienceEvidencePack, related_memory_ids: list[str]) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": item.evidence_id,
            "event_type": item.event_type,
            "tool_name": item.tool_name,
            "command": item.command,
            "path": item.path,
            "summary": item.summary,
            "success": item.success,
            "exit_code": item.exit_code,
            "output_excerpt": item.output_excerpt,
            "error_type": item.error_type,
            "artifact_path": item.artifact_path,
        }
        for item in pack.evidence_items[:20]
    ]
    return {
        "task_objective": pack.task_objective,
        "acceptance_criteria": pack.acceptance_criteria,
        "episode_result": {
            "run_status": pack.run_status,
            "terminal_signal": pack.terminal_signal,
            "no_progress": pack.no_progress,
            "progress_made": pack.progress_made,
        },
        "evidence": evidence,
        "failed_verification": pack.failed_verifications,
        "successful_verification": pack.successful_verifications,
        "repeated_actions": pack.repeated_actions,
        "blockers": pack.blockers,
        "related_memory_ids": related_memory_ids,
    }
