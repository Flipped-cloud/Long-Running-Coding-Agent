from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from longrun_agent.config import ContextConfig
from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.handoff import deterministic_handoff
from longrun_agent.context.schema import HandoffRecord, TaskContextSeed
from longrun_agent.model.base import ModelProvider

HANDOFF_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_context_handoff",
        "description": "Submit a structured context handoff based only on supplied evidence.",
        "parameters": HandoffRecord.model_json_schema(),
    },
}


class StructuredContextCompactor:
    def __init__(self, config: ContextConfig, model: ModelProvider | None = None):
        self.config = config
        self.model = model
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.protocol_error_count = 0
        self.fallback_count = 0
        self.model_call_count = 0

    def compact(
        self,
        *,
        seed: TaskContextSeed,
        buffer: ContextBuffer,
        project_id: str | None,
        session_id: str,
        source_segment_id: int,
        target_segment_id: int,
        plan_version: int,
        workspace_root: Path | None = None,
    ) -> HandoffRecord:
        if self.config.structured_handoff.use_model and self.model is not None:
            record = self._try_model_compact(
                seed=seed,
                buffer=buffer,
                project_id=project_id,
                session_id=session_id,
                source_segment_id=source_segment_id,
                target_segment_id=target_segment_id,
                plan_version=plan_version,
            )
            if record is not None:
                return record
        if not self.config.structured_handoff.fallback_deterministic:
            raise ValueError("structured handoff compaction failed and deterministic fallback is disabled")
        self.fallback_count += 1
        return deterministic_handoff(
            seed=seed,
            buffer=buffer,
            project_id=project_id,
            session_id=session_id,
            source_segment_id=source_segment_id,
            target_segment_id=target_segment_id,
            plan_version=plan_version,
            workspace_root=workspace_root,
        )

    def _try_model_compact(
        self,
        *,
        seed: TaskContextSeed,
        buffer: ContextBuffer,
        project_id: str | None,
        session_id: str,
        source_segment_id: int,
        target_segment_id: int,
        plan_version: int,
    ) -> HandoffRecord | None:
        messages = [
            {
                "role": "system",
                "content": "Create a structured context handoff using only provided evidence. Do not declare the task complete.",
            },
            {
                "role": "user",
                "content": (
                    f"project_id={project_id}; task_id={seed.task_id}; session_id={session_id}; "
                    f"source_segment_id={source_segment_id}; target_segment_id={target_segment_id}; plan_version={plan_version}; "
                    f"task_objective={seed.task_objective}; acceptance_criteria={seed.acceptance_criteria}; "
                    f"recent_messages={buffer.export_window(recent_turns=self.config.recent_full_turns)}"
                ),
            },
        ]
        for _ in range(self.config.structured_handoff.max_protocol_retries):
            try:
                self.model_call_count += 1
                response = self.model.generate(messages, [HANDOFF_TOOL_SCHEMA])
            except Exception:
                self.protocol_error_count += 1
                return None
            self.input_tokens_total += int(response.usage.get("input_tokens", 0))
            self.output_tokens_total += int(response.usage.get("output_tokens", 0))
            calls = [call for call in response.tool_calls if call.name == "submit_context_handoff"]
            if not calls:
                self.protocol_error_count += 1
                messages.append({"role": "user", "content": "Protocol error: call submit_context_handoff with valid arguments."})
                continue
            try:
                record = HandoffRecord.model_validate(calls[0].arguments)
            except ValidationError:
                self.protocol_error_count += 1
                messages.append({"role": "user", "content": "Validation error: repair the handoff and cite valid evidence ids."})
                continue
            if record.task_id != seed.task_id or record.session_id != session_id:
                self.protocol_error_count += 1
                messages.append({"role": "user", "content": "Validation error: task_id and session_id must match the current session."})
                continue
            return record
        return None
