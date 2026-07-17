from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from longrun_agent.config import ContextConfig
from longrun_agent.context.budget import ContextBudgetManager
from longrun_agent.context.buffer import ContextBuffer, InteractionTurn
from longrun_agent.context.schema import ContextItemStatus, PruneReport

CONTROL_TOOLS = {"report_progress", "report_blocker", "request_task_completion", "request_decomposition"}


@dataclass
class PruneOutcome:
    messages: list[dict[str, Any]]
    report: PruneReport


class DeterministicContextPruner:
    def __init__(self, config: ContextConfig, budget: ContextBudgetManager):
        self.config = config
        self.budget = budget

    def recent_window(self, buffer: ContextBuffer, tools: list[dict[str, Any]], *, before_tokens: int) -> PruneOutcome:
        messages = buffer.export_window(recent_turns=self.config.recent_full_turns)
        after = self.budget.measure(messages, tools).estimated_total_tokens
        return PruneOutcome(
            messages=messages,
            report=PruneReport(
                estimated_tokens_before=before_tokens,
                estimated_tokens_after=after,
                tokens_removed=max(0, before_tokens - after),
                items_removed=max(0, len(buffer.export_messages()) - len(messages)),
            ),
        )

    def prune(self, buffer: ContextBuffer, tools: list[dict[str, Any]], *, before_tokens: int) -> PruneOutcome:
        recent = {id(turn) for turn in buffer.recent_turns(self.config.recent_full_turns)}
        messages = [deepcopy(buffer.system_message), deepcopy(buffer.task_anchor_message)]
        stale_count = 0
        superseded_count = 0
        compacted_count = 0
        seen_reminders: set[str] = set()
        for item in buffer.messages_only_and_turns():
            if isinstance(item, dict):
                normalized = _normalize_message(item)
                if self.config.pruning.deduplicate_reminders and normalized in seen_reminders:
                    continue
                seen_reminders.add(normalized)
                messages.append(_strip_internal(item))
                continue
            if id(item) in recent or _contains_control_tool(item):
                messages.extend(item.export_messages())
                continue
            if item.status == ContextItemStatus.STALE:
                stale_count += 1
                messages.extend(_compact_turn(item, reason="stale", max_chars=self.config.pruning.compact_output_max_chars))
                compacted_count += 1
            elif item.status == ContextItemStatus.SUPERSEDED:
                superseded_count += 1
                messages.extend(_compact_turn(item, reason="superseded", max_chars=self.config.pruning.compact_output_max_chars))
                compacted_count += 1
            elif self.config.pruning.compact_old_tool_results:
                messages.extend(_compact_turn(item, reason="old", max_chars=self.config.pruning.compact_output_max_chars))
                compacted_count += 1
            else:
                messages.extend(item.export_messages())
        after = self.budget.measure(messages, tools).estimated_total_tokens
        return PruneOutcome(
            messages=messages,
            report=PruneReport(
                estimated_tokens_before=before_tokens,
                estimated_tokens_after=after,
                tokens_removed=max(0, before_tokens - after),
                stale_items_removed=stale_count,
                superseded_items_removed=superseded_count,
                compacted_items=compacted_count,
                items_removed=stale_count + superseded_count + compacted_count,
            ),
        )


def _compact_turn(turn: InteractionTurn, *, reason: str, max_chars: int) -> list[dict[str, Any]]:
    messages = [deepcopy(turn.assistant_message)]
    for message in turn.tool_result_messages:
        compact = deepcopy(message)
        result = _tool_result_dict(message)
        metadata = result.get("metadata") or {}
        compact_result = dict(result)
        compact_result["output"] = _compact_output(result, reason=reason, max_chars=max_chars)
        compact_result["summary"] = f"{result.get('summary', 'tool result')} [{reason}; compacted]"
        compact_result["metadata"] = {**metadata, "context_status": reason, "original_step": turn.step}
        compact["content"] = json.dumps(compact_result, ensure_ascii=False)
        messages.append(compact)
    return [_strip_internal(message) for message in messages]


def _compact_output(result: dict[str, Any], *, reason: str, max_chars: int) -> str:
    metadata = result.get("metadata") or {}
    tool = result.get("tool_name")
    if tool == "bash":
        return "\n".join(
            [
                "Tool: bash",
                f"Command: {metadata.get('normalized_command') or metadata.get('command') or ''}",
                f"Exit code: {metadata.get('exit_code')}",
                f"Summary: {result.get('summary', '')}",
                f"Artifact: {result.get('artifact_path') or metadata.get('output_artifact') or ''}",
                f"Code epoch: {metadata.get('code_epoch')}",
                f"Status: {reason}",
            ]
        )
    if tool == "read_file":
        return "\n".join(
            [
                "Tool: read_file",
                f"Path: {metadata.get('path')}",
                f"Lines: {metadata.get('start_line')}-{metadata.get('end_line')}",
                f"SHA256: {metadata.get('content_sha256')}",
                f"Status: {reason}",
                "Full original observation: event/artifact reference",
            ]
        )
    output = str(result.get("output") or "")
    return output if len(output) <= max_chars else output[: max_chars // 2] + "\n...[context compacted]...\n" + output[-max_chars // 2 :]


def _tool_result_dict(message: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(message.get("content") or "{}"))
    except json.JSONDecodeError:
        return {
            "tool_name": message.get("name"),
            "success": True,
            "summary": "raw tool message",
            "output": str(message.get("content") or ""),
        }


def _contains_control_tool(turn: InteractionTurn) -> bool:
    for call in turn.assistant_message.get("tool_calls") or []:
        name = (call.get("function") or {}).get("name")
        if name in CONTROL_TOOLS:
            return True
    return False


def _normalize_message(message: dict[str, Any]) -> str:
    return " ".join(str(message.get("content") or "").split()).lower()


def _strip_internal(message: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(message)
    for key in list(clean):
        if key.startswith("_context_"):
            clean.pop(key, None)
    return clean
