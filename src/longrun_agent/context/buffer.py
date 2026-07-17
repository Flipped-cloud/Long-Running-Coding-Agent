from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from longrun_agent.context.schema import ContextItemStatus


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clone_message(message: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(message)


@dataclass
class InteractionTurn:
    assistant_message: dict[str, Any]
    tool_call_ids: list[str]
    step: int
    created_at: str = field(default_factory=_now)
    status: ContextItemStatus = ContextItemStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_result_messages: list[dict[str, Any]] = field(default_factory=list)

    def add_tool_result(self, message: dict[str, Any]) -> None:
        call_id = str(message.get("tool_call_id") or "")
        if call_id not in self.tool_call_ids:
            raise ValueError(f"tool result does not match active assistant tool call: {call_id}")
        if any(existing.get("tool_call_id") == call_id for existing in self.tool_result_messages):
            raise ValueError(f"duplicate tool result for tool_call_id: {call_id}")
        self.tool_result_messages.append(_clone_message(message))

    @property
    def complete(self) -> bool:
        returned_ids = {str(message.get("tool_call_id") or "") for message in self.tool_result_messages}
        return set(self.tool_call_ids) == returned_ids

    def export_messages(self) -> list[dict[str, Any]]:
        if not self.complete:
            missing = sorted(set(self.tool_call_ids) - {str(message.get("tool_call_id") or "") for message in self.tool_result_messages})
            raise ValueError(f"incomplete tool interaction turn; missing tool results: {', '.join(missing)}")
        return [_clone_message(self.assistant_message), *[_clone_message(message) for message in self.tool_result_messages]]


class ContextBuffer:
    """Context buffer with native tool-call pairing invariants."""

    def __init__(self, *, system_message: dict[str, Any], task_anchor_message: dict[str, Any]):
        self.system_message = _clone_message(system_message)
        self.task_anchor_message = _clone_message(task_anchor_message)
        self._items: list[dict[str, Any] | InteractionTurn] = []
        self._active_turn: InteractionTurn | None = None
        self.context_segment_id = 1
        self.context_reset_count = 0

    @classmethod
    def from_task(cls, *, system_prompt: str, task: str) -> ContextBuffer:
        return cls(system_message={"role": "system", "content": system_prompt}, task_anchor_message={"role": "user", "content": task})

    def add_assistant_tool_turn(self, assistant_message: dict[str, Any], *, step: int, metadata: dict[str, Any] | None = None) -> None:
        self._ensure_no_active_turn()
        tool_calls = assistant_message.get("tool_calls") or []
        tool_call_ids = [str(call.get("id") or "") for call in tool_calls]
        if not tool_call_ids or any(not call_id for call_id in tool_call_ids):
            raise ValueError("assistant tool turn requires non-empty tool_call ids")
        self._active_turn = InteractionTurn(
            assistant_message=_clone_message(assistant_message),
            tool_call_ids=tool_call_ids,
            step=step,
            metadata=dict(metadata or {}),
        )

    def add_tool_result(self, message: dict[str, Any]) -> None:
        if self._active_turn is None:
            raise ValueError("cannot add tool result without an active assistant tool turn")
        self._active_turn.add_tool_result(message)

    def finalize_turn(self) -> None:
        if self._active_turn is None:
            raise ValueError("cannot finalize without an active assistant tool turn")
        if not self._active_turn.complete:
            missing = sorted(
                set(self._active_turn.tool_call_ids)
                - {str(message.get("tool_call_id") or "") for message in self._active_turn.tool_result_messages}
            )
            raise ValueError(f"cannot finalize incomplete tool turn; missing tool results: {', '.join(missing)}")
        self._items.append(self._active_turn)
        self._active_turn = None

    def add_user_reminder(self, content: str) -> None:
        self._ensure_no_active_turn()
        self._items.append({"role": "user", "content": content, "_context_kind": "user_reminder"})

    def add_protocol_correction(self, content: str) -> None:
        self._ensure_no_active_turn()
        self._items.append({"role": "user", "content": content, "_context_kind": "protocol_correction"})

    def add_assistant_final(self, message: dict[str, Any]) -> None:
        self._ensure_no_active_turn()
        self._items.append(_clone_message(message))

    def export_messages(self) -> list[dict[str, Any]]:
        return self.export_window()

    def export_window(self, *, recent_turns: int | None = None, include_messages: bool = True) -> list[dict[str, Any]]:
        self._ensure_no_active_turn()
        messages = [_clone_message(self.system_message), _clone_message(self.task_anchor_message)]
        items = self._window_items(recent_turns) if recent_turns is not None else list(self._items)
        for item in items:
            if isinstance(item, InteractionTurn):
                messages.extend(item.export_messages())
            elif include_messages:
                messages.append(_clone_message(item))
        return [_strip_context_fields(message) for message in messages]

    def recent_turns(self, count: int) -> list[InteractionTurn]:
        if count <= 0:
            return []
        turns = [item for item in self._items if isinstance(item, InteractionTurn)]
        return turns[-count:]

    def all_turns(self) -> list[InteractionTurn]:
        return [item for item in self._items if isinstance(item, InteractionTurn)]

    def messages_only(self) -> list[dict[str, Any]]:
        return [_clone_message(item) for item in self._items if isinstance(item, dict)]

    def messages_only_and_turns(self) -> list[dict[str, Any] | InteractionTurn]:
        return list(self._items)

    def replace_items(self, items: list[dict[str, Any] | InteractionTurn]) -> None:
        self._ensure_no_active_turn()
        self._items = items

    def reset_to(
        self, *, task_anchor_message: dict[str, Any], handoff_message: dict[str, Any] | None, instruction_message: dict[str, Any]
    ) -> int:
        self._ensure_no_active_turn()
        self.task_anchor_message = _clone_message(task_anchor_message)
        self._items = []
        if handoff_message is not None:
            self._items.append(_clone_message(handoff_message))
        self._items.append(_clone_message(instruction_message))
        return self.start_new_segment()

    def start_new_segment(self) -> int:
        self._ensure_no_active_turn()
        self.context_segment_id += 1
        self.context_reset_count += 1
        return self.context_segment_id

    def snapshot(self) -> dict[str, Any]:
        turns = [item for item in self._items if isinstance(item, InteractionTurn)]
        return {
            "context_segment_id": self.context_segment_id,
            "context_reset_count": self.context_reset_count,
            "message_count": len(self.export_messages()),
            "interaction_turn_count": len(turns),
            "active_turn": self._active_turn is not None,
        }

    def _window_items(self, recent_turns: int) -> list[dict[str, Any] | InteractionTurn]:
        if recent_turns <= 0:
            return [item for item in self._items if isinstance(item, dict)]
        turn_indexes = [index for index, item in enumerate(self._items) if isinstance(item, InteractionTurn)]
        if len(turn_indexes) <= recent_turns:
            return list(self._items)
        keep_from = turn_indexes[-recent_turns]
        return [item for index, item in enumerate(self._items) if index >= keep_from or isinstance(item, dict)]

    def _ensure_no_active_turn(self) -> None:
        if self._active_turn is not None:
            raise ValueError("active tool interaction turn must be finalized before modifying context")


def _strip_context_fields(message: dict[str, Any]) -> dict[str, Any]:
    clean = _clone_message(message)
    for key in list(clean):
        if key.startswith("_context_"):
            clean.pop(key, None)
    return clean
