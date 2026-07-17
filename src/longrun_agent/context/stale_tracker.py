from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from longrun_agent.context.buffer import ContextBuffer, InteractionTurn
from longrun_agent.context.schema import ContextItemStatus


@dataclass
class StaleTracker:
    code_epoch: int = 0
    stale_turn_ids: set[int] = field(default_factory=set)
    superseded_turn_ids: set[int] = field(default_factory=set)

    def refresh(self, buffer: ContextBuffer) -> None:
        reads_by_path: dict[str, list[int]] = {}
        last_bash_by_epoch_command: dict[tuple[int, str], int] = {}
        successful_verification_turns: list[tuple[int, int]] = []
        for turn in buffer.all_turns():
            for result in _turn_results(turn):
                metadata = result.get("metadata") or {}
                tool = result.get("tool_name")
                if tool == "read_file" and result.get("success"):
                    path = str(metadata.get("path") or "")
                    if path:
                        reads_by_path.setdefault(path, []).append(turn.step)
                elif tool == "write_file" and result.get("success") and metadata.get("status") in {"created", "updated"}:
                    self.code_epoch += 1
                    path = str(metadata.get("path") or "")
                    if path:
                        self.stale_turn_ids.update(reads_by_path.get(path, []))
                    for verification_step, verification_epoch in successful_verification_turns:
                        if verification_epoch < self.code_epoch:
                            self.stale_turn_ids.add(verification_step)
                elif tool == "bash":
                    command = str(metadata.get("normalized_command") or metadata.get("command") or "")
                    epoch = int(metadata.get("code_epoch") or self.code_epoch)
                    if result.get("success") and metadata.get("exit_code") == 0 and metadata.get("verification_kind"):
                        successful_verification_turns.append((turn.step, epoch))
                    key = (epoch, command)
                    if command and key in last_bash_by_epoch_command:
                        self.superseded_turn_ids.add(last_bash_by_epoch_command[key])
                    if command:
                        last_bash_by_epoch_command[key] = turn.step
        for turn in buffer.all_turns():
            if turn.step in self.stale_turn_ids:
                turn.status = ContextItemStatus.STALE
            elif turn.step in self.superseded_turn_ids:
                turn.status = ContextItemStatus.SUPERSEDED


def _turn_results(turn: InteractionTurn) -> list[dict[str, Any]]:
    results = []
    for message in turn.tool_result_messages:
        try:
            results.append(json.loads(str(message.get("content") or "{}")))
        except json.JSONDecodeError:
            continue
    return results
