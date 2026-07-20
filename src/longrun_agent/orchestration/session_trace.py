from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from longrun_agent.protocol import RunResult, ToolCall, ToolResult
from longrun_agent.tools.bash import BashArgs


def _append_unique(items: list[str], value: str | None) -> None:
    if value and value not in items:
        items.append(value)


SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "authorization", "bearer ")


@dataclass
class BashObservation:
    command: str
    argv: list[str]
    exit_code: int | None
    success: bool
    is_verification: bool
    output_excerpt: str
    error_type: str | None
    artifact_path: str | None
    operation_index: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argv": self.argv,
            "exit_code": self.exit_code,
            "success": self.success,
            "is_verification": self.is_verification,
            "output_excerpt": self.output_excerpt,
            "error_type": self.error_type,
            "artifact_path": self.artifact_path,
            "operation_index": self.operation_index,
        }


@dataclass
class SessionTrace:
    read_files: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)
    bash_exit_codes: list[int] = field(default_factory=list)
    bash_observations: list[BashObservation] = field(default_factory=list)
    successful_test_commands: list[str] = field(default_factory=list)
    successful_acceptance_commands: list[str] = field(default_factory=list)
    repeated_tool_calls: list[str] = field(default_factory=list)
    suppressed_tool_calls: list[str] = field(default_factory=list)
    policy_gate_count: int = 0
    unsupported_shell_syntax_count: int = 0
    tool_argument_protocol_retry_count: int = 0
    last_tool_summary: str | None = None
    last_bash_summary: str | None = None
    read_only_streak: int = 0
    action_required_message: str | None = None
    _last_call_key: str | None = None
    _seen_calls: Counter[str] = field(default_factory=Counter)
    _op_index: int = 0
    _last_write_index: int = 0
    _last_successful_verification_index: int = 0

    def call_key(self, call: ToolCall) -> str:
        arguments = call.arguments
        if call.name == "bash":
            try:
                arguments = BashArgs.model_validate(arguments).model_dump(exclude_none=True)
            except (TypeError, ValueError):
                pass
        return f"{call.name}:{json.dumps(arguments, sort_keys=True, default=_unsupported_argument)}"

    def should_suppress(self, call: ToolCall) -> bool:
        return self._last_call_key == self.call_key(call)

    def record_suppressed(self, call: ToolCall) -> None:
        key = self.call_key(call)
        _append_unique(self.repeated_tool_calls, key)
        _append_unique(self.suppressed_tool_calls, key)
        self.last_tool_summary = "repeated tool call suppressed; previous result is already available"
        self._last_call_key = key

    def record_policy_gate(self, result: ToolResult) -> None:
        self.policy_gate_count += 1
        self.last_tool_summary = result.summary

    def record(self, call: ToolCall, result: ToolResult) -> None:
        self._op_index += 1
        key = self.call_key(call)
        self._seen_calls[key] += 1
        if self._seen_calls[key] == 2:
            self.repeated_tool_calls.append(key)
        self.last_tool_summary = result.summary
        self._last_call_key = key
        if call.name == "read_file" and result.success:
            _append_unique(self.read_files, str(result.metadata.get("path") or call.arguments.get("path") or ""))
            self._record_read_only_success()
        elif call.name == "write_file" and result.success:
            path = str(result.metadata.get("path") or call.arguments.get("path") or "")
            _append_unique(self.written_files, path)
            if result.metadata.get("status") in {"created", "updated"}:
                _append_unique(self.changed_files, path)
                self._last_write_index = self._op_index
                self._reset_read_only_streak()
        elif call.name == "bash":
            command = str(result.metadata.get("command") or "")
            _append_unique(self.bash_commands, command)
            argv = result.metadata.get("argv") or []
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                argv = []
            if result.metadata.get("unsupported_shell_syntax"):
                self.unsupported_shell_syntax_count += 1
                self.action_required_message = (
                    "unsupported_shell_syntax: retry the same intended command once using argv and no cd, pipes, "
                    "redirection, &&, ||, or semicolons. Example: "
                    '{"argv": ["python", "-m", "pytest", "-q"], "cwd": "."}.'
                )
            exit_code = result.metadata.get("exit_code")
            if isinstance(exit_code, int):
                self.bash_exit_codes.append(exit_code)
                if exit_code == 0 and _is_verification_command(command):
                    self._last_successful_verification_index = self._op_index
                    _append_unique(self.successful_acceptance_commands, command)
                if exit_code == 0 and "pytest" in command:
                    _append_unique(self.successful_test_commands, command)
            self.bash_observations.append(
                BashObservation(
                    command=command,
                    argv=[str(item) for item in argv],
                    exit_code=exit_code if isinstance(exit_code, int) else None,
                    success=result.success,
                    is_verification=_is_verification_command(command),
                    output_excerpt=_safe_excerpt(
                        result.output,
                        artifact_path=str(result.artifact_path or result.metadata.get("output_artifact") or ""),
                    ),
                    error_type=result.error_type.value if result.error_type else None,
                    artifact_path=_sanitize_artifact_path(str(result.artifact_path or result.metadata.get("output_artifact") or "")),
                    operation_index=self._op_index,
                )
            )
            self.last_bash_summary = result.summary
            if result.success and _is_read_only_bash(command):
                self._record_read_only_success()
            elif result.success:
                self._reset_read_only_streak()
        elif call.name in {"report_blocker", "request_task_completion", "request_decomposition", "report_progress"} and result.success:
            self._reset_read_only_streak()

    def _record_read_only_success(self) -> None:
        self.read_only_streak += 1
        if self.read_only_streak >= 3:
            self.action_required_message = (
                "action_required: You have made three consecutive read-only successful tool calls. "
                "Next call must be write_file, bash running tests, report_blocker, or request_task_completion. "
                "If information is insufficient, report_blocker instead of blindly editing."
            )

    def _reset_read_only_streak(self) -> None:
        self.read_only_streak = 0
        self.action_required_message = None

    def no_progress(self, *, progress_count: int, terminal_signal: object | None) -> bool:
        return (
            not self.changed_files
            and not self.successful_test_commands
            and progress_count == 0
            and terminal_signal is None
            and self.policy_gate_count == 0
        )

    def has_completion_evidence(self, *, existing_changed_files: list[str] | None = None) -> bool:
        changed = bool(self.changed_files or existing_changed_files)
        verified = bool(self.successful_test_commands or self.successful_acceptance_commands)
        return changed and verified and self._last_successful_verification_index >= self._last_write_index

    def record_protocol_retry(self) -> None:
        self.tool_argument_protocol_retry_count += 1

    def handoff_summary(self, result: RunResult) -> str:
        completed_work = [f"modified {path}" for path in self.changed_files] or ["none recorded"]
        passed = [
            *self.successful_test_commands,
            *[cmd for cmd in self.successful_acceptance_commands if cmd not in self.successful_test_commands],
        ]
        remaining = "Send a terminal control signal if the acceptance criteria are already satisfied."
        if not passed:
            remaining = "Run the focused verification command for the remaining acceptance criteria."
        next_actions = [
            "Review the completed work and passed verification listed above.",
            "Verify only any still-unchecked acceptance criterion.",
            "Call request_task_completion if satisfied; otherwise call report_blocker with the exact remaining issue.",
        ]
        return "\n".join(
            [
                "Completed work:",
                *[f"- {item}" for item in completed_work],
                "Passed verification:",
                *([f"- {command}" for command in passed] or ["- none recorded"]),
                "Remaining acceptance criteria:",
                f"- {remaining}",
                "Next required action:",
                *[f"- {action}" for action in next_actions[:3]],
                "Do not repeat:",
                *([f"- read {path}" for path in self.read_files[-5:]] or ["- no prior reads recorded"]),
                *([f"- {command}" for command in passed[-3:]] or ["- no passed commands recorded"]),
            ]
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "read_files": self.read_files,
            "written_files": self.written_files,
            "changed_files": self.changed_files,
            "bash_commands": self.bash_commands,
            "bash_exit_codes": self.bash_exit_codes,
            "bash_observations": [item.model_dump() for item in self.bash_observations],
            "successful_test_commands": self.successful_test_commands,
            "successful_acceptance_commands": self.successful_acceptance_commands,
            "repeated_tool_calls": self.repeated_tool_calls,
            "suppressed_tool_calls": self.suppressed_tool_calls,
            "policy_gate_count": self.policy_gate_count,
            "unsupported_shell_syntax_count": self.unsupported_shell_syntax_count,
            "tool_argument_protocol_retry_count": self.tool_argument_protocol_retry_count,
            "last_tool_summary": self.last_tool_summary,
        }


def _is_read_only_bash(command: str) -> bool:
    first = command.strip().split(maxsplit=1)[0].lower() if command.strip() else ""
    return first in {"ls", "dir", "find", "grep", "rg", "cat", "type", "pwd", "python"} and not _is_verification_command(command)


def _unsupported_argument(value: Any) -> dict[str, str]:
    return {"unsupported_type": type(value).__name__}


def _is_verification_command(command: str) -> bool:
    lowered = command.lower()
    return "pytest" in lowered or "task_service.cli" in lowered or "validate" in lowered


def _safe_excerpt(output: str, *, workspace: object | None = None, artifact_path: str = "", limit: int = 4000) -> str:
    text = output or ""
    if not text and artifact_path:
        try:
            text = Path(artifact_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
    for raw_path in [workspace, Path(artifact_path).parent if artifact_path else None]:
        if raw_path:
            text = text.replace(str(raw_path), "<workspace>")
    text = re.sub(r"[A-Za-z]:\\[^\s:]+", "<path>", text)
    text = re.sub(r"/(?:Users|home|tmp|var)/[^\s:]+", "<path>", text)
    text = _redact_secret_lines(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _redact_secret_lines(text: str) -> str:
    redacted: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            redacted.append("[redacted credential line]")
        else:
            redacted.append(line)
    return "\n".join(redacted)


def _sanitize_artifact_path(path: str) -> str | None:
    if not path:
        return None
    return Path(path).name
