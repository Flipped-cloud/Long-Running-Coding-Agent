from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time

from pydantic import BaseModel, Field, model_validator

from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.path_guard import ensure_workspace_root, is_inside_path


class BashArgs(BaseModel):
    command: str | None = Field(default=None, min_length=1)
    argv: list[str] | None = None
    cwd: str = "."
    timeout: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_command_or_argv(self) -> BashArgs:
        if self.argv is None and not self.command:
            raise ValueError("bash requires either argv or command")
        if self.argv is not None and not self.argv:
            raise ValueError("bash argv must not be empty")
        return self


DENY_PATTERNS = [
    r"\brm\s+-[^&|;]*r[^&|;]*f\s+/",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bformat\b",
    r"\buseradd\b",
    r"\bdeluser\b",
    r"\bpasswd\b",
]
DESTRUCTIVE_WITH_ABSOLUTE_PATH = re.compile(r"\b(rm|del|erase|rmdir)\b.*(\s/|\s[A-Za-z]:\\)")
UNSUPPORTED_SHELL_TOKENS = ("&&", "||", "|", ">", "<", ";")


def _reject_reason(command: str) -> str | None:
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in DENY_PATTERNS):
        return "dangerous command is not allowed"
    if DESTRUCTIVE_WITH_ABSOLUTE_PATH.search(command):
        return "destructive absolute-path command is not allowed"
    return None


def _unsupported_shell_syntax(command: str) -> str | None:
    stripped = command.strip()
    first = stripped.split(maxsplit=1)[0].lower() if stripped else ""
    if first == "cd":
        return "cd is not supported because commands already run in the selected cwd"
    for token in UNSUPPORTED_SHELL_TOKENS:
        if _contains_unquoted(command, token):
            return f"shell syntax '{token}' is not supported when shell=false"
    return None


def _contains_unquoted(command: str, token: str) -> bool:
    quote: str | None = None
    i = 0
    while i < len(command):
        char = command[i]
        if char in {"'", '"'} and (i == 0 or command[i - 1] != "\\"):
            quote = None if quote == char else char if quote is None else quote
        if quote is None and command.startswith(token, i):
            return True
        i += 1
    return False


def _split_command(command: str, shell: bool) -> list[str] | str:
    if shell:
        return command
    return shlex.split(command)


def _display_command(arguments: BashArgs) -> str:
    if arguments.argv is not None:
        return " ".join(arguments.argv)
    return arguments.command or ""


def _verification_kind(command: str) -> str | None:
    lowered = command.lower()
    if "pytest" in lowered:
        return "pytest"
    if "task_service.cli" in lowered or " validate" in f" {lowered} ":
        return "acceptance"
    return None


def _resolve_cwd(workspace, requested: str):
    root = ensure_workspace_root(workspace)
    raw = os.fspath(requested)
    path = os.path.normpath(raw or ".")
    if os.path.isabs(path):
        raise ValueError("absolute cwd paths are not allowed")
    candidate = (root / path).resolve(strict=False)
    if not is_inside_path(candidate, root):
        raise ValueError("cwd escapes workspace")
    if not candidate.exists():
        raise FileNotFoundError(requested)
    return candidate


def _truncate_stream(name: str, value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return f"{name}:\n{value}", False
    half = limit // 2
    return f"{name}:\n{value[:half]}\n...[truncated]...\n{value[-half:]}", True


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Run one command inside the workspace with timeout and output capture. Prefer argv, for example "
        '{"argv": ["find", ".", "-type", "f"], "cwd": "."} or '
        '{"argv": ["python", "-m", "pytest", "-q"], "cwd": "."}. '
        "Commands already run in cwd; do not use cd, pipes, redirection, &&, ||, semicolons, or shell built-ins."
    )
    args_model = BashArgs

    def execute(self, call_id: str, arguments: BashArgs, context: ToolContext) -> ToolResult:
        command = _display_command(arguments)
        normalized_command = " ".join(command.split())
        try:
            cwd = _resolve_cwd(context.workspace, arguments.cwd)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary="bash rejected: cwd escapes workspace",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
                metadata={"command": command, "normalized_command": normalized_command, "cwd": arguments.cwd},
            )
        if not cwd.is_dir():
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary="bash rejected: cwd is not a directory",
                output=f"cwd is not a directory: {arguments.cwd}",
                error_type=ErrorType.TOOL,
                error_message="cwd is not a directory",
                metadata={"command": command, "normalized_command": normalized_command, "cwd": str(cwd)},
            )
        if arguments.command and not context.config.bash.shell:
            unsupported = _unsupported_shell_syntax(arguments.command)
            if unsupported:
                return ToolResult(
                    tool_call_id=call_id,
                    tool_name=self.name,
                    success=False,
                    summary="unsupported_shell_syntax",
                    output=(
                        f"{unsupported}. Commands already run inside the workspace. "
                        'Use argv such as {"argv": ["python", "-m", "pytest", "-q"], "cwd": "."} '
                        'or command "python -m pytest -q" without cd, &&, pipes, redirection, or shell built-ins.'
                    ),
                    error_type=ErrorType.PROTOCOL,
                    error_message="unsupported_shell_syntax",
                    metadata={
                        "command": command,
                        "normalized_command": normalized_command,
                        "cwd": str(cwd),
                        "unsupported_shell_syntax": True,
                    },
                )
        reason = _reject_reason(command)
        if reason:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"bash rejected: {reason}",
                output=reason,
                error_type=ErrorType.TOOL,
                error_message=reason,
                metadata={"command": command, "normalized_command": normalized_command, "cwd": str(cwd)},
            )
        timeout = min(arguments.timeout or context.config.bash.timeout_seconds, context.config.bash.timeout_seconds)
        started = time.monotonic()
        timed_out = False
        try:
            argv = arguments.argv if arguments.argv is not None else _split_command(arguments.command or "", context.config.bash.shell)
            popen_kwargs = {
                "cwd": cwd,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "shell": context.config.bash.shell,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["preexec_fn"] = os.setsid
            process = subprocess.Popen(argv, **popen_kwargs)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            duration = time.monotonic() - started
            stdout = stdout or ""
            stderr = stderr or ""
            artifact = context.tool_outputs_dir / f"tool-output-{call_id}.txt"
            full_output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            artifact.write_text(full_output, encoding="utf-8")
            stream_limit = max(50, context.config.bash.max_output_chars // 2)
            stdout_output, stdout_truncated = _truncate_stream("STDOUT", stdout, stream_limit)
            stderr_output, stderr_truncated = _truncate_stream("STDERR", stderr, stream_limit)
            truncated = stdout_truncated or stderr_truncated
            exit_code = -1 if timed_out else int(process.returncode or 0)
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=not timed_out,
                summary=f"bash finished with exit code {exit_code}",
                output=f"{stdout_output}\n{stderr_output}",
                metadata={
                    "command": command,
                    "normalized_command": normalized_command,
                    "argv": arguments.argv,
                    "cwd": str(cwd),
                    "exit_code": exit_code,
                    "code_epoch": int(getattr(context, "code_epoch", 0)),
                    "verification_kind": _verification_kind(command),
                    "duration_seconds": duration,
                    "timed_out": timed_out,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "truncated": truncated,
                    "output_artifact": str(artifact),
                    "stdout_artifact": str(artifact),
                    "combined_artifact": str(artifact),
                    "platform": sys.platform,
                },
                artifact_path=str(artifact),
                error_type=ErrorType.TOOL if timed_out else None,
                error_message="command timed out" if timed_out else None,
            )
        except (OSError, ValueError) as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"bash failed: {type(exc).__name__}",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
                metadata={"command": command, "normalized_command": normalized_command, "argv": arguments.argv, "cwd": str(cwd)},
            )
