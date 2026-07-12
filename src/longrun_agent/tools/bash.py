from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time

from pydantic import BaseModel, Field

from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext


class BashArgs(BaseModel):
    command: str = Field(min_length=1)
    timeout: int | None = Field(default=None, ge=1)


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


def _reject_reason(command: str) -> str | None:
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in DENY_PATTERNS):
        return "dangerous command is not allowed"
    if DESTRUCTIVE_WITH_ABSOLUTE_PATH.search(command):
        return "destructive absolute-path command is not allowed"
    return None


def _split_command(command: str, shell: bool) -> list[str] | str:
    if shell:
        return command
    return shlex.split(command)


def _truncate_stream(name: str, value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return f"{name}:\n{value}", False
    half = limit // 2
    return f"{name}:\n{value[:half]}\n...[truncated]...\n{value[-half:]}", True


class BashTool(BaseTool):
    name = "bash"
    description = "Run a command inside the workspace with timeout and output capture."
    args_model = BashArgs

    def execute(self, call_id: str, arguments: BashArgs, context: ToolContext) -> ToolResult:
        reason = _reject_reason(arguments.command)
        if reason:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"bash rejected: {reason}",
                output=reason,
                error_type=ErrorType.TOOL,
                error_message=reason,
                metadata={"command": arguments.command, "cwd": str(context.workspace)},
            )
        timeout = min(arguments.timeout or context.config.bash.timeout_seconds, context.config.bash.timeout_seconds)
        started = time.monotonic()
        timed_out = False
        try:
            argv = _split_command(arguments.command, context.config.bash.shell)
            popen_kwargs = {
                "cwd": context.workspace,
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
                    "command": arguments.command,
                    "cwd": str(context.workspace),
                    "exit_code": exit_code,
                    "duration_seconds": duration,
                    "timed_out": timed_out,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "truncated": truncated,
                    "output_artifact": str(artifact),
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
                metadata={"command": arguments.command, "cwd": str(context.workspace)},
            )
