from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

from longrun_agent.exceptions import WorkspaceSecurityError
from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.path_guard import relative_to_workspace, resolve_workspace_path


class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_range(self) -> ReadFileArgs:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


def _looks_binary(path) -> bool:
    return b"\x00" in path.read_bytes()[:4096]


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file inside the workspace with line numbers."
    args_model = ReadFileArgs

    def execute(self, call_id: str, arguments: ReadFileArgs, context: ToolContext) -> ToolResult:
        try:
            path = resolve_workspace_path(context.workspace, arguments.path, must_exist=True)
            if path.is_dir():
                raise IsADirectoryError(arguments.path)
            if _looks_binary(path):
                return ToolResult(
                    tool_call_id=call_id,
                    tool_name=self.name,
                    success=False,
                    summary="read_file failed: binary files are not supported",
                    error_type=ErrorType.TOOL,
                    error_message="binary files are not supported",
                )

            text = path.read_text(encoding="utf-8")
            raw_bytes = path.read_bytes()
            stat = path.stat()
            lines = text.splitlines()
            total = len(lines)
            requested_end = arguments.end_line or (arguments.start_line + context.config.read_file.max_lines - 1)
            max_end = arguments.start_line + context.config.read_file.max_lines - 1
            end_line = min(requested_end, max_end)
            line_truncated = requested_end > end_line
            start = min(arguments.start_line, total + 1)
            end = min(end_line, total)
            selected = lines[start - 1 : end] if start <= end else []
            body = "\n".join(f"{idx} | {line}" for idx, line in enumerate(selected, start=start))
            char_truncated = len(body) > context.config.read_file.max_chars
            if char_truncated:
                half = context.config.read_file.max_chars // 2
                body = f"{body[:half]}\n...[truncated by max_chars]...\n{body[-half:]}"
            truncated = line_truncated or char_truncated or end < total
            rel = relative_to_workspace(context.workspace, path)
            output = f"File: {rel}\nTotal lines: {total}\nShowing lines: {start}-{end}\nTruncated: {str(truncated).lower()}\n\n{body}"
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=True,
                summary=f"read_file succeeded: {rel} lines {start}-{end}",
                output=output,
                metadata={
                    "path": rel,
                    "total_lines": total,
                    "start_line": start,
                    "end_line": end,
                    "truncated": truncated,
                    "encoding": "utf-8",
                    "has_previous": start > 1,
                    "has_next": end < total,
                    "content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "size_bytes": len(raw_bytes),
                    "modified_time_ns": stat.st_mtime_ns,
                },
            )
        except (FileNotFoundError, WorkspaceSecurityError, UnicodeDecodeError, IsADirectoryError) as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"read_file failed: {type(exc).__name__}",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
            )
