from __future__ import annotations

import difflib
import hashlib
import os
import tempfile

from pydantic import BaseModel, Field

from longrun_agent.exceptions import WorkspaceSecurityError
from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.path_guard import relative_to_workspace
from longrun_agent.tools.workspace_policy import ACCESS_DENIED_MESSAGE, WorkspaceAccessDenied


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1)
    content: str


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write a complete UTF-8 file inside the workspace using an atomic replace."
    args_model = WriteFileArgs

    def execute(self, call_id: str, arguments: WriteFileArgs, context: ToolContext) -> ToolResult:
        try:
            if len(arguments.content) > context.config.write_file.max_chars:
                return ToolResult(
                    tool_call_id=call_id,
                    tool_name=self.name,
                    success=False,
                    summary="write_file failed: content exceeds max_chars",
                    error_type=ErrorType.TOOL,
                    error_message="content exceeds configured max_chars",
                )
            path = context.workspace_policy.resolve_write(arguments.path)
            if path.exists() and path.is_dir():
                raise IsADirectoryError(arguments.path)
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            created = not path.exists()
            after = arguments.content
            rel = relative_to_workspace(context.workspace, path)
            code_epoch = int(getattr(context, "code_epoch", 0))
            if before == after and not created:
                return ToolResult(
                    tool_call_id=call_id,
                    tool_name=self.name,
                    success=True,
                    summary=f"write_file no_change: {rel}",
                    output=f"no_change {rel}",
                    metadata={
                        "path": rel,
                        "status": "no_change",
                        "before_sha256": sha256_text(before),
                        "after_sha256": sha256_text(after),
                        "before_hash": sha256_text(before),
                        "after_hash": sha256_text(after),
                        "current_sha256": sha256_text(after),
                        "code_epoch": code_epoch,
                        "size_before": len(before.encode("utf-8")),
                        "size_after": len(after.encode("utf-8")),
                        "before_line_count": len(before.splitlines()) if before else 0,
                        "after_line_count": len(after.splitlines()) if after else 0,
                        "changed_lines": 0,
                        "atomic": True,
                    },
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, newline="") as handle:
                handle.write(after)
                temp_name = handle.name
            os.replace(temp_name, path)

            diff = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
            artifact_path = None
            if context.config.write_file.save_diff and diff:
                artifact = context.diffs_dir / f"diff-{call_id}.patch"
                artifact.write_text(diff, encoding="utf-8")
                artifact_path = str(artifact)
            changed_lines = sum(1 for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            if len(diff) > 4000:
                diff_output = diff[:2000] + "\n...[truncated, full diff saved to artifact]...\n" + diff[-2000:]
            else:
                diff_output = diff
            status = "created" if created else "updated"
            code_epoch += 1
            context.code_epoch = code_epoch
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=True,
                summary=f"write_file {status}: {rel}",
                output=diff_output or f"{status} {rel}",
                metadata={
                    "path": rel,
                    "status": status,
                    "before_sha256": None if created else sha256_text(before),
                    "after_sha256": sha256_text(after),
                    "before_hash": None if created else sha256_text(before),
                    "after_hash": sha256_text(after),
                    "current_sha256": sha256_text(after),
                    "code_epoch": code_epoch,
                    "size_before": len(before.encode("utf-8")),
                    "size_after": len(after.encode("utf-8")),
                    "before_line_count": len(before.splitlines()) if before else 0,
                    "after_line_count": len(after.splitlines()) if after else 0,
                    "changed_lines": changed_lines,
                    "diff_artifact": artifact_path,
                    "atomic": True,
                },
                artifact_path=artifact_path,
            )
        except WorkspaceAccessDenied:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary="write_file denied by workspace policy",
                output=ACCESS_DENIED_MESSAGE,
                error_type=ErrorType.WORKSPACE_ACCESS_DENIED,
                error_message=ACCESS_DENIED_MESSAGE,
                retryable=True,
            )
        except (WorkspaceSecurityError, OSError, UnicodeDecodeError) as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"write_file failed: {type(exc).__name__}",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
            )
