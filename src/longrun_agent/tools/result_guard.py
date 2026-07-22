from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.workspace_policy import WorkspaceAccessPolicy

AGENT_VISIBLE_METADATA_KEYS = frozenset(
    {
        "after_hash",
        "after_line_count",
        "after_sha256",
        "argument_normalizations",
        "atomic",
        "before_hash",
        "before_line_count",
        "before_sha256",
        "changed_lines",
        "code_epoch",
        "combined_artifact",
        "command",
        "content_sha256",
        "current_sha256",
        "diff_artifact",
        "duration_seconds",
        "encoding",
        "exit_code",
        "failure_code",
        "has_next",
        "has_previous",
        "modified_time_ns",
        "normalization_code",
        "normalized_command",
        "output_artifact",
        "path",
        "platform",
        "size_after",
        "size_before",
        "size_bytes",
        "start_line",
        "status",
        "stderr_artifact",
        "stderr_chars",
        "stdout_artifact",
        "stdout_chars",
        "timed_out",
        "total_lines",
        "truncated",
        "unsupported_shell_syntax",
        "verification_kind",
        "marker_categories",
        "private_data_blocked",
        "workspace_access_denied",
    }
)


def project_agent_visible_tool_result(result: ToolResult, policy: WorkspaceAccessPolicy) -> ToolResult:
    observations = {
        "summary": result.summary,
        "output": result.output,
        "error_message": result.error_message,
    }
    normalized = "\n".join(str(value) for value in observations.values() if value is not None).replace("\\", "/").casefold()
    categories = []
    for root in policy.denied_roots:
        marker = str(root).replace("\\", "/").casefold()
        if marker and marker in normalized:
            categories.append("denied_root")
            break
    if any(marker.casefold() in normalized for marker in policy.private_markers if len(marker.strip()) >= 6):
        categories.append("private_marker")
    if categories:
        policy.record_private_block(tool_name=result.tool_name, categories=categories)
        return ToolResult(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            success=False,
            summary="private data blocked from agent-visible tool result",
            output="Private evaluation data was blocked.",
            error_type=ErrorType.PRIVATE_DATA_BLOCKED,
            error_message="Private evaluation data was blocked.",
            retryable=True,
            metadata={"private_data_blocked": True, "marker_categories": sorted(set(categories))},
        )

    metadata = {
        key: value
        for key, value in result.metadata.items()
        if key in AGENT_VISIBLE_METADATA_KEYS and _metadata_value_is_safe(value, policy)
    }
    return result.model_copy(
        update={
            "metadata": metadata,
            "artifact_path": _project_artifact_path(result.artifact_path, policy),
        },
        deep=True,
    )


def sanitize_agent_visible_tool_result(result: ToolResult, policy: WorkspaceAccessPolicy) -> ToolResult:
    return project_agent_visible_tool_result(result, policy)


def _project_artifact_path(artifact_path: str | None, policy: WorkspaceAccessPolicy) -> str | None:
    if not artifact_path:
        return None
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = policy.workspace_root / candidate
    try:
        return candidate.resolve(strict=False).relative_to(policy.workspace_root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def _metadata_value_is_safe(value: Any, policy: WorkspaceAccessPolicy) -> bool:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    normalized = text.replace("\\", "/").casefold()
    for root in policy.denied_roots:
        marker = str(root).replace("\\", "/").casefold()
        if marker and marker in normalized:
            return False
    return not any(marker.casefold() in normalized for marker in policy.private_markers if len(marker.strip()) >= 6)
