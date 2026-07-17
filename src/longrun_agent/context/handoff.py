from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.schema import (
    ConfirmedFact,
    EvidenceReference,
    FileState,
    HandoffRecord,
    NextAction,
    TaskContextSeed,
    VerificationState,
)


def deterministic_handoff(
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
    evidence = []
    files: dict[str, FileState] = {}
    verifications: list[VerificationState] = []
    stale_steps = []
    source_steps = []
    for turn in buffer.all_turns():
        source_steps.append(turn.step)
        for result in _turn_results(turn):
            metadata = result.get("metadata") or {}
            evidence_id = f"step-{turn.step}-{result.get('tool_call_id')}"
            evidence.append(
                EvidenceReference(
                    evidence_id=evidence_id,
                    type=str(result.get("tool_name") or "tool"),
                    artifact_path=result.get("artifact_path") or metadata.get("output_artifact"),
                    step=turn.step,
                    description=str(result.get("summary") or ""),
                )
            )
            if result.get("tool_name") == "write_file" and result.get("success"):
                path = str(metadata.get("path") or "")
                if path:
                    files[path] = FileState(
                        path=path,
                        sha256=metadata.get("after_sha256") or metadata.get("current_sha256"),
                        status="current",
                        summary=str(result.get("summary") or ""),
                        last_write_step=turn.step,
                        evidence_id=evidence_id,
                    )
            elif result.get("tool_name") == "read_file" and result.get("success"):
                path = str(metadata.get("path") or "")
                if path and path not in files:
                    files[path] = FileState(
                        path=path,
                        sha256=metadata.get("content_sha256") or _file_sha(workspace_root, path),
                        status="observed",
                        summary=str(result.get("summary") or ""),
                        evidence_id=evidence_id,
                    )
            elif result.get("tool_name") == "bash" and metadata.get("verification_kind"):
                stale = bool(metadata.get("context_status") == "stale" or turn.status.value == "stale")
                if stale:
                    stale_steps.append(evidence_id)
                verifications.append(
                    VerificationState(
                        command=str(metadata.get("normalized_command") or metadata.get("command") or ""),
                        exit_code=int(metadata.get("exit_code") or 0),
                        result=str(result.get("summary") or ""),
                        code_epoch=int(metadata.get("code_epoch") or 0),
                        artifact_path=result.get("artifact_path") or metadata.get("output_artifact"),
                        source_step=turn.step,
                        stale=stale,
                    )
                )
    confirmed = []
    if evidence:
        confirmed.append(
            ConfirmedFact(
                statement="Context reset preserved observed file and verification evidence without changing task status.",
                evidence_ids=[evidence[0].evidence_id],
                confidence=1.0,
                source_step=evidence[0].step,
            )
        )
    return HandoffRecord(
        handoff_id=f"ctx-{uuid.uuid4()}",
        project_id=project_id,
        task_id=seed.task_id,
        session_id=session_id,
        source_segment_id=source_segment_id,
        target_segment_id=target_segment_id,
        plan_version=plan_version,
        task_objective=seed.task_objective,
        acceptance_criteria=seed.acceptance_criteria,
        confirmed_progress=confirmed,
        files_current=sorted(files.values(), key=lambda item: item.path),
        verifications=verifications[-5:],
        next_actions=[NextAction(action="Continue the current task from the preserved handoff and verify after any further write.")],
        evidence_references=evidence,
        stale_items_excluded=stale_steps,
        source_steps=source_steps,
        generator="deterministic",
    )


def _turn_results(turn) -> list[dict[str, Any]]:
    rows = []
    for message in turn.tool_result_messages:
        try:
            rows.append(json.loads(str(message.get("content") or "{}")))
        except json.JSONDecodeError:
            continue
    return rows


def _file_sha(workspace_root: Path | None, rel_path: str) -> str | None:
    if workspace_root is None:
        return None
    path = (workspace_root / rel_path).resolve()
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
