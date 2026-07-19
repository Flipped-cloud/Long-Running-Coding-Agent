from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from longrun_agent.knowledge.schema import ExperienceEvidenceItem, ExperienceEvidencePack, utc_now

SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "authorization", "bearer ")


class RepositoryProfile:
    def __init__(
        self,
        *,
        repository_fingerprint: str,
        language_tags: list[str],
        framework_tags: list[str],
        tool_tags: list[str],
    ):
        self.repository_fingerprint = repository_fingerprint
        self.language_tags = language_tags
        self.framework_tags = framework_tags
        self.tool_tags = tool_tags


class RepositoryProfiler:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()

    def profile(self) -> RepositoryProfile:
        try:
            files = [path for path in self.workspace_root.rglob("*") if path.is_file() and not _skip(path, self.workspace_root)]
        except OSError:
            files = []
        relative = sorted(_relative(path, self.workspace_root) for path in files)
        extensions = sorted({Path(path).suffix.lower() for path in relative if Path(path).suffix})
        language_tags = _language_tags(extensions)
        framework_tags = _framework_tags(self.workspace_root)
        tool_tags = _tool_tags(self.workspace_root, extensions)
        digest_source = "\n".join([*relative[:200], *framework_tags, *tool_tags])
        fingerprint = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16] if digest_source else "unknown"
        return RepositoryProfile(
            repository_fingerprint=fingerprint,
            language_tags=language_tags,
            framework_tags=framework_tags,
            tool_tags=tool_tags,
        )


def build_experience_pack(
    *,
    project_id: str,
    task_id: str,
    task_objective: str,
    acceptance_criteria: list[str],
    session_record: dict[str, Any],
    plan_version: int,
    starting_task_status: str,
    ending_task_status: str,
    workspace_root: Path,
    max_evidence_items: int = 40,
) -> ExperienceEvidencePack:
    profile = RepositoryProfiler(workspace_root).profile()
    session_id = str(session_record.get("session_id") or "")
    run_id = str(session_record.get("run_id") or session_id)
    items: list[ExperienceEvidenceItem] = []

    def add(event_type: str, **kwargs: Any) -> None:
        if len(items) >= max_evidence_items:
            return
        evidence_id = f"{session_id}:e{len(items) + 1}"
        items.append(
            ExperienceEvidenceItem(
                evidence_id=evidence_id,
                project_id=project_id,
                task_id=task_id,
                session_id=session_id,
                run_id=run_id,
                context_segment_id=int(session_record.get("context_segment_count") or 1),
                step=int(kwargs.pop("step", len(items) + 1) or 0),
                event_type=event_type,
                timestamp=str(session_record.get("finished_at") or utc_now()),
                **_sanitize_kwargs(kwargs, workspace_root),
            )
        )

    for raw_path in session_record.get("read_files") or []:
        safe_path = _sanitize_path(str(raw_path), workspace_root)
        add("read_file", tool_name="read_file", path=safe_path, summary=f"read {safe_path}", success=True)
    for raw_path in session_record.get("changed_files") or []:
        safe_path = _sanitize_path(str(raw_path), workspace_root)
        add("write_file", tool_name="write_file", path=safe_path, summary=f"changed {safe_path}", success=True)
    bash_observations = session_record.get("bash_observations") or []
    if bash_observations:
        for observation in bash_observations:
            command = str(observation.get("command") or "")
            exit_code = observation.get("exit_code")
            is_verification = bool(observation.get("is_verification")) or _is_verification(command)
            event_type = "test_command" if is_verification else "bash"
            add(
                event_type,
                tool_name="bash",
                command=command,
                summary=f"command exited {exit_code}: {command}",
                success=bool(observation.get("success")) and exit_code == 0,
                exit_code=exit_code if isinstance(exit_code, int) else None,
                output_excerpt=observation.get("output_excerpt"),
                error_type=observation.get("error_type"),
                artifact_path=observation.get("artifact_path"),
                step=int(observation.get("operation_index") or len(items) + 1),
            )
    else:
        for command, exit_code in zip(session_record.get("bash_commands") or [], session_record.get("bash_exit_codes") or [], strict=False):
            event_type = "test_command" if _is_verification(command) else "bash"
            add(
                event_type,
                tool_name="bash",
                command=command,
                summary=f"command exited {exit_code}: {command}",
                success=exit_code == 0,
                exit_code=exit_code,
            )
    for command in session_record.get("successful_test_commands") or []:
        add(
            "successful_verification",
            tool_name="bash",
            command=command,
            summary=f"successful verification: {command}",
            success=True,
            exit_code=0,
        )
    for action in session_record.get("repeated_tool_calls") or []:
        add("repeated_tool_call", summary=f"repeated action: {action}", success=False)
    if session_record.get("no_progress"):
        add("no_progress", summary="session ended without meaningful progress", success=False)
    terminal = session_record.get("terminal_signal")
    if terminal == "blocker":
        add("blocker", summary="task reported blocker", success=False)
    if terminal == "completion_request":
        add("completion_request", summary="task requested completion", success=True)
    if int(session_record.get("context_reset_count") or 0) > 0:
        add("context_reset_observed", summary="context reset occurred inside session", success=True)

    failed_verifications = [
        str(observation.get("command") or "")
        for observation in bash_observations
        if (bool(observation.get("is_verification")) or _is_verification(str(observation.get("command") or "")))
        and observation.get("exit_code") != 0
    ]
    if not failed_verifications:
        failed_verifications = [
            command
            for command, exit_code in zip(
                session_record.get("bash_commands") or [], session_record.get("bash_exit_codes") or [], strict=False
            )
            if _is_verification(command) and exit_code != 0
        ]
    successful_verifications = list(session_record.get("successful_acceptance_commands") or []) or list(
        session_record.get("successful_test_commands") or []
    )
    progress_made = bool(session_record.get("changed_files") or successful_verifications or terminal == "completion_request")
    return ExperienceEvidencePack(
        project_id=project_id,
        task_id=task_id,
        task_objective=task_objective,
        acceptance_criteria=acceptance_criteria,
        session_id=session_id,
        run_id=run_id,
        plan_version=plan_version,
        repository_fingerprint=profile.repository_fingerprint,
        language_tags=profile.language_tags,
        framework_tags=profile.framework_tags,
        tool_tags=profile.tool_tags,
        starting_task_status=starting_task_status,
        ending_task_status=ending_task_status,
        terminal_signal=terminal,
        run_status=str(session_record.get("run_status") or ""),
        progress_made=progress_made,
        no_progress=bool(session_record.get("no_progress")),
        files_read=_sanitize_list(session_record.get("read_files") or [], workspace_root),
        files_touched=_sanitize_list(session_record.get("files_touched") or [], workspace_root),
        files_changed=_sanitize_list(session_record.get("changed_files") or [], workspace_root),
        commands=[_sanitize_text(command) for command in session_record.get("bash_commands") or []],
        failed_verifications=failed_verifications,
        successful_verifications=successful_verifications,
        repeated_actions=list(session_record.get("repeated_tool_calls") or []),
        blockers=["task blocker"] if terminal == "blocker" else [],
        context_reset_count=int(session_record.get("context_reset_count") or 0),
        evidence_items=items,
    )


def _sanitize_kwargs(kwargs: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    sanitized = {}
    for key, value in kwargs.items():
        if value is None:
            sanitized[key] = None
        elif key in {"path", "artifact_path"}:
            sanitized[key] = _sanitize_path(str(value), workspace_root)
        elif key == "command":
            sanitized[key] = _sanitize_text(str(value))
        elif isinstance(value, str):
            sanitized[key] = _sanitize_text(value)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_list(items: list[str], workspace_root: Path) -> list[str]:
    deduped: list[str] = []
    for item in items:
        path = _sanitize_path(str(item), workspace_root)
        if path and path not in deduped:
            deduped.append(path)
    return deduped


def _sanitize_path(value: str, workspace_root: Path) -> str:
    """Return a portable workspace-relative path or a safe basename.

    The function deliberately understands both POSIX and Windows path syntax,
    regardless of the host operating system. Paths inside ``workspace_root``
    are stored as POSIX-style relative paths. External absolute paths and
    traversal paths are reduced to their basename so host directory details
    are not persisted in long-term knowledge.
    """
    raw = value.strip()
    if not raw:
        return ""

    workspace = workspace_root.resolve()
    native_path = Path(raw)

    # First handle paths that are absolute according to the host platform.
    if native_path.is_absolute():
        try:
            return native_path.resolve(strict=False).relative_to(workspace).as_posix()
        except (OSError, RuntimeError, ValueError):
            return native_path.name

    # On POSIX, pathlib.Path does not recognize Windows drive/UNC paths as
    # absolute. Parse them explicitly so ``C:\\outside\\secrets.py`` does
    # not leak its full host path into a portable evidence record.
    windows_path = PureWindowsPath(raw)
    is_windows_absolute = bool(windows_path.drive) or windows_path.is_absolute() or raw.startswith("\\\\")
    if is_windows_absolute:
        return windows_path.name

    # Preserve portable relative paths, including Windows-style relative paths,
    # only when they resolve inside the configured workspace.
    if "\\" in raw:
        relative_parts = windows_path.parts
        basename = windows_path.name
    else:
        posix_path = PurePosixPath(raw)
        relative_parts = posix_path.parts
        basename = posix_path.name

    try:
        candidate = workspace.joinpath(*relative_parts).resolve(strict=False)
        return candidate.relative_to(workspace).as_posix()
    except (OSError, RuntimeError, ValueError):
        return basename


def _sanitize_text(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        return "[redacted secret-bearing text]"
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _skip(path: Path, root: Path) -> bool:
    try:
        parts = {part.lower() for part in path.resolve(strict=False).relative_to(root).parts}
    except ValueError:
        parts = {part.lower() for part in path.parts}
    return any(part in parts for part in {".git", ".runs", "__pycache__", ".pytest_cache", ".ruff_cache"})


def _language_tags(extensions: list[str]) -> list[str]:
    mapping = {".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json", ".md": "markdown"}
    return sorted({mapping[ext] for ext in extensions if ext in mapping})


def _framework_tags(root: Path) -> list[str]:
    tags: set[str] = set()
    if (root / "pyproject.toml").exists():
        text = _safe_read(root / "pyproject.toml")
        if "pytest" in text:
            tags.add("pytest")
        if "pydantic" in text:
            tags.add("pydantic")
    if (root / "package.json").exists():
        text = _safe_read(root / "package.json")
        if "react" in text:
            tags.add("react")
        if "vitest" in text:
            tags.add("vitest")
    return sorted(tags)


def _tool_tags(root: Path, extensions: list[str]) -> list[str]:
    tags = set()
    if ".py" in extensions:
        tags.add("pytest")
        tags.add("python")
    if (root / "package.json").exists():
        tags.add("npm")
    return sorted(tags)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()[:20000]
    except OSError:
        return ""


def _is_verification(command: str) -> bool:
    lowered = command.lower()
    return "pytest" in lowered or "validate" in lowered or "test" in lowered
