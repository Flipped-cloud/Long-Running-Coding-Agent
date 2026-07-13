from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from longrun_agent.exceptions import ConfigurationError, StateStoreError
from longrun_agent.state.schema import PlanRevision, ProjectState
from longrun_agent.tools.path_guard import is_inside_path


class ProjectStateStore:
    def __init__(self, root: Path, *, workspace_root: Path | None = None, atomic_write: bool = True):
        self.root = root.resolve()
        self.atomic_write = atomic_write
        if workspace_root is not None:
            workspace = workspace_root.resolve()
            if is_inside_path(self.root, workspace):
                raise ConfigurationError("state root must not be inside the agent workspace")
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def state_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project_state.json"

    def events_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project_events.jsonl"

    def sessions_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "sessions.jsonl"

    def metrics_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project_metrics.json"

    def final_verification_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "final_verification.txt"

    def revisions_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "plan_revisions"

    def revision_path(self, project_id: str, revision_id: str) -> Path:
        return self.revisions_dir(project_id) / f"{revision_id}.json"

    def exists(self, project_id: str) -> bool:
        return self.state_path(project_id).exists()

    def list_projects(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if (path / "project_state.json").exists())

    def create(self, state: ProjectState) -> None:
        if self.exists(state.project_id):
            raise FileExistsError(state.project_id)
        self.save(state)

    def load(self, project_id: str) -> ProjectState:
        path = self.state_path(project_id)
        try:
            return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StateStoreError(f"project state does not exist: {path}") from exc
        except (ValueError, ValidationError) as exc:
            raise StateStoreError(f"project state is not readable JSON or failed validation: {path}: {exc}") from exc

    def save(self, state: ProjectState) -> None:
        state = ProjectState.model_validate(state.model_dump())
        project_dir = self.project_dir(state.project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        self.revisions_dir(state.project_id).mkdir(exist_ok=True)
        path = self.state_path(state.project_id)
        payload = json.dumps(state.model_dump(mode="json"), indent=2)
        if self.atomic_write:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        else:
            path.write_text(payload, encoding="utf-8")
        self.save_revisions(state.project_id, state.revisions)

    def save_revisions(self, project_id: str, revisions: list[PlanRevision]) -> None:
        self.revisions_dir(project_id).mkdir(parents=True, exist_ok=True)
        for revision in revisions:
            self.revision_path(project_id, revision.revision_id).write_text(
                json.dumps(revision.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

    def append_session(self, project_id: str, payload: dict) -> None:
        path = self.sessions_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StateStoreError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        return rows

    def read_sessions(self, project_id: str) -> list[dict[str, Any]]:
        return self.read_jsonl(self.sessions_path(project_id))

    def read_events(self, project_id: str) -> list[dict[str, Any]]:
        return self.read_jsonl(self.events_path(project_id))

    def write_metrics(self, project_id: str, metrics: dict[str, Any]) -> None:
        path = self.metrics_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
