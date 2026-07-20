from __future__ import annotations

from pydantic import BaseModel

from longrun_agent.protocol import RunStatus


class ProjectRunOutcome(BaseModel):
    project_id: str
    status: str
    sessions_run: int
    state_path: str
    run_statuses: list[RunStatus]
    verification_verdict: str | None = None
    verification_report_id: str | None = None
