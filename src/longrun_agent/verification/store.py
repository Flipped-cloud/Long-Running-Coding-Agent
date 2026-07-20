from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from longrun_agent.exceptions import ConfigurationError, StateStoreError
from longrun_agent.verification.schema import TestCandidate, VerificationContract, VerificationPurpose, VerificationReport


class VerificationStore:
    def __init__(self, root: Path, project_id: str, *, workspace_root: Path, atomic_write: bool = True):
        self.root = root.resolve() / project_id / "verification"
        workspace = workspace_root.resolve()
        if self.root == workspace or workspace in self.root.parents:
            raise ConfigurationError("verification store must be outside workspace")
        self.atomic_write = atomic_write
        for path in (
            self.contracts_dir,
            self.baseline_dir,
            self.reports_dir,
            self.test_candidates_dir,
            self.private_artifacts_dir,
            self.public_artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def contracts_dir(self) -> Path:
        return self.root / "contracts"

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def test_candidates_dir(self) -> Path:
        return self.root / "test_candidates"

    @property
    def private_artifacts_dir(self) -> Path:
        return self.root / "artifacts" / "private"

    @property
    def public_artifacts_dir(self) -> Path:
        return self.root / "artifacts" / "public"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def private_events_path(self) -> Path:
        return self.root / "private_events.jsonl"

    def contract_path(self, contract: VerificationContract) -> Path:
        return self.contracts_dir / f"{contract.contract_id}-v{contract.contract_version}.json"

    def save_contract(self, contract: VerificationContract) -> Path:
        frozen = contract.freeze()
        path = self.contract_path(frozen)
        if path.exists():
            existing = VerificationContract.model_validate_json(path.read_text(encoding="utf-8"))
            if existing.model_dump(mode="json") != frozen.model_dump(mode="json"):
                raise StateStoreError("frozen verification contract cannot be modified in place")
            return path
        self._write_model(path, frozen)
        return path

    def load_contract(self, contract_id: str, version: int | None = None) -> VerificationContract:
        pattern = f"{contract_id}-v{version}.json" if version is not None else f"{contract_id}-v*.json"
        paths = sorted(self.contracts_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(contract_id)
        return VerificationContract.model_validate_json(paths[-1].read_text(encoding="utf-8"))

    def verify_contract_hash(self, contract: VerificationContract) -> bool:
        return bool(contract.contract_hash) and contract.contract_hash == contract.canonical_hash()

    def save_report(self, report: VerificationReport) -> Path:
        path = self.reports_dir / f"{report.report_id}.json"
        self._write_model(path, report)
        return path

    def load_report(self, report_id: str) -> VerificationReport:
        path = self.reports_dir / f"{report_id}.json"
        return VerificationReport.model_validate_json(path.read_text(encoding="utf-8"))

    def list_reports(self, purpose: VerificationPurpose | None = None) -> list[VerificationReport]:
        reports = []
        for path in sorted(self.reports_dir.glob("*.json")):
            try:
                report = VerificationReport.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, ValidationError):
                continue
            if purpose is None or report.purpose == purpose:
                reports.append(report)
        return sorted(reports, key=lambda report: (report.created_at, report.report_id))

    def save_test_candidate(self, candidate: TestCandidate) -> Path:
        path = self.test_candidates_dir / f"{candidate.candidate_id}.json"
        self._write_model(path, candidate)
        return path

    def load_test_candidate(self, candidate_id: str) -> TestCandidate:
        path = self.test_candidates_dir / f"{candidate_id}.json"
        return TestCandidate.model_validate_json(path.read_text(encoding="utf-8"))

    def list_test_candidates(self, *, task_id: str | None = None) -> list[TestCandidate]:
        candidates = []
        for path in sorted(self.test_candidates_dir.glob("*.json")):
            try:
                candidate = TestCandidate.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, ValidationError):
                continue
            if task_id is None or candidate.task_id == task_id:
                candidates.append(candidate)
        return candidates

    def append_verification_event(self, event_type: str, **payload: Any) -> None:
        self._append_event(self.events_path, event_type, payload)

    def append_private_verification_event(self, event_type: str, **payload: Any) -> None:
        self._append_event(self.private_events_path, event_type, payload)

    def _append_event(self, path: Path, event_type: str, payload: dict[str, Any]) -> None:
        from longrun_agent.state.schema import utc_now

        defaults = {
            "project_id": None,
            "task_id": None,
            "session_id": None,
            "contract_id": None,
            "contract_hash": None,
            "report_id": None,
            "check_id": None,
            "trial_id": None,
            "case_id": None,
            "verdict": None,
            "sanitized_reason": "",
            "evidence_ids": [],
            "artifact_paths": [],
        }
        record = {"event_type": event_type, "timestamp": utc_now(), **defaults, **payload}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_model(self, path: Path, model) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True)
        if not self.atomic_write:
            path.write_text(payload, encoding="utf-8")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
