from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longrun_agent.evaluation.schema import AdapterVerificationResult, EvaluationTaskCase, TrialDescriptor
from longrun_agent.state.schema import utc_now
from longrun_agent.verification.gateway import VerificationGateway
from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import OraclePrivateContract, VerificationPurpose
from longrun_agent.verification.snapshot import CopySnapshotProvider
from longrun_agent.verification.store import VerificationStore


class OfflineOracleEvaluator:
    def prepare_baseline(self, *, case: EvaluationTaskCase, descriptor: TrialDescriptor, workspace: Path) -> None:
        self._load_private_contract(descriptor)
        root = self._oracle_root(descriptor)
        snapshots = CopySnapshotProvider(workspace, root / "snapshots")
        manifest = snapshots.create_baseline()
        payload = {
            "baseline_fingerprint": manifest.fingerprint,
            "baseline_manifest": str(snapshots.baseline_manifest_path),
            "created_at": utc_now(),
        }
        (root / "baseline.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._append_event(
            root / "private_events.jsonl",
            "oracle_baseline_created",
            case_id=case.case_id,
            trial_id=descriptor.trial_id,
            baseline_fingerprint=manifest.fingerprint,
            baseline_manifest=str(snapshots.baseline_manifest_path),
        )

    def evaluate(
        self,
        *,
        case: EvaluationTaskCase,
        descriptor: TrialDescriptor,
        project_id: str,
        final_workspace: Path,
    ) -> AdapterVerificationResult:
        private = self._load_private_contract(descriptor)
        root = self._oracle_root(descriptor)
        snapshots = CopySnapshotProvider(final_workspace, root / "snapshots")
        if not snapshots.baseline_manifest_path.exists():
            raise ValueError("oracle baseline is missing; reset must create it before the agent runs")

        source_contract = private.contract
        source_contract_hash = private.private_fingerprint
        if source_contract.project_id == "__PROJECT_ID__":
            source_contract = source_contract.model_copy(update={"project_id": project_id})
        elif source_contract.project_id != project_id:
            raise ValueError(f"oracle contract project_id {source_contract.project_id!r} does not match project {project_id!r}")
        contract = source_contract.freeze()
        store = VerificationStore(root / "store", project_id, workspace_root=final_workspace)
        store.save_contract(contract)
        gateway = VerificationGateway(
            store=store,
            snapshot_manager=snapshots,
            runner=VerificationRunner(store.root / "artifacts"),
            preserve_failed_snapshot=False,
            purpose=VerificationPurpose.ORACLE,
        )
        report = gateway.verify(contract, test_candidates=[])
        private_path = store.reports_dir / f"{report.report_id}.json"
        total = report.summary.required_checks_passed + report.summary.required_checks_failed
        verifier_seconds = sum(result.duration_seconds for result in [*report.baseline_results, *report.candidate_results])
        sanitized_summary = (
            f"Oracle evaluation finished with verdict {report.verdict.value}; "
            f"{report.summary.required_checks_passed} of {total} required categories passed."
        )
        event = {
            "case_id": case.case_id,
            "trial_id": descriptor.trial_id,
            "oracle_report_id": report.report_id,
            "oracle_verdict": report.verdict.value,
            "f2p_rate": report.summary.f2p_rate,
            "p2p_rate": report.summary.p2p_rate,
            "integrity_passed": report.summary.integrity_passed,
            "required_checks_passed": report.summary.required_checks_passed,
            "required_checks_failed": report.summary.required_checks_failed,
            "hidden_checks_passed_count": report.summary.hidden_checks_passed,
            "hidden_checks_failed_count": report.summary.hidden_checks_failed,
        }
        self._append_event(root / "public_events.jsonl", "oracle_verification_finished", **event)
        self._append_event(
            root / "private_events.jsonl",
            "oracle_report_saved",
            **event,
            oracle_report_private_path=str(private_path),
        )
        return AdapterVerificationResult(
            oracle_report_id=report.report_id,
            oracle_verdict=report.verdict.value,
            oracle_report_private_path=str(private_path),
            oracle_f2p_rate=report.summary.f2p_rate,
            oracle_p2p_rate=report.summary.p2p_rate,
            oracle_integrity_passed=report.summary.integrity_passed,
            oracle_partial_resolution=report.summary.partial_resolution,
            oracle_required_checks_passed=report.summary.required_checks_passed,
            oracle_required_checks_failed=report.summary.required_checks_failed,
            oracle_total_requirements=total,
            oracle_verified_requirements=report.summary.required_checks_passed,
            oracle_verifier_seconds=verifier_seconds,
            oracle_contract_id=contract.contract_id,
            oracle_contract_hash=source_contract_hash,
            oracle_baseline_fingerprint=report.baseline_fingerprint,
            oracle_candidate_fingerprint=report.candidate_fingerprint,
            sanitized_summary=sanitized_summary,
        )

    @staticmethod
    def _load_private_contract(descriptor: TrialDescriptor) -> OraclePrivateContract:
        path = descriptor.trial_dir / "oracle" / "private" / "contract.json"
        if not path.exists():
            raise ValueError("oracle private contract is missing; prepare must freeze it before execution")
        return OraclePrivateContract.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _oracle_root(descriptor: TrialDescriptor) -> Path:
        root = descriptor.trial_dir / "oracle"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _append_event(path: Path, event_type: str, **payload: Any) -> None:
        record = {"event_type": event_type, "timestamp": utc_now(), **payload}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
