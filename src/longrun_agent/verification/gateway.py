from __future__ import annotations

from pathlib import Path

from longrun_agent.verification.generated_tests import TestCandidateValidator
from longrun_agent.verification.integrity import IntegrityValidator
from longrun_agent.verification.renderer import render_agent_feedback
from longrun_agent.verification.runner import VerificationRunner, environment_fingerprint
from longrun_agent.verification.schema import (
    CheckKind,
    CheckVisibility,
    ExecutionStatus,
    TestCandidate,
    VerificationContract,
    VerificationPurpose,
    VerificationReport,
    VerificationSummary,
    VerificationVerdict,
)
from longrun_agent.verification.snapshot import CopySnapshotProvider, SnapshotError
from longrun_agent.verification.store import VerificationStore
from longrun_agent.verification.transitions import baseline_matches_contract, build_transitions, summarize_and_decide


class VerificationGateway:
    def __init__(
        self,
        *,
        store: VerificationStore,
        snapshot_manager: CopySnapshotProvider,
        runner: VerificationRunner,
        preserve_failed_snapshot: bool = True,
        purpose: VerificationPurpose = VerificationPurpose.RUNTIME,
    ):
        self.store = store
        self.snapshot_manager = snapshot_manager
        self.runner = runner
        self.preserve_failed_snapshot = preserve_failed_snapshot
        self.purpose = purpose

    def verify(
        self,
        contract: VerificationContract,
        *,
        task_id: str | None = None,
        test_candidates: list[TestCandidate] | None = None,
    ) -> VerificationReport:
        if not self.store.verify_contract_hash(contract):
            report = self._invalid_contract(contract, task_id, "verification contract hash mismatch")
            self.store.save_report(report)
            self.store.append_verification_event(
                "verification_contract_mismatch",
                project_id=contract.project_id,
                task_id=task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                report_id=report.report_id,
                verdict=report.verdict.value,
                sanitized_reason="contract hash mismatch",
            )
            return report
        self.store.append_verification_event(
            "verification_started",
            project_id=contract.project_id,
            task_id=task_id,
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
        )
        candidate_root = None
        try:
            baseline_manifest = self._ensure_baseline()
            candidate_root, candidate_manifest = self.snapshot_manager.create_candidate()
            self.store.append_verification_event(
                "candidate_snapshot_created",
                project_id=contract.project_id,
                task_id=task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
            )
            baseline_root = self.snapshot_manager.create_baseline_working_copy()
            try:
                if contract.hidden_assets_root:
                    self.snapshot_manager.inject_hidden_assets(baseline_root, contract.hidden_assets_root)
                    self.snapshot_manager.inject_hidden_assets(candidate_root, contract.hidden_assets_root)
                baseline_results = self._run_checks(contract, baseline_root, "baseline", run_candidate_only=False)
                candidate_results = self._run_checks(contract, candidate_root, "candidate", run_candidate_only=True)
            finally:
                self.snapshot_manager.cleanup(baseline_root)
            integrity = IntegrityValidator().validate(
                baseline_manifest,
                candidate_manifest,
                contract.integrity_rules,
                candidate_root,
            )
            transitions = build_transitions(contract.checks, baseline_results, candidate_results)
            for transition in transitions:
                check = next(item for item in contract.checks if item.check_id == transition.check_id)
                self.store.append_verification_event(
                    "verification_transition_computed",
                    project_id=contract.project_id,
                    task_id=task_id,
                    contract_id=contract.contract_id,
                    contract_hash=contract.contract_hash,
                    check_id=transition.check_id if check.visibility == CheckVisibility.PUBLIC else None,
                    sanitized_reason=f"{check.kind.value} transition {transition.transition.value}",
                    verdict=transition.transition.value,
                )
                if transition.transition.value == "P2F":
                    self.store.append_verification_event(
                        "verification_transition_p2f",
                        project_id=contract.project_id,
                        task_id=task_id,
                        contract_id=contract.contract_id,
                        contract_hash=contract.contract_hash,
                        check_id=transition.check_id if check.visibility == CheckVisibility.PUBLIC else None,
                        sanitized_reason="regression category changed from pass to fail",
                    )
            for violation in integrity:
                self.store.append_verification_event(
                    "integrity_violation_detected",
                    project_id=contract.project_id,
                    task_id=task_id,
                    contract_id=contract.contract_id,
                    contract_hash=contract.contract_hash,
                    sanitized_reason=violation.agent_visible_summary,
                    evidence_ids=violation.evidence,
                )
            summary, verdict = summarize_and_decide(
                contract.checks,
                transitions,
                integrity,
                baseline_results,
                candidate_results,
            )
            if not baseline_matches_contract(contract.checks, baseline_results):
                self.store.append_verification_event(
                    "verification_baseline_mismatch",
                    project_id=contract.project_id,
                    task_id=task_id,
                    contract_id=contract.contract_id,
                    contract_hash=contract.contract_hash,
                    verdict=VerificationVerdict.INCONCLUSIVE.value,
                    sanitized_reason="baseline check categories do not match the frozen contract expectations",
                )
            validated_candidates = self._validate_candidates(test_candidates or [], candidate_root)
            artifact_paths = [
                path
                for result in [*baseline_results, *candidate_results]
                for path in [result.stdout_artifact, result.stderr_artifact]
                if path
            ]
            report = VerificationReport(
                purpose=self.purpose,
                project_id=contract.project_id,
                task_id=task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                verdict=verdict,
                baseline_fingerprint=baseline_manifest.fingerprint,
                candidate_fingerprint=candidate_manifest.fingerprint,
                environment_fingerprint=environment_fingerprint(),
                baseline_results=baseline_results,
                candidate_results=candidate_results,
                transitions=transitions,
                integrity_violations=integrity,
                test_candidates=validated_candidates,
                summary=summary,
                infrastructure_error=next(
                    (result.infrastructure_error for result in [*baseline_results, *candidate_results] if result.infrastructure_error),
                    None,
                ),
                artifact_paths=artifact_paths,
            )
            report.sanitized_feedback = render_agent_feedback(report)
        except (OSError, SnapshotError) as exc:
            report = VerificationReport(
                purpose=self.purpose,
                project_id=contract.project_id,
                task_id=task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                verdict=VerificationVerdict.INFRASTRUCTURE_ERROR,
                summary=VerificationSummary(integrity_passed=False),
                infrastructure_error=str(exc),
                sanitized_feedback="Independent verification infrastructure failed; no implementation failure was inferred.",
            )
        self.store.save_report(report)
        self.store.append_verification_event(
            "verification_report_created",
            project_id=contract.project_id,
            task_id=task_id,
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            report_id=report.report_id,
            verdict=report.verdict.value,
            sanitized_reason=report.sanitized_feedback,
            evidence_ids=[],
            artifact_paths=[
                path
                for result in [*report.baseline_results, *report.candidate_results]
                if result.visibility == CheckVisibility.PUBLIC
                for path in [result.stdout_artifact, result.stderr_artifact]
                if path
            ],
        )
        if candidate_root and (report.verdict == VerificationVerdict.VERIFIED or not self.preserve_failed_snapshot):
            self.snapshot_manager.cleanup(candidate_root)
        return report

    def _ensure_baseline(self):
        if self.snapshot_manager.baseline_manifest_path.exists():
            return self.snapshot_manager.load_baseline_manifest()
        manifest = self.snapshot_manager.create_baseline()
        self.store.append_verification_event("baseline_snapshot_created", contract_id=None, contract_hash=None)
        return manifest

    def _run_checks(self, contract: VerificationContract, root: Path, kind: str, *, run_candidate_only: bool):
        results = []
        for check in contract.checks:
            if check.kind == CheckKind.INTEGRITY:
                continue
            if not run_candidate_only and check.kind in {CheckKind.CANDIDATE_ONLY, CheckKind.STATIC, CheckKind.GENERATED_TEST}:
                continue
            self.store.append_verification_event(
                "verification_check_started",
                project_id=contract.project_id,
                task_id=contract.task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                check_id=check.check_id if check.visibility == CheckVisibility.PUBLIC else None,
            )
            if check.visibility == CheckVisibility.HIDDEN:
                self.store.append_private_verification_event(
                    "verification_check_started",
                    project_id=contract.project_id,
                    task_id=contract.task_id,
                    contract_id=contract.contract_id,
                    contract_hash=contract.contract_hash,
                    check_id=check.check_id,
                )
            result = self.runner.run_check(check, root, kind)
            results.append(result)
            self.store.append_verification_event(
                "verification_check_finished",
                project_id=contract.project_id,
                task_id=contract.task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                check_id=check.check_id if check.visibility == CheckVisibility.PUBLIC else None,
                sanitized_reason=("check passed" if result.status == ExecutionStatus.PASSED else "check category failed"),
            )
            if check.visibility == CheckVisibility.HIDDEN:
                self.store.append_private_verification_event(
                    "verification_check_finished",
                    project_id=contract.project_id,
                    task_id=contract.task_id,
                    contract_id=contract.contract_id,
                    contract_hash=contract.contract_hash,
                    check_id=check.check_id,
                    verdict=result.status.value,
                    artifact_paths=[path for path in [result.stdout_artifact, result.stderr_artifact] if path],
                )
        return results

    def _validate_candidates(self, candidates: list[TestCandidate], candidate_root: Path) -> list[TestCandidate]:
        validator = TestCandidateValidator(self.snapshot_manager, self.runner)
        validated = []
        for candidate in candidates:
            result = (
                candidate
                if candidate.transition is not None and candidate.baseline_result is not None and candidate.candidate_result is not None
                else validator.validate(candidate, candidate_root)
            )
            self.store.save_test_candidate(result)
            if result is not candidate:
                self._record_candidate_validation(result)
            validated.append(result)
        return validated

    def validate_test_candidate(self, candidate: TestCandidate) -> TestCandidate:
        candidate_root, _manifest = self.snapshot_manager.create_candidate()
        try:
            result = TestCandidateValidator(self.snapshot_manager, self.runner).validate(candidate, candidate_root)
        finally:
            self.snapshot_manager.cleanup(candidate_root)
        self.store.save_test_candidate(result)
        self._record_candidate_validation(result)
        return result

    def _record_candidate_validation(self, result: TestCandidate) -> None:
        self.store.append_verification_event(
            "test_candidate_validated",
            task_id=result.task_id,
            session_id=result.session_id,
            sanitized_reason=(result.transition.value if result.transition else "invalid"),
            verdict=result.transition.value if result.transition else "invalid",
            valid=result.valid,
            rejection_category=result.rejection_reasons[0] if result.rejection_reasons else None,
            evidence_ids=[result.candidate_id],
        )

    def _invalid_contract(self, contract: VerificationContract, task_id: str | None, reason: str) -> VerificationReport:
        return VerificationReport(
            purpose=self.purpose,
            project_id=contract.project_id,
            task_id=task_id,
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            verdict=VerificationVerdict.CONTRACT_INVALID,
            sanitized_feedback=reason,
        )
