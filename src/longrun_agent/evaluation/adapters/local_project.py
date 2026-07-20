from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from longrun_agent.config import load_config
from longrun_agent.evaluation.failure_taxonomy import termination_reason_from_status
from longrun_agent.evaluation.oracle import OfflineOracleEvaluator
from longrun_agent.evaluation.schema import (
    AdapterVerificationResult,
    EvaluationOutcome,
    EvaluationTaskCase,
    ProgressSnapshot,
    TerminationReason,
    TrialDescriptor,
)
from longrun_agent.model.base import ModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.state.schema import TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.schema import VerificationPurpose, VerificationReport
from longrun_agent.verification.store import VerificationStore


class LocalProjectAdapter:
    def __init__(self, provider_factory: Callable[[object, EvaluationTaskCase, int], ModelProvider]):
        self.provider_factory = provider_factory

    def prepare(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None:
        if case.fixture is None:
            raise ValueError("local project case requires fixture")
        workspace = descriptor.trial_dir / "workspace"
        descriptor.trial_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(case.fixture, workspace, ignore=shutil.ignore_patterns(".git", ".runs", "__pycache__"))

    def reset(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None:
        script = self.workspace(case, descriptor) / "reset_repo.py"
        if script.exists():
            subprocess.run([_python(), str(script)], cwd=script.parent, shell=False, check=True)
        OfflineOracleEvaluator().prepare_baseline(
            case=case,
            descriptor=descriptor,
            workspace=self.workspace(case, descriptor),
        )

    def objective(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> str:
        path = self.workspace(case, descriptor) / (case.task_file or Path("TASK.md"))
        return path.read_text(encoding="utf-8")

    def workspace(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> Path:
        return descriptor.trial_dir / "workspace"

    def verification_contract(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> Path | None:
        return case.contract_path

    def run_agent(self, case: EvaluationTaskCase, config_path: Path, seed: int, descriptor: TrialDescriptor):
        config = load_config(config_path)
        config.workspace.root = self.workspace(case, descriptor)
        config.state.root = descriptor.trial_dir / "state"
        config.telemetry.run_root = descriptor.trial_dir / "telemetry"
        config.knowledge.root = descriptor.shared_knowledge_root or descriptor.trial_dir / "knowledge"
        config.verification.store_root = config.state.root
        if case.contract_path:
            config.verification.contract.path = case.contract_path
        provider = self.provider_factory(config, case, seed)
        project_id = f"{descriptor.trial_id}-{case.case_id}"
        return ProjectOrchestrator(config, provider, project_id=project_id).start(self.objective(case, descriptor))

    def verify(self, case: EvaluationTaskCase, outcome, descriptor: TrialDescriptor) -> AdapterVerificationResult:
        runtime_reports = _reports(
            descriptor.trial_dir / "state",
            outcome.project_id,
            self.workspace(case, descriptor),
            purpose=VerificationPurpose.RUNTIME,
        )
        runtime_report = runtime_reports[-1] if runtime_reports else None
        result = OfflineOracleEvaluator().evaluate(
            case=case,
            descriptor=descriptor,
            project_id=outcome.project_id,
            final_workspace=self.workspace(case, descriptor),
        )
        return result.model_copy(
            update={
                "runtime_report_id": runtime_report.report_id if runtime_report else None,
                "runtime_verdict": runtime_report.verdict.value if runtime_report else None,
            }
        )

    def collect_artifacts(
        self,
        case: EvaluationTaskCase,
        outcome,
        verification: AdapterVerificationResult,
        descriptor: TrialDescriptor,
    ) -> EvaluationOutcome:
        state_root = descriptor.trial_dir / "state"
        store = ProjectStateStore(state_root, workspace_root=self.workspace(case, descriptor))
        state = store.load(outcome.project_id)
        sessions = store.read_sessions(outcome.project_id)
        events = store.read_events(outcome.project_id)
        reports = _reports(
            state_root,
            outcome.project_id,
            self.workspace(case, descriptor),
            purpose=VerificationPurpose.RUNTIME,
        )
        test_candidates = [candidate for report in reports for candidate in report.test_candidates]
        termination_reason = termination_reason_from_status(state.status.value)
        if verification.oracle_verdict in {"contract_invalid", "infrastructure_error", "inconclusive"}:
            termination_reason = {
                "contract_invalid": TerminationReason.CONTRACT_INVALID,
                "infrastructure_error": TerminationReason.ENVIRONMENT_ERROR,
                "inconclusive": TerminationReason.VERIFICATION_INCONCLUSIVE,
            }[verification.oracle_verdict]
        progress = _oracle_progress(verification, project_session=len(sessions))
        return EvaluationOutcome(
            evaluation_id=descriptor.evaluation_id,
            case_id=case.case_id,
            config_id=descriptor.config_id,
            trial_id=descriptor.trial_id,
            seed=descriptor.seed,
            project_id=outcome.project_id,
            project_status=state.status.value,
            verification_verdict=verification.oracle_verdict,
            runtime_verification_verdict=verification.runtime_verdict,
            runtime_verification_report_id=verification.runtime_report_id,
            oracle_verification_verdict=verification.oracle_verdict,
            oracle_verification_report_id=verification.oracle_report_id,
            oracle_total_requirements=verification.oracle_total_requirements,
            oracle_verified_requirements=verification.oracle_verified_requirements,
            oracle_verifier_seconds=verification.oracle_verifier_seconds,
            termination_reason=termination_reason,
            full_resolution=verification.oracle_verdict == "verified",
            partial_resolution=verification.oracle_partial_resolution,
            f2p_rate=verification.oracle_f2p_rate,
            p2p_rate=verification.oracle_p2p_rate,
            integrity_passed=verification.oracle_integrity_passed,
            completion_requests=sum(event.get("event_type") == "task_completion_requested" for event in events),
            false_completion_requests=sum(event.get("event_type") == "task_reopened" for event in events),
            task_verified_count=sum(task.status == TaskStatus.VERIFIED for task in state.tasks),
            task_reopened_count=sum(task.reopen_count for task in state.tasks),
            wall_clock_seconds=sum(float(session.get("duration_seconds") or 0) for session in sessions),
            input_tokens=sum(int(session.get("input_tokens_total") or 0) for session in sessions),
            output_tokens=sum(int(session.get("output_tokens_total") or 0) for session in sessions),
            compactor_tokens=sum(
                int(session.get("compactor_input_tokens") or 0) + int(session.get("compactor_output_tokens") or 0) for session in sessions
            ),
            knowledge_tokens=sum(int(session.get("knowledge_tokens_injected") or 0) for session in sessions),
            verifier_seconds=sum(
                result.duration_seconds for report in reports for result in [*report.baseline_results, *report.candidate_results]
            ),
            tool_calls=sum(int(session.get("tool_call_count") or 0) for session in sessions),
            sessions=len(sessions),
            context_resets=sum(int(session.get("context_reset_count") or 0) for session in sessions),
            plan_revisions=len(state.revisions),
            memory_uses=sum(int(session.get("memories_referenced") or 0) for session in sessions),
            skill_uses=sum(int(session.get("skills_referenced") or 0) for session in sessions),
            progress_snapshots=[progress],
            artifact_paths=[
                str(store.state_path(outcome.project_id)),
                *[str(state_root / outcome.project_id / "verification" / "reports" / f"{item.report_id}.json") for item in reports],
            ],
            test_candidates=len(test_candidates),
            well_formed_test_candidates=sum(item.valid for item in test_candidates),
            f2p_tests=sum(item.transition is not None and item.transition.value == "F2P" for item in test_candidates),
            p2p_irrelevant_tests=sum(item.valid_but_irrelevant for item in test_candidates),
            harmful_tests=sum("harmful_test" in item.rejection_reasons for item in test_candidates),
        )

    def cleanup(self, case: EvaluationTaskCase, descriptor: TrialDescriptor) -> None:
        return None


def _reports(
    state_root: Path,
    project_id: str,
    workspace: Path,
    *,
    purpose: VerificationPurpose,
) -> list[VerificationReport]:
    store = VerificationStore(state_root, project_id, workspace_root=workspace)
    return store.list_reports(purpose=purpose)


def _oracle_progress(verification: AdapterVerificationResult, *, project_session: int) -> ProgressSnapshot:
    passed = [f"requirement-{index}" for index in range(1, verification.oracle_verified_requirements + 1)]
    failed = [
        f"requirement-{index}" for index in range(verification.oracle_verified_requirements + 1, verification.oracle_total_requirements + 1)
    ]
    score = (
        verification.oracle_verified_requirements / verification.oracle_total_requirements
        if verification.oracle_total_requirements
        else 0.0
    )
    return ProgressSnapshot(
        project_session=project_session,
        passed_milestones=passed,
        failed_milestones=failed,
        score=score,
        source_report_id=verification.oracle_report_id,
    )


def _python() -> str:
    import sys

    return sys.executable
