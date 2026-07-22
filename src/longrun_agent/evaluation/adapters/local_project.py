from __future__ import annotations

import json
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
from longrun_agent.evaluation.tool_calls import summarize_tool_calls
from longrun_agent.evaluation.workspace_artifacts import preserve_final_workspace_artifacts
from longrun_agent.model.base import ModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.state.schema import TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.tools.sandbox import build_subprocess_sandbox
from longrun_agent.tools.workspace_policy import WorkspaceAccessPolicy
from longrun_agent.verification.contract import load_contract, private_marker_registry, split_contract
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
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(case.fixture, workspace, ignore=shutil.ignore_patterns(".git", ".runs", "__pycache__"))
        if case.contract_path is not None:
            source = load_contract(case.contract_path, workspace_root=workspace)
            public, private = split_contract(source)
            public_path = workspace / ".longrun" / "agent_contract.json"
            public_path.parent.mkdir(parents=True, exist_ok=True)
            public_path.write_text(public.model_dump_json(indent=2), encoding="utf-8")
            private_path = descriptor.trial_dir / "oracle" / "private" / "contract.json"
            private_path.parent.mkdir(parents=True, exist_ok=True)
            private_path.write_text(private.model_dump_json(indent=2), encoding="utf-8")

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
        public_path = self.workspace(case, descriptor) / ".longrun" / "agent_contract.json"
        return public_path if public_path.exists() else None

    def run_agent(self, case: EvaluationTaskCase, config_path: Path, seed: int, descriptor: TrialDescriptor):
        config = load_config(config_path)
        config.workspace.root = self.workspace(case, descriptor)
        config.state.root = descriptor.trial_dir / "state"
        config.telemetry.run_root = descriptor.trial_dir / "telemetry"
        config.knowledge.root = descriptor.shared_knowledge_root or descriptor.trial_dir / "knowledge"
        config.verification.store_root = config.state.root
        private_path = descriptor.trial_dir / "oracle" / "private" / "contract.json"
        if private_path.exists():
            from longrun_agent.verification.schema import OraclePrivateContract

            private = OraclePrivateContract.model_validate_json(private_path.read_text(encoding="utf-8"))
            config.verification.contract.path = self.workspace(case, descriptor) / ".longrun" / "agent_contract.json"
            config.evaluation.denied_roots = [
                descriptor.trial_dir / "oracle",
                descriptor.trial_dir / "state",
                case.contract_path.parent if case.contract_path else descriptor.trial_dir / "oracle",
                *[
                    path
                    for path in descriptor.trial_dir.parent.iterdir()
                    if path.is_dir() and path.resolve() != descriptor.trial_dir.resolve()
                ],
            ]
            config.evaluation.private_markers = private_marker_registry(private)
            config.evaluation.private_audit_path = descriptor.trial_dir / "oracle" / "private" / "tool_result_blocks.jsonl"
        config.evaluation.enabled = True
        config.evaluation.isolation_enabled = config.model.provider != "fake"
        if config.evaluation.isolation_enabled:
            policy = WorkspaceAccessPolicy.for_workspace(
                config.workspace.root,
                evaluation_isolation_enabled=True,
                denied_roots=config.evaluation.denied_roots,
                private_markers=config.evaluation.private_markers,
                private_audit_path=config.evaluation.private_audit_path,
            )
            sandbox = build_subprocess_sandbox(policy)
            sandbox.preflight()
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
        verification_store = VerificationStore(state_root, outcome.project_id, workspace_root=self.workspace(case, descriptor))
        test_candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in [
                *verification_store.list_test_candidates(),
                *[candidate for report in reports for candidate in report.test_candidates],
            ]
        }
        test_candidates = list(test_candidates_by_id.values())
        workspace_artifacts = preserve_final_workspace_artifacts(descriptor.trial_dir, self.workspace(case, descriptor))
        tool_call_summary = summarize_tool_calls(
            sorted((descriptor.trial_dir / "telemetry").rglob("events.jsonl")) if (descriptor.trial_dir / "telemetry").exists() else []
        )
        tool_calls_path = descriptor.trial_dir / "artifacts" / "tool_calls.json"
        tool_calls_path.write_text(json.dumps(tool_call_summary, indent=2, sort_keys=True), encoding="utf-8")
        termination_reason = termination_reason_from_status(state.status.value)
        if termination_reason == TerminationReason.UNKNOWN:
            for session in reversed(sessions):
                session_reason = termination_reason_from_status(str(session.get("run_status") or ""))
                if session_reason in {
                    TerminationReason.TASK_LIMIT,
                    TerminationReason.TIME_LIMIT,
                    TerminationReason.SESSION_LIMIT,
                    TerminationReason.CONTEXT_LIMIT,
                    TerminationReason.PROVIDER_ERROR,
                    TerminationReason.INVALID_FORMAT,
                }:
                    termination_reason = session_reason
                    break
                if session_reason == TerminationReason.COMPLETED and state.status.value == "active":
                    termination_reason = session_reason
                    break
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
            tool_calls=len(tool_call_summary),
            sessions=len(sessions),
            context_resets=sum(int(session.get("context_reset_count") or 0) for session in sessions),
            plan_revisions=len(state.revisions),
            memory_uses=sum(int(session.get("memories_referenced") or 0) for session in sessions),
            skill_uses=sum(int(session.get("skills_referenced") or 0) for session in sessions),
            progress_snapshots=[progress],
            artifact_paths=[
                str(store.state_path(outcome.project_id)),
                *[str(state_root / outcome.project_id / "verification" / "reports" / f"{item.report_id}.json") for item in reports],
                *[str(path) for path in workspace_artifacts],
                str(tool_calls_path),
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
