from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import AppConfig
from longrun_agent.context.lifecycle import ContextLifecycleManager
from longrun_agent.control.channel import ControlSignalType, TaskControlChannel
from longrun_agent.control.tools import control_tools
from longrun_agent.knowledge.consolidator import KnowledgeConsolidator, KnowledgeSessionOutcome
from longrun_agent.knowledge.evidence import RepositoryProfiler, build_experience_pack
from longrun_agent.knowledge.renderer import render_bundle
from longrun_agent.knowledge.retrieval import retrieve_bundle
from longrun_agent.knowledge.schema import ExperienceEvidenceItem, KnowledgeRetrievalQuery, KnowledgeUseType
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.knowledge.tools import KnowledgeUseChannel, ReportKnowledgeUseTool
from longrun_agent.model.base import ModelProvider
from longrun_agent.orchestration.outcome import ProjectRunOutcome
from longrun_agent.orchestration.session_prompt import build_task_context_seed, build_task_session_prompt
from longrun_agent.orchestration.session_trace import SessionTrace
from longrun_agent.planning.decomposer import AsNeededDecomposer
from longrun_agent.planning.initial_planner import InitialPlanner
from longrun_agent.planning.recovery_evaluator import RecoveryCandidateEvaluator
from longrun_agent.planning.recovery_generator import RecoveryCandidateGenerator
from longrun_agent.protocol import ErrorType, RunResult, RunStatus, ToolResult
from longrun_agent.state.aggregation import aggregate_candidate_complete_parents, aggregate_verified_parents, project_statistics
from longrun_agent.state.schema import CompletionCandidate, PlanRevision, ProjectState, ProjectStatus, TaskNode, TaskStatus, utc_now
from longrun_agent.state.selector import TaskSelector
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.state.transitions import StateTransitionController
from longrun_agent.telemetry.project_logger import ProjectLogger
from longrun_agent.tools.arguments import render_command
from longrun_agent.tools.bash import BashArgs
from longrun_agent.tools.router import ToolRouter
from longrun_agent.verification.contract import load_contract
from longrun_agent.verification.gateway import VerificationGateway
from longrun_agent.verification.renderer import render_agent_feedback
from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import (
    CheckExecutionResult,
    CheckKind,
    CheckVisibility,
    ExecutionStatus,
    VerificationCheck,
    VerificationContract,
    VerificationPurpose,
    VerificationReport,
    VerificationSummary,
    VerificationVerdict,
)
from longrun_agent.verification.snapshot import CopySnapshotProvider, GitWorktreeSnapshotProvider
from longrun_agent.verification.store import VerificationStore


class ProjectOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        model: ModelProvider,
        *,
        store: ProjectStateStore | None = None,
        project_id: str | None = None,
    ):
        self.config = config
        self.model = model
        self.project_id = project_id or str(uuid.uuid4())
        self.store = store or ProjectStateStore(
            config.state.root, workspace_root=config.workspace.root, atomic_write=config.state.atomic_write
        )
        self.transitions = StateTransitionController()
        self.selector = TaskSelector(
            self.transitions,
            dependency_satisfaction=config.verification.policy.dependency_satisfaction,
        )
        self._last_final_verification_exit_code: int | None = None
        self.verification_store: VerificationStore | None = None
        self.verification_contract: VerificationContract | None = None
        self._latest_verification_report: VerificationReport | None = None
        self.knowledge_store = (
            KnowledgeStore(
                config.knowledge.root,
                workspace_root=config.workspace.root,
                atomic_write=config.state.atomic_write,
                record_mutation_policy=config.knowledge.record_mutation_policy,
            )
            if config.knowledge.mode != "disabled"
            else None
        )

    def start(self, objective: str) -> ProjectRunOutcome:
        if self.store.exists(self.project_id):
            state = self.store.load(self.project_id)
        else:
            state = ProjectState(project_id=self.project_id, objective=objective)
            self.store.create(state)
            self._logger(state).log("project_created", project_id=state.project_id, plan_version=state.plan_version)
        self._initialize_verification(state)
        if self._verification_initialization_failed():
            return self._outcome(state, 0, [])
        return self.run_project(state)

    def resume(self, project_id: str) -> ProjectRunOutcome:
        self.project_id = project_id
        state = self.store.load(project_id)
        self._initialize_verification(state)
        if self._verification_initialization_failed():
            return self._outcome(state, 0, [])
        if state.status in {ProjectStatus.SESSION_LIMIT_REACHED, ProjectStatus.TIME_LIMIT_REACHED, ProjectStatus.FAILED}:
            state.status = ProjectStatus.ACTIVE
            state.updated_at = utc_now()
            for task in state.tasks:
                if task.status == TaskStatus.FAILED and len(task.session_ids) < self.config.planning.execution.max_sessions_per_task:
                    task.status = TaskStatus.READY
                    task.updated_at = utc_now()
        self._logger(state).log("project_resumed", project_id=state.project_id, plan_version=state.plan_version)
        return self.run_project(state)

    def _initialize_verification(self, state: ProjectState) -> None:
        if self.config.verification.mode != "contract":
            return
        assert self.config.verification.store_root is not None
        self.verification_store = VerificationStore(
            self.config.verification.store_root,
            state.project_id,
            workspace_root=self.config.workspace.root,
            atomic_write=self.config.state.atomic_write,
        )
        stored_contract_id = state.project_verification_contract_id or next(
            (task.verification_contract_id for task in state.tasks if task.verification_contract_id),
            None,
        )
        if stored_contract_id:
            contract = self.verification_store.load_contract(stored_contract_id)
            if not self.verification_store.verify_contract_hash(contract):
                self.verification_contract = contract
                report = self._verification_gateway().verify(contract)
                self._latest_verification_report = report
                state.status = ProjectStatus.VERIFICATION_INCONCLUSIVE
                state.latest_project_verification_report_id = report.report_id
                self.store.save(state)
                return
        else:
            assert self.config.verification.contract.path is not None
            contract = load_contract(self.config.verification.contract.path, workspace_root=self.config.workspace.root)
            if contract.project_id == "__PROJECT_ID__":
                contract = contract.model_copy(update={"project_id": state.project_id})
            if contract.project_id != state.project_id:
                raise ValueError(f"verification contract project_id {contract.project_id!r} does not match project {state.project_id!r}")
            contract = contract.freeze()
            self.verification_store.save_contract(contract)
            if contract.scope == "project":
                state.project_verification_contract_id = contract.contract_id
            else:
                task = next(
                    (
                        item
                        for item in state.tasks
                        if item.id == contract.task_id or (contract.task_key is not None and item.key == contract.task_key)
                    ),
                    None,
                )
                if task is not None:
                    task.verification_contract_id = contract.contract_id
            self.verification_store.append_verification_event(
                "verification_contract_frozen",
                project_id=state.project_id,
                task_id=contract.task_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
            )
            self.store.save(state)
        self.verification_contract = contract
        manager = self._snapshot_manager()
        if not manager.baseline_manifest_path.exists():
            manifest = manager.create_baseline()
            self.verification_store.append_verification_event(
                "baseline_snapshot_created",
                project_id=state.project_id,
                contract_id=contract.contract_id,
                contract_hash=contract.contract_hash,
                artifact_paths=[str(manager.baseline_manifest_path)],
                sanitized_reason=f"baseline fingerprint {manifest.fingerprint}",
            )

    def _verification_initialization_failed(self) -> bool:
        return bool(
            self._latest_verification_report is not None
            and self._latest_verification_report.verdict == VerificationVerdict.CONTRACT_INVALID
        )

    def _snapshot_manager(self):
        assert self.verification_store is not None
        provider = GitWorktreeSnapshotProvider if self.config.verification.execution.isolation == "git_worktree" else CopySnapshotProvider
        return provider(
            self.config.workspace.root,
            self.verification_store.root,
            cache_patterns=self.config.verification.execution.cache_patterns,
        )

    def _verification_gateway(self) -> VerificationGateway:
        assert self.verification_store is not None
        return VerificationGateway(
            store=self.verification_store,
            snapshot_manager=self._snapshot_manager(),
            runner=VerificationRunner(
                self.verification_store.root / "artifacts",
                max_output_chars=self.config.verification.execution.max_output_chars,
            ),
            preserve_failed_snapshot=self.config.verification.execution.preserve_failed_snapshot,
        )

    def _contract_for_task(self, task: TaskNode) -> VerificationContract | None:
        contract = self.verification_contract
        if contract is None or contract.scope != "task":
            return None
        if contract.task_id == task.id or (contract.task_key is not None and contract.task_key == task.key):
            return contract
        return None

    def _verify_task_candidate(
        self,
        state: ProjectState,
        task: TaskNode,
        channel: TaskControlChannel | None,
    ) -> VerificationReport | None:
        contract = self._contract_for_task(task)
        retrying_infrastructure = task.status == TaskStatus.BLOCKED and task.verification_status == "infrastructure_error"
        if contract is None or (task.status != TaskStatus.CANDIDATE_COMPLETE and not retrying_infrastructure):
            return None
        self._transition(
            state,
            task.id,
            TaskStatus.VERIFICATION_PENDING,
            reason="completion candidate awaiting independent verification",
            source="verification",
        )
        task.verification_attempts += 1
        test_candidates = channel.test_candidates if channel is not None else self.verification_store.list_test_candidates()
        report = self._verification_gateway().verify(contract, task_id=task.id, test_candidates=test_candidates)
        self._latest_verification_report = report
        task.latest_verification_report_id = report.report_id
        task.verification_status = report.verdict.value
        if report.verdict == VerificationVerdict.VERIFIED:
            self._transition(state, task.id, TaskStatus.VERIFIED, reason="verification contract passed", source="verification")
            task.verified_at = utc_now()
            task.verified_contract_hash = report.contract_hash
            self._logger(state).log(
                "task_verified",
                project_id=state.project_id,
                task_id=task.id,
                plan_version=state.plan_version,
                payload={"report_id": report.report_id, "verdict": report.verdict.value},
            )
        elif report.verdict in {VerificationVerdict.PARTIAL, VerificationVerdict.REOPENED}:
            self._reopen_task_after_verification(state, task, report)
        elif report.verdict == VerificationVerdict.INFRASTRUCTURE_ERROR:
            task.progress_notes.append(report.sanitized_feedback)
            self._transition(
                state,
                task.id,
                TaskStatus.BLOCKED,
                reason="verification infrastructure unavailable; resume permitted",
                source="verification",
            )
        else:
            self._transition(
                state,
                task.id,
                TaskStatus.BLOCKED,
                reason=report.sanitized_feedback or "verification inconclusive",
                source="verification",
            )
        return report

    def _persist_test_candidates(self, state: ProjectState, channel: TaskControlChannel) -> None:
        if self.verification_store is None:
            return
        for candidate in channel.test_candidates:
            self.verification_store.save_test_candidate(candidate)
            self.verification_store.append_verification_event(
                "test_candidate_registered",
                project_id=state.project_id,
                task_id=candidate.task_id,
                session_id=candidate.session_id,
                contract_id=self.verification_contract.contract_id if self.verification_contract else None,
                contract_hash=self.verification_contract.contract_hash if self.verification_contract else None,
                sanitized_reason="Agent-authored test candidate registered for independent validation",
                evidence_ids=[candidate.candidate_id],
            )

    def _reopen_task_after_verification(self, state: ProjectState, task: TaskNode, report: VerificationReport) -> None:
        task.reopen_count += 1
        task.progress_notes.append(render_agent_feedback(report))
        self._transition(state, task.id, TaskStatus.REOPENED, reason="verification requirements not met", source="verification")
        self._logger(state).log(
            "task_reopened",
            project_id=state.project_id,
            task_id=task.id,
            plan_version=state.plan_version,
            reason=report.sanitized_feedback,
            payload={"report_id": report.report_id, "verdict": report.verdict.value},
        )
        if task.reopen_count > self.config.verification.policy.max_task_reopens:
            self._transition(state, task.id, TaskStatus.BLOCKED, reason="maximum verification reopens reached", source="verification")
        else:
            task.completion_candidate = None
            task.completion_summary = None
            self._transition(state, task.id, TaskStatus.READY, reason="continue after verification feedback", source="verification")

    def run_project(self, state: ProjectState) -> ProjectRunOutcome:
        project_started = time.monotonic()
        project_deadline = project_started + self.config.planning.execution.max_project_seconds
        run_statuses = []
        if self._project_time_exhausted(project_deadline):
            self._mark_project_time_limit(state, project_started)
            return self._outcome(state, 0, run_statuses)
        if not state.tasks and self.config.planning.mode != "disabled":
            self._create_initial_plan(state)
        if (
            state.status == ProjectStatus.VERIFICATION_PENDING
            and self.verification_contract is not None
            and self.verification_contract.scope == "project"
        ):
            self._finalize_contract_verification(state, project_started)
            return self._outcome(state, 0, run_statuses)
        for task in state.tasks:
            if task.status == TaskStatus.BLOCKED and task.verification_status == "infrastructure_error":
                self._verify_task_candidate(state, task, None)
        if self._project_time_exhausted(project_deadline):
            self._mark_project_time_limit(state, project_started)
            return self._outcome(state, 0, run_statuses)
        sessions_run = 0
        while state.session_count < self.config.planning.execution.max_project_sessions:
            if self._project_time_exhausted(project_deadline):
                self._mark_project_time_limit(state, project_started)
                break
            for parent_id in aggregate_candidate_complete_parents(state):
                self._logger(state).log(
                    "parent_task_aggregated", project_id=state.project_id, task_id=parent_id, plan_version=state.plan_version
                )
            for parent_id in aggregate_verified_parents(state):
                self._logger(state).log(
                    "parent_task_verified", project_id=state.project_id, task_id=parent_id, plan_version=state.plan_version
                )
            task = self.selector.select_next(state)
            self.store.save(state)
            if task is None:
                if state.status == ProjectStatus.CANDIDATE_COMPLETE:
                    state.status = ProjectStatus.ACTIVE
                    self._finalize_candidate_complete(state, project_started)
                elif state.status == ProjectStatus.BLOCKED:
                    self._logger(state).log("project_blocked", project_id=state.project_id, plan_version=state.plan_version)
                    self.store.save(state)
                    self._write_metrics(state, project_started)
                break
            if len(task.session_ids) >= self.config.planning.execution.max_sessions_per_task:
                self._transition(state, task.id, TaskStatus.FAILED, reason="task session limit reached", source="orchestrator")
                state.status = ProjectStatus.FAILED
                state.updated_at = utc_now()
                self._logger(state).log(
                    "task_session_limit_reached",
                    project_id=state.project_id,
                    task_id=task.id,
                    plan_version=state.plan_version,
                    payload={"max_sessions_per_task": self.config.planning.execution.max_sessions_per_task},
                )
                self.store.save(state)
                self._write_metrics(state, project_started)
                break
            if task.status == TaskStatus.READY:
                self._transition(state, task.id, TaskStatus.IN_PROGRESS, reason="selected", source="orchestrator")
            state.session_count += 1
            sessions_run += 1
            task.attempts += 1
            session_id = f"{state.project_id}-s{state.session_count}"
            task.session_ids.append(session_id)
            self._logger(state).log(
                "task_started", project_id=state.project_id, task_id=task.id, session_id=session_id, plan_version=state.plan_version
            )
            channel = TaskControlChannel(
                workspace=self.config.workspace.root,
                task_id=task.id,
                session_id=session_id,
                verification_contract=self._contract_for_task(task),
                max_test_candidates=(
                    self.config.verification.generated_tests.max_candidates_per_task if self.config.verification.mode == "contract" else 0
                ),
            )
            starting_task_status = task.status.value
            result, trace, knowledge_channel, knowledge_bundle = self._run_task_session(state, task, channel, session_id, project_deadline)
            run_statuses.append(result.status)
            self._process_control_signals(state, task, channel, session_id, result, trace)
            self._persist_test_candidates(state, channel)
            session_record = self._session_record(state, task, session_id, result, channel, trace)
            verification_report = self._verify_task_candidate(state, task, channel)
            project_verification_attempted = False
            if (
                verification_report is None
                and self.verification_contract is not None
                and self.verification_contract.scope == "project"
                and state.leaf_tasks()
                and all(item.status in {TaskStatus.CANDIDATE_COMPLETE, TaskStatus.VERIFIED} for item in state.leaf_tasks())
            ):
                self._finalize_contract_verification(state, project_started)
                verification_report = self._latest_verification_report
                project_verification_attempted = True
            self._process_knowledge_after_session(
                state,
                task,
                session_record,
                starting_task_status=starting_task_status,
                knowledge_channel=knowledge_channel,
                knowledge_bundle=knowledge_bundle,
                verification_report=verification_report,
            )
            self.store.append_session(state.project_id, session_record)
            self.store.save(state)
            self._write_metrics(state, project_started)
            if project_verification_attempted:
                break
            if state.status in {ProjectStatus.CANDIDATE_COMPLETE, ProjectStatus.BLOCKED, ProjectStatus.FAILED}:
                break
            if self._project_time_exhausted(project_deadline):
                self._mark_project_time_limit(state, project_started)
                break
        for parent_id in aggregate_candidate_complete_parents(state):
            self._logger(state).log(
                "parent_task_aggregated", project_id=state.project_id, task_id=parent_id, plan_version=state.plan_version
            )
        for parent_id in aggregate_verified_parents(state):
            self._logger(state).log("parent_task_verified", project_id=state.project_id, task_id=parent_id, plan_version=state.plan_version)
        leaves = state.leaf_tasks()
        if (
            state.status == ProjectStatus.ACTIVE
            and leaves
            and all(task.status in {TaskStatus.CANDIDATE_COMPLETE, TaskStatus.VERIFIED} for task in leaves)
        ):
            self._finalize_candidate_complete(state, project_started)
        elif state.session_count >= self.config.planning.execution.max_project_sessions and state.status == ProjectStatus.ACTIVE:
            state.status = ProjectStatus.SESSION_LIMIT_REACHED
            self._logger(state).log("project_session_limit_reached", project_id=state.project_id, plan_version=state.plan_version)
            self.store.save(state)
            self._write_metrics(state, project_started)
        else:
            self._write_metrics(state, project_started)
        return self._outcome(state, sessions_run, run_statuses)

    def _create_initial_plan(self, state: ProjectState) -> None:
        logger = self._logger(state)
        logger.log("initial_plan_requested", project_id=state.project_id, plan_version=state.plan_version)
        planner = InitialPlanner(self.model, self.config.planning.initial_plan)
        if self.config.planning.initial_plan.source == "file":
            tasks = planner.load_from_file(project_id=state.project_id)
            logger.log(
                "initial_plan_loaded_from_file",
                project_id=state.project_id,
                plan_version=state.plan_version,
                payload={"plan_file": str(self.config.planning.initial_plan.plan_file)},
            )
        else:
            tasks = planner.plan(project_id=state.project_id, objective=state.objective)
        state.tasks = tasks
        if self.verification_contract is not None and self.verification_contract.scope == "task":
            for task in state.tasks:
                if self.verification_contract.task_id == task.id or self.verification_contract.task_key == task.key:
                    task.verification_contract_id = self.verification_contract.contract_id
        state.plan_version += 1
        state.updated_at = utc_now()
        state.revisions.append(
            PlanRevision(trigger="initial_plan", reason="project start", added_task_ids=[task.id for task in tasks], superseded_task_ids=[])
        )
        logger.log(
            "initial_plan_generated",
            project_id=state.project_id,
            plan_version=state.plan_version,
            payload={"task_ids": [task.id for task in tasks]},
        )
        self.store.save(state)

    def _run_task_session(
        self,
        state: ProjectState,
        task: TaskNode,
        channel: TaskControlChannel,
        session_id: str,
        project_deadline: float,
    ):
        knowledge_context, knowledge_bundle = self._retrieve_knowledge_for_task(state, task, session_id)
        knowledge_channel = KnowledgeUseChannel(
            exposed_memory_ids=knowledge_bundle.primary_memory_ids,
            exposed_skill_ids=knowledge_bundle.primary_skill_ids,
        )
        knowledge_tools = [ReportKnowledgeUseTool()] if self.config.knowledge.mode != "disabled" else []
        router = ToolRouter(
            [
                *default_router().tools.values(),
                *control_tools(
                    generated_tests=(self.config.verification.mode == "contract" and self.config.verification.generated_tests.enabled)
                ),
                *knowledge_tools,
            ]
        )
        trace = SessionTrace()
        seed = build_task_context_seed(
            state,
            task,
            knowledge_context=knowledge_context,
            knowledge_retrieval_id=knowledge_bundle.retrieval_id if knowledge_context else None,
        )
        context_manager = ContextLifecycleManager(
            self.config.context,
            seed=seed,
            model=self.model,
            store=self.store,
            project_id=state.project_id,
            task_id=task.id,
            session_id=session_id,
            run_id=session_id,
            plan_version=state.plan_version,
            workspace_root=self.config.workspace.root,
            event_sink=lambda event_type, payload: self._log_session_event(state, task, session_id, event_type, payload),
        )
        loop = AgentLoop(
            self.config,
            self.model,
            router=router,
            run_id=session_id,
            on_event=lambda event_type, payload: self._log_session_event(state, task, session_id, event_type, payload),
        )
        router_with_channel = _ChannelRouter(
            router,
            channel,
            trace,
            on_suppressed=lambda call_key: self._logger(state).log(
                "repeated_tool_call_suppressed",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload={"call_key": call_key},
            ),
            on_unsupported_shell=lambda result: self._logger(state).log(
                "unsupported_shell_syntax",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload={"command": result.metadata.get("command"), "message": result.error_message},
            ),
            knowledge_channel=knowledge_channel,
        )
        loop.router = router_with_channel
        result = loop.run_with_controls(
            self.config.workspace.root,
            build_task_session_prompt(state, task, self.config),
            deadline_monotonic=project_deadline,
            stop_condition=lambda: channel.terminal_signal is not None,
            require_external_terminal=True,
            completion_evidence=lambda: trace.has_completion_evidence(existing_changed_files=task.files_touched),
            context_seed=seed,
            context_manager=context_manager,
            project_id=state.project_id,
            task_id=task.id,
            session_id=session_id,
        )
        if result.latest_context_handoff_id:
            task.latest_context_handoff_id = result.latest_context_handoff_id
        task.context_reset_count += result.context_reset_count
        task.context_compaction_count += result.structured_compaction_count
        if result.terminal_grace_turn_count:
            self._logger(state).log(
                "terminal_grace_turn_finished",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload={"terminal_signal_recovered": result.terminal_signal_recovered},
            )
        if result.terminal_signal_recovered:
            self._logger(state).log(
                "terminal_signal_recovered",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
            )
        return result, trace, knowledge_channel, knowledge_bundle

    def _retrieve_knowledge_for_task(self, state: ProjectState, task: TaskNode, session_id: str):
        if self.config.knowledge.mode == "disabled":
            from longrun_agent.knowledge.schema import RetrievedKnowledgeBundle

            return None, RetrievedKnowledgeBundle()
        assert self.knowledge_store is not None
        try:
            profile = RepositoryProfiler(self.config.workspace.root).profile()
            query = KnowledgeRetrievalQuery(
                task_objective=task.objective,
                acceptance_criteria=task.acceptance_criteria,
                repository_fingerprint=profile.repository_fingerprint,
                language_tags=profile.language_tags,
                framework_tags=profile.framework_tags,
                tool_tags=profile.tool_tags,
                project_id=state.project_id,
                blocker=task.blocker,
                recent_error_signatures=task.progress_notes[-3:],
            )
            bundle, _scores = retrieve_bundle(self.config.knowledge, self.knowledge_store, query)
            rendered, tokens = render_bundle(bundle, self.config.knowledge)
            bundle.total_estimated_tokens = tokens
            primary_memories = [memory for memory in bundle.memories if memory.memory_id in bundle.primary_memory_ids]
            primary_skills = [skill for skill in bundle.skills if skill.skill_id in bundle.primary_skill_ids]
            if primary_memories:
                for memory in primary_memories:
                    self.knowledge_store.add_memory_usage(
                        memory.memory_id,
                        KnowledgeUseType.EXPOSED,
                        project_id=state.project_id,
                        task_id=task.id,
                        session_id=session_id,
                        retrieval_id=bundle.retrieval_id,
                        reason="injected into task context",
                    )
                self.knowledge_store.append_event(
                    "memory_exposed",
                    project_id=state.project_id,
                    task_id=task.id,
                    retrieval_id=bundle.retrieval_id,
                    memory_id=[memory.memory_id for memory in primary_memories],
                    token_usage=tokens,
                )
            if primary_skills:
                for skill in primary_skills:
                    self.knowledge_store.add_skill_usage(
                        skill.skill_id,
                        KnowledgeUseType.EXPOSED,
                        project_id=state.project_id,
                        task_id=task.id,
                        session_id=session_id,
                        retrieval_id=bundle.retrieval_id,
                        reason="injected into task context",
                    )
                self.knowledge_store.append_event(
                    "skill_exposed",
                    project_id=state.project_id,
                    task_id=task.id,
                    retrieval_id=bundle.retrieval_id,
                    skill_id=[skill.skill_id for skill in primary_skills],
                    token_usage=tokens,
                )
            return rendered or None, bundle
        except Exception as exc:
            self.knowledge_store.append_event("knowledge_error", project_id=state.project_id, task_id=task.id, reason=str(exc))
            if self.config.knowledge.strict_errors:
                raise
            from longrun_agent.knowledge.schema import RetrievedKnowledgeBundle

            return None, RetrievedKnowledgeBundle()

    def _process_knowledge_after_session(
        self,
        state: ProjectState,
        task: TaskNode,
        session_record: dict,
        *,
        starting_task_status: str,
        knowledge_channel: KnowledgeUseChannel,
        knowledge_bundle,
        verification_report: VerificationReport | None = None,
    ) -> None:
        if self.config.knowledge.mode == "disabled":
            return
        assert self.knowledge_store is not None
        try:
            primary_memory_ids = list(getattr(knowledge_bundle, "primary_memory_ids", []))
            primary_skill_ids = list(getattr(knowledge_bundle, "primary_skill_ids", []))
            session_record["memories_retrieved"] = len(getattr(knowledge_bundle, "memories", []))
            session_record["memories_exposed"] = len(primary_memory_ids)
            session_record["skills_retrieved"] = len(getattr(knowledge_bundle, "skills", []))
            session_record["skills_exposed"] = len(primary_skill_ids)
            session_record["knowledge_tokens_injected"] = int(getattr(knowledge_bundle, "total_estimated_tokens", 0) or 0)
            for field in (
                "memories_referenced",
                "memories_helpful",
                "memories_harmful",
                "skills_referenced",
                "skills_helpful",
                "skills_harmful",
                "episodes_created",
                "reflection_candidates",
                "active_memories_created",
                "quarantined_memories",
                "skills_created",
                "skills_validated",
            ):
                session_record[field] = 0

            referenced_memory_ids = _dedupe([memory_id for record in knowledge_channel.records for memory_id in record.memory_ids])
            referenced_skill_ids = _dedupe([skill_id for record in knowledge_channel.records for skill_id in record.skill_ids])
            if knowledge_channel.decision_recorded and not knowledge_channel.records and knowledge_channel.not_used_reason:
                self.knowledge_store.append_event(
                    "knowledge_reviewed_not_used",
                    project_id=state.project_id,
                    task_id=task.id,
                    session_id=session_record.get("session_id"),
                    retrieval_id=getattr(knowledge_bundle, "retrieval_id", None),
                    memory_ids=sorted(knowledge_channel.exposed_memory_ids),
                    skill_ids=sorted(knowledge_channel.exposed_skill_ids),
                    reason=knowledge_channel.not_used_reason,
                )
            unreferenced_memory_ids = sorted(set(primary_memory_ids) - set(referenced_memory_ids))
            unreferenced_skill_ids = sorted(set(primary_skill_ids) - set(referenced_skill_ids))
            if unreferenced_memory_ids or unreferenced_skill_ids:
                self.knowledge_store.append_event(
                    "knowledge_exposed_but_not_referenced",
                    project_id=state.project_id,
                    task_id=task.id,
                    session_id=session_record.get("session_id"),
                    memory_ids=unreferenced_memory_ids,
                    skill_ids=unreferenced_skill_ids,
                )

            pack = build_experience_pack(
                project_id=state.project_id,
                task_id=task.id,
                task_objective=task.objective,
                acceptance_criteria=task.acceptance_criteria,
                session_record=session_record,
                plan_version=state.plan_version,
                starting_task_status=starting_task_status,
                ending_task_status=task.status.value,
                workspace_root=self.config.workspace.root,
                max_evidence_items=self.config.knowledge.episode.max_evidence_items,
            )
            if verification_report is not None:
                pack.verification_report_id = verification_report.report_id
                pack.verification_verdict = verification_report.verdict.value
                pack.failed_check_categories = sorted(
                    {
                        item.kind.value
                        for item in verification_report.transitions
                        if item.required and item.transition.value not in {"F2P", "P2P"}
                    }
                )
                pack.f2p_rate = verification_report.summary.f2p_rate
                pack.p2p_rate = verification_report.summary.p2p_rate
                pack.integrity_violations = [item.category for item in verification_report.integrity_violations]
                pack.infrastructure_error = verification_report.infrastructure_error
                if verification_report.verdict == VerificationVerdict.VERIFIED:
                    pack.successful_verifications.append(f"verification_report:{verification_report.report_id}")
                    pack.evidence_items.append(
                        ExperienceEvidenceItem(
                            evidence_id=f"{session_record.get('session_id')}:verification",
                            project_id=state.project_id,
                            task_id=task.id,
                            session_id=str(session_record.get("session_id") or ""),
                            run_id=str(session_record.get("run_id") or session_record.get("session_id") or ""),
                            event_type="successful_verification",
                            summary="Independent verification contract passed all required checks with integrity preserved.",
                            success=True,
                            timestamp=verification_report.created_at,
                        )
                    )
                elif verification_report.verdict in {VerificationVerdict.PARTIAL, VerificationVerdict.REOPENED}:
                    pack.failed_verifications.append(f"verification_report:{verification_report.report_id}:required_check_category_failed")
            attribution = _knowledge_attribution(task, pack)
            verified = task.status == TaskStatus.VERIFIED or (
                verification_report is not None and verification_report.verdict == VerificationVerdict.VERIFIED
            )
            if self.config.verification.mode != "contract":
                verified = task.status == TaskStatus.CANDIDATE_COMPLETE
            outcome = KnowledgeSessionOutcome(
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_record.get("session_id") or "",
                repository_fingerprint=pack.repository_fingerprint,
                referenced_memory_ids=referenced_memory_ids,
                referenced_skill_ids=referenced_skill_ids,
                attribution=attribution,
                verification_passed=(
                    verification_report.verdict == VerificationVerdict.VERIFIED
                    if verification_report is not None
                    else bool(pack.successful_verifications)
                ),
                candidate_complete=task.status == TaskStatus.CANDIDATE_COMPLETE,
                verified=verified,
                experience_pack=pack,
            )
            result = KnowledgeConsolidator(self.config.knowledge, self.knowledge_store, self.model).consolidate(outcome)

            session_record["memories_referenced"] = len(referenced_memory_ids)
            session_record["skills_referenced"] = len(referenced_skill_ids)
            if attribution == KnowledgeUseType.HELPFUL:
                session_record["memories_helpful"] = len(referenced_memory_ids)
                session_record["skills_helpful"] = len(referenced_skill_ids)
            elif attribution == KnowledgeUseType.HARMFUL:
                session_record["memories_harmful"] = len(referenced_memory_ids)
                session_record["skills_harmful"] = len(referenced_skill_ids)
            episode_path = self.knowledge_store.episode_path(self.store.project_dir(state.project_id), pack.episode_id)
            session_record["episodes_created"] = int(episode_path.exists())
            created_memories = [self.knowledge_store.load_memory(memory_id) for memory_id in result.created_memory_ids]
            session_record["reflection_candidates"] = len(created_memories)
            session_record["active_memories_created"] = sum(memory.status.value == "active" for memory in created_memories)
            session_record["quarantined_memories"] = sum(memory.status.value == "quarantined" for memory in created_memories)
            created_skills = [self.knowledge_store.load_skill(skill_id) for skill_id in result.created_skill_ids]
            session_record["skills_created"] = len(created_skills)
            session_record["skills_validated"] = sum(skill.status.value == "validated" for skill in created_skills)
        except Exception as exc:
            self.knowledge_store.append_event(
                "knowledge_error",
                project_id=state.project_id,
                task_id=task.id,
                reason=str(exc),
            )
            if self.config.knowledge.strict_errors:
                raise

    def _process_control_signals(
        self,
        state: ProjectState,
        task: TaskNode,
        channel: TaskControlChannel,
        session_id: str,
        result: RunResult,
        trace: SessionTrace,
    ) -> None:
        logger = self._logger(state)
        for signal in channel.progress_signals:
            task.progress_notes.append(signal.summary or "")
            task.files_touched.extend(signal.files_touched)
            logger.log(
                "task_progress",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload=signal.model_dump(),
            )
        task.files_touched = _dedupe([*task.files_touched, *trace.changed_files])
        task.read_files = _dedupe([*task.read_files, *trace.read_files])
        no_progress = trace.no_progress(progress_count=len(channel.progress_signals), terminal_signal=channel.terminal_signal)
        task.consecutive_no_progress_sessions = task.consecutive_no_progress_sessions + 1 if no_progress else 0
        terminal = channel.terminal_signal
        candidate = None
        if terminal is None or terminal.type == ControlSignalType.COMPLETION_REQUEST:
            candidate = self._completion_candidate_for(state, task, trace)
            if candidate is not None and task.completion_candidate is None:
                task.completion_candidate = candidate
                logger.log(
                    "completion_candidate_created",
                    project_id=state.project_id,
                    task_id=task.id,
                    session_id=session_id,
                    plan_version=state.plan_version,
                    payload=candidate.model_dump(mode="json"),
                )
        if terminal is None:
            logger.log(
                "session_ended_without_task_signal",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
            )
            if result.status in {
                RunStatus.MAX_STEPS_REACHED,
                RunStatus.TIME_LIMIT_REACHED,
                RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL,
                RunStatus.TERMINAL_SIGNAL_MISSING,
            }:
                summary = trace.handoff_summary(result)
                task.last_handoff_summary = summary
                task.progress_notes.append(summary)
                logger.log(
                    "session_handoff_created",
                    project_id=state.project_id,
                    task_id=task.id,
                    session_id=session_id,
                    plan_version=state.plan_version,
                    payload={"handoff_summary": summary, **trace.model_dump()},
                )
            if candidate is not None:
                self._internal_complete_task(state, task, session_id, candidate)
                return
            if (
                self.config.planning.mode in {"adaptive", "adaptive_search"}
                and task.attempts >= self.config.planning.execution.attempts_before_decomposition
            ):
                self._trigger_decomposition(state, task, "session ended without terminal signal")
            return
        if terminal.type == ControlSignalType.COMPLETION_REQUEST:
            task.completion_summary = terminal.summary
            self._transition(
                state, task.id, TaskStatus.CANDIDATE_COMPLETE, reason=terminal.summary or "completion requested", source="control"
            )
            logger.log(
                "task_completion_requested",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload=terminal.model_dump(),
            )
            if task.completion_candidate is not None:
                logger.log(
                    "completion_candidate_confirmed",
                    project_id=state.project_id,
                    task_id=task.id,
                    session_id=session_id,
                    plan_version=state.plan_version,
                    payload=task.completion_candidate.model_dump(mode="json"),
                )
        elif terminal.type == ControlSignalType.BLOCKER:
            task.blocker = terminal.reason
            logger.log(
                "task_blocked",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload=terminal.model_dump(),
            )
            if terminal.decomposition_recommended and self.config.planning.mode in {"adaptive", "adaptive_search"}:
                self._trigger_decomposition(state, task, terminal.reason or "blocked")
            else:
                self._transition(state, task.id, TaskStatus.BLOCKED, reason=terminal.reason or "blocked", source="control")
        elif terminal.type == ControlSignalType.DECOMPOSITION_REQUEST:
            logger.log(
                "task_decomposition_requested",
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload=terminal.model_dump(),
            )
            if self.config.planning.mode in {"adaptive", "adaptive_search"}:
                self._trigger_decomposition(state, task, terminal.reason or "decomposition requested")
            else:
                self._transition(state, task.id, TaskStatus.BLOCKED, reason="decomposition requested but mode is static", source="control")

    def _completion_candidate_for(self, state: ProjectState, task: TaskNode, trace: SessionTrace) -> CompletionCandidate | None:
        if task.blocker:
            return None
        changed_files = _dedupe([*task.files_touched, *trace.changed_files])
        successful_tests = list(trace.successful_test_commands)
        verification_commands = _dedupe([*trace.successful_acceptance_commands, *successful_tests])
        evidence: list[str] = []
        if changed_files and successful_tests:
            evidence.append("changed_files_and_successful_tests")
        if self._last_final_verification_exit_code == 0:
            evidence.append("final_verification_exit_code_0")
        if verification_commands:
            evidence.append("successful_bash_verification_exit_code_0")
        if not evidence:
            return None
        if not self._acceptance_criteria_satisfied(task, verification_commands):
            return None
        return CompletionCandidate(
            task_id=task.id,
            evidence=evidence,
            changed_files=changed_files,
            successful_tests=successful_tests,
            verification_commands=verification_commands,
        )

    def _acceptance_criteria_satisfied(self, task: TaskNode, verification_commands: list[str]) -> bool:
        if self.config.verification.mode == "contract":
            return bool(verification_commands)
        if self._last_final_verification_exit_code == 0:
            return True
        if not verification_commands:
            return False
        command_text = " ".join(verification_commands).lower()
        for criterion in task.acceptance_criteria:
            lowered = criterion.lower()
            if "pytest" in lowered and "pytest" not in command_text:
                return False
            if "validate" in lowered and "validate" not in command_text and "pytest" not in command_text:
                return False
            if ("cli" in lowered or " get " in f" {lowered} " or "get command" in lowered) and (
                "task_service.cli" not in command_text and " get " not in f" {command_text} "
            ):
                return False
        return True

    def _internal_complete_task(
        self,
        state: ProjectState,
        task: TaskNode,
        session_id: str,
        candidate: CompletionCandidate,
    ) -> None:
        task.completion_candidate = candidate
        task.completion_summary = "Auto completion recovered from deterministic verification evidence."
        task.auto_completion_recovered = True
        self._transition(
            state,
            task.id,
            TaskStatus.CANDIDATE_COMPLETE,
            reason=task.completion_summary,
            source="orchestrator",
        )
        self._logger(state).log(
            "auto_completion_recovered",
            project_id=state.project_id,
            task_id=task.id,
            session_id=session_id,
            plan_version=state.plan_version,
            payload=candidate.model_dump(mode="json"),
        )

    def _trigger_decomposition(self, state: ProjectState, task: TaskNode, reason: str) -> None:
        if self.config.planning.mode == "static":
            self._transition(state, task.id, TaskStatus.BLOCKED, reason=reason, source="orchestrator")
            return
        selected_candidate_id = None
        candidate_ids: list[str] = []
        selected_decompose_children: list[TaskNode] | None = None
        if self.config.planning.mode == "adaptive_search" and self.config.planning.bounded_search.enabled:
            candidates = RecoveryCandidateGenerator(self.model, self.config.planning.bounded_search).generate(task, reason)
            self._logger(state).log(
                "recovery_candidates_generated",
                project_id=state.project_id,
                task_id=task.id,
                plan_version=state.plan_version,
                candidate_ids=[candidate.id for candidate in candidates],
            )
            evaluator = RecoveryCandidateEvaluator(self.model, self.config.planning.bounded_search, self.config.planning.decomposition)
            valid = evaluator.filter_candidates(task, candidates)
            candidate_ids = [candidate.id for candidate in candidates]
            for _candidate_id, rejection in evaluator.rejections.items():
                self._logger(state).log(
                    "recovery_candidate_rejected",
                    project_id=state.project_id,
                    task_id=task.id,
                    plan_version=state.plan_version,
                    reason=rejection,
                )
            selection = evaluator.select(valid)
            selected_candidate_id = selection.selected_candidate_id
            self._logger(state).log(
                "recovery_candidate_selected",
                project_id=state.project_id,
                task_id=task.id,
                plan_version=state.plan_version,
                candidate_ids=candidate_ids,
                selected_candidate_id=selected_candidate_id,
                reason=selection.selection_reason,
            )
            selected = next(candidate for candidate in valid if candidate.id == selected_candidate_id)
            if selected.kind == "mark_blocked":
                self._transition(state, task.id, TaskStatus.BLOCKED, reason=selected.description, source="recovery")
                return
            if selected.kind == "retry_with_guidance":
                guidance = f"Recovery guidance: {selected.description}"
                task.progress_notes.append(guidance)
                task.blocker = None
                self._logger(state).log(
                    "recovery_guidance_recorded",
                    project_id=state.project_id,
                    task_id=task.id,
                    plan_version=state.plan_version,
                    selected_candidate_id=selected.id,
                    reason=selected.description,
                )
                self._transition(state, task.id, TaskStatus.READY, reason=selected.description, source="recovery")
                return
            if selected.kind == "decompose":
                selected_decompose_children = evaluator.validator.children_to_task_nodes(task, selected.child_tasks)
        children = selected_decompose_children or AsNeededDecomposer(self.model, self.config.planning.decomposition).decompose(task, reason)
        state.tasks.extend(children)
        self._transition(state, task.id, TaskStatus.DECOMPOSED, reason=reason, source="decomposer")
        state.plan_version += 1
        state.revisions.append(
            PlanRevision(
                trigger="decomposition",
                task_id=task.id,
                reason=reason,
                candidate_ids=candidate_ids,
                selected_candidate_id=selected_candidate_id,
                added_task_ids=[child.id for child in children],
                superseded_task_ids=[task.id],
            )
        )
        self._logger(state).log(
            "decomposition_generated",
            project_id=state.project_id,
            task_id=task.id,
            plan_version=state.plan_version,
            payload={"children": [child.id for child in children]},
        )

    def _session_record(
        self,
        state: ProjectState,
        task: TaskNode,
        session_id: str,
        result: RunResult,
        channel: TaskControlChannel,
        trace: SessionTrace,
    ) -> dict:
        terminal = channel.terminal_signal
        return {
            "project_id": state.project_id,
            "task_id": task.id,
            "session_id": session_id,
            "run_id": result.run_id,
            "task_attempt": task.attempts,
            "run_status": result.status.value,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_seconds": _duration_seconds(result.started_at, result.finished_at),
            "repository_fingerprint": RepositoryProfiler(self.config.workspace.root).profile().repository_fingerprint,
            "steps": result.steps,
            "tool_call_count": result.tool_call_count,
            "total_tokens": result.total_tokens,
            "terminal_signal": terminal.type.value if terminal else None,
            "files_touched": list(dict.fromkeys(task.files_touched)),
            "read_files": trace.read_files,
            "written_files": trace.written_files,
            "changed_files": trace.changed_files,
            "bash_commands": trace.bash_commands,
            "bash_exit_codes": trace.bash_exit_codes,
            "bash_observations": [item.model_dump() for item in trace.bash_observations],
            "successful_test_commands": trace.successful_test_commands,
            "successful_acceptance_commands": trace.successful_acceptance_commands,
            "repeated_tool_calls": trace.repeated_tool_calls,
            "suppressed_tool_calls": trace.suppressed_tool_calls,
            "terminal_grace_turn_count": result.terminal_grace_turn_count,
            "terminal_signal_recovered": result.terminal_signal_recovered,
            "auto_completion_recovered": task.auto_completion_recovered,
            "completion_candidate": task.completion_candidate.model_dump(mode="json") if task.completion_candidate else None,
            "unsupported_shell_syntax_count": trace.unsupported_shell_syntax_count,
            "protocol_error_count": result.protocol_error_count,
            "recoverable_protocol_error_count": result.recoverable_protocol_error_count,
            "fatal_protocol_error_count": result.fatal_protocol_error_count,
            "provider_error_count": result.provider_error_count,
            "tool_argument_protocol_retry_count": max(
                trace.tool_argument_protocol_retry_count,
                result.tool_argument_protocol_retry_count,
            ),
            "input_tokens_total": result.input_tokens_total,
            "output_tokens_total": result.output_tokens_total,
            "compactor_input_tokens": result.compactor_input_tokens,
            "compactor_output_tokens": result.compactor_output_tokens,
            "max_estimated_input_tokens": result.max_estimated_input_tokens,
            "max_actual_input_tokens": result.max_actual_input_tokens,
            "max_context_usage_ratio": result.max_context_usage_ratio,
            "context_segment_count": result.context_segment_count,
            "context_reset_count": result.context_reset_count,
            "deterministic_prune_count": result.deterministic_prune_count,
            "structured_compaction_count": result.structured_compaction_count,
            "pruned_item_count": result.pruned_item_count,
            "stale_item_count": result.stale_item_count,
            "superseded_item_count": result.superseded_item_count,
            "estimated_tokens_removed": result.estimated_tokens_removed,
            "context_budget_exhausted": result.context_budget_exhausted,
            "latest_context_handoff_id": result.latest_context_handoff_id,
            "no_progress": trace.no_progress(progress_count=len(channel.progress_signals), terminal_signal=terminal),
            "handoff_summary": task.last_handoff_summary
            if terminal is None
            and result.status
            in {
                RunStatus.MAX_STEPS_REACHED,
                RunStatus.TIME_LIMIT_REACHED,
                RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL,
                RunStatus.TERMINAL_SIGNAL_MISSING,
            }
            else None,
        }

    def _log_session_event(self, state: ProjectState, task: TaskNode, session_id: str, event_type: str, payload: dict) -> None:
        if event_type.startswith("context_") or event_type == "token_estimation_error_recorded":
            context_payload = {
                "project_id": state.project_id,
                "task_id": task.id,
                "session_id": session_id,
                "event_type": event_type,
                **payload,
            }
            self.store.append_context_event(context_payload)
            self._logger(state).log(
                event_type,
                project_id=state.project_id,
                task_id=task.id,
                session_id=session_id,
                plan_version=state.plan_version,
                payload=context_payload,
            )
            return
        if event_type not in {
            "terminal_grace_turn_started",
            "terminal_grace_turn_finished",
            "terminal_signal_recovered",
            "tool_arguments_protocol_error",
            "tool_arguments_protocol_retry",
            "tool_arguments_protocol_recovered",
        }:
            return
        self._logger(state).log(
            event_type,
            project_id=state.project_id,
            task_id=task.id,
            session_id=session_id,
            plan_version=state.plan_version,
            payload=payload,
        )

    def _write_metrics(self, state: ProjectState, project_started: float | None = None) -> None:
        wall_clock_seconds = None if project_started is None else max(0.0, time.monotonic() - project_started)
        self.store.write_metrics(
            state.project_id,
            project_statistics(
                state,
                self.store.read_sessions(state.project_id),
                configured_max_project_seconds=self.config.planning.execution.max_project_seconds,
                wall_clock_seconds=wall_clock_seconds,
                final_verification_exit_code=self._last_final_verification_exit_code,
            ),
        )

    def _project_time_exhausted(self, project_deadline: float) -> bool:
        return time.monotonic() >= project_deadline

    def _mark_project_time_limit(self, state: ProjectState, project_started: float) -> None:
        if state.status != ProjectStatus.TIME_LIMIT_REACHED:
            state.status = ProjectStatus.TIME_LIMIT_REACHED
            state.updated_at = utc_now()
            self._logger(state).log("project_time_limit_reached", project_id=state.project_id, plan_version=state.plan_version)
        self.store.save(state)
        self._write_metrics(state, project_started)

    def _finalize_candidate_complete(self, state: ProjectState, project_started: float) -> None:
        if self.config.verification.mode == "contract":
            self._finalize_contract_verification(state, project_started)
            return
        command = self.config.planning.execution.final_verification_command
        if command:
            self._logger(state).log("final_verification_started", project_id=state.project_id, plan_version=state.plan_version)
            try:
                result = subprocess.run(
                    command,
                    cwd=self.config.workspace.root,
                    text=True,
                    capture_output=True,
                    timeout=self.config.planning.execution.final_verification_timeout_seconds,
                    check=False,
                )
                exit_code = int(result.returncode)
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                exit_code = -1
                stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr or "" if isinstance(exc.stderr, str) else ""
                timed_out = True
            self._last_final_verification_exit_code = exit_code
            self.store.final_verification_path(state.project_id).write_text(
                "\n".join(
                    [
                        f"command: {' '.join(command)}",
                        f"exit_code: {exit_code}",
                        f"timed_out: {str(timed_out).lower()}",
                        "STDOUT:",
                        stdout,
                        "STDERR:",
                        stderr,
                    ]
                ),
                encoding="utf-8",
            )
            self._logger(state).log(
                "final_verification_finished",
                project_id=state.project_id,
                plan_version=state.plan_version,
                payload={"exit_code": exit_code, "timed_out": timed_out},
            )
            if self.config.verification.mode == "legacy_command":
                self._save_legacy_verification_report(
                    state,
                    command=command,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    stdout=stdout,
                    stderr=stderr,
                )
            if exit_code != 0:
                state.status = ProjectStatus.FAILED
                state.updated_at = utc_now()
                self.store.save(state)
                self._write_metrics(state, project_started)
                return
        state.status = ProjectStatus.CANDIDATE_COMPLETE
        state.updated_at = utc_now()
        self._logger(state).log("project_candidate_complete", project_id=state.project_id, plan_version=state.plan_version)
        self.store.save(state)
        self._write_metrics(state, project_started)

    def _save_legacy_verification_report(
        self,
        state: ProjectState,
        *,
        command: list[str],
        exit_code: int,
        timed_out: bool,
        stdout: str,
        stderr: str,
    ) -> None:
        assert self.config.verification.store_root is not None
        store = VerificationStore(
            self.config.verification.store_root,
            state.project_id,
            workspace_root=self.config.workspace.root,
            atomic_write=self.config.state.atomic_write,
        )
        contract = VerificationContract(
            contract_id=f"legacy-{state.project_id}",
            project_id=state.project_id,
            source="legacy",
            checks=[
                VerificationCheck(
                    check_id="legacy-final-command",
                    title="Legacy final verification command",
                    kind=CheckKind.CANDIDATE_ONLY,
                    visibility=CheckVisibility.PUBLIC,
                    argv=command,
                    timeout_seconds=self.config.planning.execution.final_verification_timeout_seconds,
                )
            ],
        ).freeze()
        store.save_contract(contract)
        status = ExecutionStatus.TIMEOUT if timed_out else ExecutionStatus.PASSED if exit_code == 0 else ExecutionStatus.FAILED
        result = CheckExecutionResult(
            check_id="legacy-final-command",
            kind=CheckKind.CANDIDATE_ONLY,
            visibility=CheckVisibility.PUBLIC,
            workspace_kind="current",
            started_at=utc_now(),
            finished_at=utc_now(),
            duration_seconds=0,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout_excerpt=stdout[-self.config.verification.execution.max_output_chars :],
            stderr_excerpt=stderr[-self.config.verification.execution.max_output_chars :],
            status=status,
            infrastructure_error="legacy verification timed out" if timed_out else None,
        )
        report = VerificationReport(
            purpose=VerificationPurpose.RUNTIME,
            project_id=state.project_id,
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            verdict=VerificationVerdict.VERIFIED if exit_code == 0 else VerificationVerdict.REOPENED,
            candidate_results=[result],
            summary=VerificationSummary(
                required_checks_passed=int(exit_code == 0),
                required_checks_failed=int(exit_code != 0),
                integrity_passed=True,
            ),
            sanitized_feedback="Legacy final verification passed." if exit_code == 0 else "Legacy final verification failed.",
        )
        store.save_report(report)
        self._latest_verification_report = report
        state.latest_project_verification_report_id = report.report_id

    def _finalize_contract_verification(self, state: ProjectState, project_started: float) -> None:
        contract = self.verification_contract
        if contract is None or contract.scope != "project":
            if self.config.verification.policy.require_project_contract:
                state.status = ProjectStatus.VERIFICATION_INCONCLUSIVE
                self._logger(state).log(
                    "verification_inconclusive",
                    project_id=state.project_id,
                    plan_version=state.plan_version,
                    reason="project verification contract is missing",
                )
            else:
                state.status = ProjectStatus.CANDIDATE_COMPLETE
            self.store.save(state)
            self._write_metrics(state, project_started)
            return
        if state.project_verification_attempts >= self.config.verification.policy.max_project_verification_attempts:
            state.status = ProjectStatus.VERIFICATION_INCONCLUSIVE
            self.store.save(state)
            self._write_metrics(state, project_started)
            return
        state.status = ProjectStatus.VERIFICATION_PENDING
        state.project_verification_attempts += 1
        self._logger(state).log(
            "project_verification_pending",
            project_id=state.project_id,
            plan_version=state.plan_version,
            payload={"contract_id": contract.contract_id, "contract_hash": contract.contract_hash},
        )
        test_candidates = self.verification_store.list_test_candidates() if self.verification_store else []
        report = self._verification_gateway().verify(contract, test_candidates=test_candidates)
        self._latest_verification_report = report
        state.latest_project_verification_report_id = report.report_id
        if report.verdict == VerificationVerdict.VERIFIED:
            self._mark_project_tasks_verified(state, report)
            state.status = ProjectStatus.VERIFIED
            state.verified_at = utc_now()
            self._logger(state).log(
                "project_verified",
                project_id=state.project_id,
                plan_version=state.plan_version,
                payload={"report_id": report.report_id, "verdict": report.verdict.value},
            )
        elif report.verdict == VerificationVerdict.PARTIAL:
            state.status = ProjectStatus.PARTIALLY_VERIFIED
            self._reopen_relevant_project_task(state, report)
        elif report.verdict == VerificationVerdict.REOPENED:
            state.status = ProjectStatus.ACTIVE
            self._reopen_relevant_project_task(state, report)
        elif report.verdict == VerificationVerdict.INFRASTRUCTURE_ERROR:
            state.status = ProjectStatus.VERIFICATION_PENDING
            self._logger(state).log(
                "verification_infrastructure_error",
                project_id=state.project_id,
                plan_version=state.plan_version,
                reason=report.sanitized_feedback,
                payload={"report_id": report.report_id},
            )
        else:
            state.status = ProjectStatus.VERIFICATION_INCONCLUSIVE
            self._logger(state).log(
                "verification_inconclusive",
                project_id=state.project_id,
                plan_version=state.plan_version,
                reason=report.sanitized_feedback,
                payload={"report_id": report.report_id, "verdict": report.verdict.value},
            )
        state.updated_at = utc_now()
        self.store.save(state)
        self._write_metrics(state, project_started)

    def _mark_project_tasks_verified(self, state: ProjectState, report: VerificationReport) -> None:
        for task in state.leaf_tasks():
            if task.status != TaskStatus.CANDIDATE_COMPLETE:
                continue
            self._transition(
                state,
                task.id,
                TaskStatus.VERIFICATION_PENDING,
                reason="project verification contract passed",
                source="verification",
            )
            self._transition(
                state,
                task.id,
                TaskStatus.VERIFIED,
                reason="covered by verified project contract",
                source="verification",
            )
            task.verification_status = report.verdict.value
            task.latest_verification_report_id = report.report_id
            task.verified_at = utc_now()
            task.verified_contract_hash = report.contract_hash
            self._logger(state).log(
                "task_verified",
                project_id=state.project_id,
                task_id=task.id,
                plan_version=state.plan_version,
                payload={"report_id": report.report_id, "verdict": report.verdict.value, "scope": "project"},
            )

    def _reopen_relevant_project_task(self, state: ProjectState, report: VerificationReport) -> None:
        candidates = [task for task in reversed(state.leaf_tasks()) if task.status in {TaskStatus.CANDIDATE_COMPLETE, TaskStatus.VERIFIED}]
        if not candidates:
            return
        task = candidates[0]
        if task.status == TaskStatus.VERIFIED:
            task.status = TaskStatus.CANDIDATE_COMPLETE
        self._transition(
            state,
            task.id,
            TaskStatus.VERIFICATION_PENDING,
            reason="project verification mapped failure to task",
            source="verification",
        )
        self._reopen_task_after_verification(state, task, report)

    def _outcome(self, state: ProjectState, sessions_run: int, run_statuses: list[RunStatus]) -> ProjectRunOutcome:
        return ProjectRunOutcome(
            project_id=state.project_id,
            status=state.status.value,
            sessions_run=sessions_run,
            state_path=str(self.store.state_path(state.project_id)),
            run_statuses=run_statuses,
            verification_verdict=(self._latest_verification_report.verdict.value if self._latest_verification_report else None),
            verification_report_id=(self._latest_verification_report.report_id if self._latest_verification_report else None),
        )

    def _transition(self, state: ProjectState, task_id: str, new_status: TaskStatus, *, reason: str, source: str) -> None:
        record = self.transitions.transition(state, task_id, new_status, reason=reason, source=source)
        self._logger(state).log(
            "state_transition",
            project_id=state.project_id,
            task_id=task_id,
            plan_version=state.plan_version,
            old_status=record.old_status.value,
            new_status=record.new_status.value,
            reason=reason,
            trigger=source,
        )

    def _logger(self, state: ProjectState) -> ProjectLogger:
        return ProjectLogger(self.store.events_path(state.project_id))


def _duration_seconds(started_at: str, finished_at: str) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds())
    except ValueError:
        return 0.0


def _knowledge_attribution(task: TaskNode, pack) -> KnowledgeUseType:
    formally_verified = (
        pack.verification_verdict == "verified" if pack.verification_verdict else task.status == TaskStatus.CANDIDATE_COMPLETE
    )
    verified_success = formally_verified and bool(pack.successful_verifications)
    if verified_success:
        return KnowledgeUseType.HELPFUL
    if pack.files_changed and pack.failed_verifications and not pack.successful_verifications:
        return KnowledgeUseType.HARMFUL
    return KnowledgeUseType.NEUTRAL


class _ChannelRouter(ToolRouter):
    def __init__(
        self,
        inner: ToolRouter,
        channel: TaskControlChannel,
        trace: SessionTrace,
        on_suppressed=None,
        on_unsupported_shell=None,
        knowledge_channel=None,
    ):
        super().__init__(list(inner.tools.values()))
        self.channel = channel
        self.trace = trace
        self.on_suppressed = on_suppressed
        self.on_unsupported_shell = on_unsupported_shell
        self.knowledge_channel = knowledge_channel
        self.action_required_message: str | None = None

    def execute(self, call, context):
        context.control_channel = self.channel
        context.knowledge_channel = self.knowledge_channel
        if self._knowledge_decision_blocks(call):
            result = ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                summary="knowledge_decision_required",
                output=(
                    "Knowledge decision required before this action.\n"
                    "Call report_knowledge_use:\n"
                    "- with the exposed IDs that materially affected the task; or\n"
                    "- with empty ID lists and a reason when none were used."
                ),
                error_type=ErrorType.POLICY_GATE,
                error_message="knowledge_decision_required",
                metadata={"knowledge_decision_required": True},
            )
            self.trace.record_policy_gate(result)
            self.action_required_message = result.output
            return result
        if self.trace.should_suppress(call):
            self.trace.record_suppressed(call)
            call_key = self.trace.call_key(call)
            if self.on_suppressed:
                self.on_suppressed(call_key)
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=True,
                summary="repeated_tool_call_suppressed",
                output="This exact tool call was made immediately before; reuse the previous observation instead of repeating it.",
                metadata={"call_key": call_key, "suppressed": True},
            )
        result = super().execute(call, context)
        self.trace.record(call, result)
        if result.metadata.get("unsupported_shell_syntax") and self.on_unsupported_shell:
            self.on_unsupported_shell(result)
        if len(result.output) > 12000:
            result.output = result.output[:6000] + "\n...[truncated for project session]...\n" + result.output[-6000:]
        self.action_required_message = self.trace.action_required_message
        return result

    def clear_action_required(self) -> None:
        self.action_required_message = None

    def record_protocol_retry(self) -> None:
        self.trace.record_protocol_retry()

    def knowledge_decision_pending(self) -> bool:
        return bool(
            self.knowledge_channel is not None
            and self.knowledge_channel.has_exposed_knowledge()
            and not self.knowledge_channel.decision_recorded
        )

    def _knowledge_decision_blocks(self, call) -> bool:
        if not self.knowledge_decision_pending():
            return False
        if call.name in {
            "write_file",
            "request_task_completion",
            "report_blocker",
            "request_decomposition",
        }:
            return True
        if call.name == "bash" and _is_verification_tool_call(call.arguments):
            return True
        return False


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _is_verification_tool_call(arguments: dict) -> bool:
    try:
        parsed = BashArgs.model_validate(arguments)
    except (TypeError, ValueError):
        return False
    command = parsed.command or render_command(parsed.argv or [])
    lowered = command.lower()
    return "pytest" in lowered or " validate" in f" {lowered} " or " test" in f" {lowered} "
