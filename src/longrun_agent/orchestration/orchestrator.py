from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime

from longrun_agent.agent.loop import AgentLoop, default_router
from longrun_agent.config import AppConfig
from longrun_agent.control.channel import ControlSignalType, TaskControlChannel
from longrun_agent.control.tools import control_tools
from longrun_agent.model.base import ModelProvider
from longrun_agent.orchestration.outcome import ProjectRunOutcome
from longrun_agent.orchestration.session_prompt import build_task_session_prompt
from longrun_agent.orchestration.session_trace import SessionTrace
from longrun_agent.planning.decomposer import AsNeededDecomposer
from longrun_agent.planning.initial_planner import InitialPlanner
from longrun_agent.planning.recovery_evaluator import RecoveryCandidateEvaluator
from longrun_agent.planning.recovery_generator import RecoveryCandidateGenerator
from longrun_agent.protocol import RunResult, RunStatus, ToolResult
from longrun_agent.state.aggregation import aggregate_candidate_complete_parents, project_statistics
from longrun_agent.state.schema import CompletionCandidate, PlanRevision, ProjectState, ProjectStatus, TaskNode, TaskStatus, utc_now
from longrun_agent.state.selector import TaskSelector
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.state.transitions import StateTransitionController
from longrun_agent.telemetry.project_logger import ProjectLogger
from longrun_agent.tools.router import ToolRouter


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
        self.selector = TaskSelector(self.transitions)
        self._last_final_verification_exit_code: int | None = None

    def start(self, objective: str) -> ProjectRunOutcome:
        if self.store.exists(self.project_id):
            state = self.store.load(self.project_id)
        else:
            state = ProjectState(project_id=self.project_id, objective=objective)
            self.store.create(state)
            self._logger(state).log("project_created", project_id=state.project_id, plan_version=state.plan_version)
        return self.run_project(state)

    def resume(self, project_id: str) -> ProjectRunOutcome:
        self.project_id = project_id
        state = self.store.load(project_id)
        if state.status in {ProjectStatus.SESSION_LIMIT_REACHED, ProjectStatus.TIME_LIMIT_REACHED, ProjectStatus.FAILED}:
            state.status = ProjectStatus.ACTIVE
            state.updated_at = utc_now()
            for task in state.tasks:
                if task.status == TaskStatus.FAILED and len(task.session_ids) < self.config.planning.execution.max_sessions_per_task:
                    task.status = TaskStatus.READY
                    task.updated_at = utc_now()
        self._logger(state).log("project_resumed", project_id=state.project_id, plan_version=state.plan_version)
        return self.run_project(state)

    def run_project(self, state: ProjectState) -> ProjectRunOutcome:
        project_started = time.monotonic()
        project_deadline = project_started + self.config.planning.execution.max_project_seconds
        run_statuses = []
        if self._project_time_exhausted(project_deadline):
            self._mark_project_time_limit(state, project_started)
            return self._outcome(state, 0, run_statuses)
        if not state.tasks and self.config.planning.mode != "disabled":
            self._create_initial_plan(state)
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
            channel = TaskControlChannel()
            result, trace = self._run_task_session(state, task, channel, session_id, project_deadline)
            run_statuses.append(result.status)
            self._process_control_signals(state, task, channel, session_id, result, trace)
            self.store.append_session(state.project_id, self._session_record(state, task, session_id, result, channel, trace))
            self.store.save(state)
            self._write_metrics(state, project_started)
            if state.status in {ProjectStatus.CANDIDATE_COMPLETE, ProjectStatus.BLOCKED, ProjectStatus.FAILED}:
                break
            if self._project_time_exhausted(project_deadline):
                self._mark_project_time_limit(state, project_started)
                break
        for parent_id in aggregate_candidate_complete_parents(state):
            self._logger(state).log(
                "parent_task_aggregated", project_id=state.project_id, task_id=parent_id, plan_version=state.plan_version
            )
        leaves = state.leaf_tasks()
        if state.status == ProjectStatus.ACTIVE and leaves and all(task.status == TaskStatus.CANDIDATE_COMPLETE for task in leaves):
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
        router = ToolRouter([*default_router().tools.values(), *control_tools()])
        trace = SessionTrace()
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
        )
        loop.router = router_with_channel
        result = loop.run_with_controls(
            self.config.workspace.root,
            build_task_session_prompt(state, task, self.config),
            deadline_monotonic=project_deadline,
            stop_condition=lambda: channel.terminal_signal is not None,
            require_external_terminal=True,
            completion_evidence=lambda: trace.has_completion_evidence(existing_changed_files=task.files_touched),
        )
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
        return result, trace

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
            "successful_test_commands": trace.successful_test_commands,
            "successful_acceptance_commands": trace.successful_acceptance_commands,
            "repeated_tool_calls": trace.repeated_tool_calls,
            "suppressed_tool_calls": trace.suppressed_tool_calls,
            "terminal_grace_turn_count": result.terminal_grace_turn_count,
            "terminal_signal_recovered": result.terminal_signal_recovered,
            "auto_completion_recovered": task.auto_completion_recovered,
            "completion_candidate": task.completion_candidate.model_dump(mode="json") if task.completion_candidate else None,
            "unsupported_shell_syntax_count": trace.unsupported_shell_syntax_count,
            "tool_argument_protocol_retry_count": max(
                trace.tool_argument_protocol_retry_count,
                result.tool_argument_protocol_retry_count,
            ),
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

    def _outcome(self, state: ProjectState, sessions_run: int, run_statuses: list[RunStatus]) -> ProjectRunOutcome:
        return ProjectRunOutcome(
            project_id=state.project_id,
            status=state.status.value,
            sessions_run=sessions_run,
            state_path=str(self.store.state_path(state.project_id)),
            run_statuses=run_statuses,
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


class _ChannelRouter(ToolRouter):
    def __init__(
        self,
        inner: ToolRouter,
        channel: TaskControlChannel,
        trace: SessionTrace,
        on_suppressed=None,
        on_unsupported_shell=None,
    ):
        super().__init__(list(inner.tools.values()))
        self.channel = channel
        self.trace = trace
        self.on_suppressed = on_suppressed
        self.on_unsupported_shell = on_unsupported_shell
        self.action_required_message: str | None = None

    def execute(self, call, context):
        context.control_channel = self.channel
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


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
