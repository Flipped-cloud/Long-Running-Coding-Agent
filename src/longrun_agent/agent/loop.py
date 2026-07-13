from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.agent.prompt import SYSTEM_PROMPT
from longrun_agent.config import AppConfig
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import ErrorType, ModelResponse, RunResult, RunStatus
from longrun_agent.telemetry.logger import EventLogger
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.bash import BashTool
from longrun_agent.tools.path_guard import ensure_workspace_root
from longrun_agent.tools.read_file import ReadFileTool
from longrun_agent.tools.router import ToolRouter
from longrun_agent.tools.write_file import WriteFileTool


def default_router() -> ToolRouter:
    return ToolRouter([ReadFileTool(), WriteFileTool(), BashTool()])


class AgentLoop:
    """Minimal model/tool/observation loop."""

    def __init__(
        self,
        config: AppConfig,
        model: ModelProvider,
        router: ToolRouter | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        run_id: str | None = None,
    ):
        self.config = config
        self.model = model
        self.router = router or default_router()
        self.on_event = on_event
        self.run_id = run_id or str(uuid.uuid4())

    def run(self, workspace: str | Path | None, task: str) -> RunResult:
        return self.run_with_controls(workspace, task)

    def run_with_controls(
        self,
        workspace: str | Path | None,
        task: str,
        *,
        deadline_monotonic: float | None = None,
        stop_condition: Callable[[], bool] | None = None,
        require_external_terminal: bool = False,
        completion_evidence: Callable[[], bool] | None = None,
    ) -> RunResult:
        workspace_path = ensure_workspace_root(workspace or self.config.workspace.root)
        run_dir = self.config.telemetry.run_root / self.run_id
        logger = EventLogger(self.run_id, run_dir, self.config.model.model_name)
        context = ToolContext(
            workspace=workspace_path,
            tool_outputs_dir=logger.tool_outputs_dir,
            diffs_dir=logger.diffs_dir,
            config=self.config.tools,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        started_at = datetime.now(UTC).isoformat()
        total_tokens = 0
        tool_count = 0
        consecutive_errors = 0
        final_answer: str | None = None
        status = RunStatus.MAX_STEPS_REACHED
        step = 0
        seen_final_without_signal = False
        terminal_grace_turn_count = 0
        terminal_signal_recovered = False
        tool_argument_protocol_retry_count = 0
        session_deadline = time.monotonic() + self.config.agent.max_session_seconds
        if deadline_monotonic is not None:
            session_deadline = min(session_deadline, deadline_monotonic)

        logger.log(0, "run_started", payload={"workspace": str(workspace_path), "config": self.config.sanitized()})
        self._emit("run_started", {"run_id": self.run_id, "run_dir": str(run_dir)})

        for step in range(1, self.config.agent.max_steps + 1):
            if time.monotonic() >= session_deadline:
                status = RunStatus.TIME_LIMIT_REACHED
                logger.log(
                    step,
                    "time_limit_reached",
                    action_type="run",
                    success=False,
                    error_type=ErrorType.ENVIRONMENT.value,
                    error_message="session deadline reached before next model request",
                )
                break
            remaining = self.config.agent.max_steps - step + 1
            terminal_tools_only = False
            if require_external_terminal and remaining == 3:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Three model turns remain. Stop exploration, complete verification, "
                            "and reserve the final turn for a terminal control tool."
                        ),
                    }
                )
            if require_external_terminal and remaining == 1:
                terminal_tools_only = True
                if completion_evidence is not None and completion_evidence():
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Final model turn: implementation and verification evidence already exists. "
                                "Only call request_task_completion if the acceptance criteria are satisfied, "
                                "otherwise call report_blocker with the exact remaining issue. Do not run more tools."
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Final model turn: no completion evidence is recorded. "
                                "Call report_blocker with the exact remaining issue, or request_task_completion only if "
                                "you can cite satisfied acceptance criteria from existing observations."
                            ),
                        }
                    )
            tools_for_request = self._schemas(terminal_tools_only=terminal_tools_only)
            logger.log(
                step,
                "model_request",
                action_type="model",
                payload={"message_count": len(messages), "tool_count": len(tools_for_request), "terminal_tools_only": terminal_tools_only},
            )
            if self.config.telemetry.save_prompts:
                logger.save_prompt(step, {"messages": messages, "tools": tools_for_request})
            self._emit("model_request", {"step": step})
            response, retry_count, protocol_failed = self._generate_with_protocol_retries(
                messages=messages,
                tools=tools_for_request,
                logger=logger,
                step=step,
            )
            tool_argument_protocol_retry_count += retry_count
            if protocol_failed:
                status = RunStatus.PROTOCOL_ERROR
                consecutive_errors += 1
                break
            if response is None:
                status = RunStatus.PROVIDER_ERROR
                consecutive_errors += 1
                break

            total_tokens += int(response.usage.get("total_tokens", 0))
            logger.log(
                step,
                "model_response",
                action_type=response.kind,
                input_tokens=response.usage.get("input_tokens"),
                output_tokens=response.usage.get("output_tokens"),
                payload={"provider_request_id": response.provider_request_id},
            )
            self._emit("model_response", {"step": step, "kind": response.kind, "usage": response.usage})

            if response.final_answer is not None and not response.tool_calls:
                final_answer = response.final_answer.content
                logger.log(step, "final_answer", action_type="final_answer", success=True, summary=final_answer)
                messages.append(response.raw_metadata.get("message", {"role": "assistant", "content": final_answer}))
                if require_external_terminal and (stop_condition is None or not stop_condition()):
                    seen_final_without_signal = True
                    logger.log(
                        step,
                        "project_session_final_without_signal",
                        action_type="model",
                        success=False,
                        error_type=ErrorType.PROTOCOL.value,
                        error_message="Project Session requires request_task_completion, report_blocker, or request_decomposition",
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FinalAnswer does not change Project Task state. Call exactly one terminal control tool: "
                                "request_task_completion, report_blocker, or request_decomposition."
                            ),
                        }
                    )
                    continue
                status = RunStatus.COMPLETED
                break

            if not response.tool_calls:
                consecutive_errors += 1
                logger.log(
                    step,
                    "protocol_error",
                    action_type="model",
                    success=False,
                    error_type=ErrorType.PROTOCOL.value,
                    error_message="model response had no tool calls or final answer",
                )
                if consecutive_errors >= self.config.agent.max_consecutive_errors:
                    status = RunStatus.ABORTED
                    break
                messages.append({"role": "user", "content": "Return valid tool calls or a final answer."})
                continue

            messages.append(self._assistant_tool_message(response))
            for call in response.tool_calls:
                tool_count += 1
                logger.log(
                    step,
                    "tool_started",
                    action_type="tool_call",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    sanitized_arguments=call.arguments,
                )
                self._emit("tool_started", {"step": step, "tool": call.name})
                result = self.router.execute(call, context)
                if result.success:
                    consecutive_errors = 0
                elif result.error_type != ErrorType.ENVIRONMENT:
                    consecutive_errors += 1
                logger.log(
                    step,
                    "tool_finished",
                    action_type="tool_result",
                    tool_call_id=result.tool_call_id,
                    tool_name=result.tool_name,
                    success=result.success,
                    summary=result.summary,
                    duration_seconds=result.metadata.get("duration_seconds"),
                    exit_code=result.metadata.get("exit_code"),
                    artifact_path=result.artifact_path,
                    error_type=result.error_type.value if result.error_type else None,
                    error_message=result.error_message,
                    payload={"metadata": result.metadata},
                )
                self._emit(
                    "tool_finished",
                    {"step": step, "tool": call.name, "success": result.success, "summary": result.summary, "metadata": result.metadata},
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.model_dump_json(),
                    }
                )
            action_required = getattr(self.router, "action_required_message", None)
            if action_required:
                messages.append({"role": "user", "content": action_required})
                clear_action_required = getattr(self.router, "clear_action_required", None)
                if clear_action_required:
                    clear_action_required()
            if stop_condition is not None and stop_condition():
                status = RunStatus.COMPLETED
                break
            if consecutive_errors >= self.config.agent.max_consecutive_errors:
                status = RunStatus.ABORTED
                break
        else:
            if seen_final_without_signal:
                status = RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL

        if status == RunStatus.MAX_STEPS_REACHED and seen_final_without_signal:
            status = RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL

        if (
            require_external_terminal
            and self.config.agent.terminal_grace_turns > 0
            and (stop_condition is None or not stop_condition())
            and status in {RunStatus.MAX_STEPS_REACHED, RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL}
            and completion_evidence is not None
            and completion_evidence()
        ):
            for grace_index in range(1, self.config.agent.terminal_grace_turns + 1):
                terminal_grace_turn_count += 1
                grace_step = self.config.agent.max_steps + grace_index
                logger.log(
                    grace_step,
                    "terminal_grace_turn_started",
                    action_type="model",
                    payload={"grace_index": grace_index},
                )
                self._emit("terminal_grace_turn_started", {"step": grace_step, "grace_index": grace_index})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Verification evidence is complete. A completion candidate has been generated. "
                            "Confirm completion by calling request_task_completion. Otherwise call report_blocker "
                            "with the exact remaining issue. Do not perform more exploration."
                        ),
                    }
                )
                tools_for_request = self._schemas(terminal_tools_only=True)
                response, retry_count, protocol_failed = self._generate_with_protocol_retries(
                    messages=messages,
                    tools=tools_for_request,
                    logger=logger,
                    step=grace_step,
                )
                tool_argument_protocol_retry_count += retry_count
                if protocol_failed:
                    status = RunStatus.PROTOCOL_ERROR
                    break
                if response is None:
                    status = RunStatus.PROVIDER_ERROR
                    break
                total_tokens += int(response.usage.get("total_tokens", 0))
                logger.log(
                    grace_step,
                    "model_response",
                    action_type=response.kind,
                    input_tokens=response.usage.get("input_tokens"),
                    output_tokens=response.usage.get("output_tokens"),
                    payload={"provider_request_id": response.provider_request_id, "terminal_grace_turn": True},
                )
                self._emit("model_response", {"step": grace_step, "kind": response.kind, "usage": response.usage})
                if response.tool_calls:
                    messages.append(self._assistant_tool_message(response))
                    for call in response.tool_calls:
                        tool_count += 1
                        logger.log(
                            grace_step,
                            "tool_started",
                            action_type="tool_call",
                            tool_call_id=call.id,
                            tool_name=call.name,
                            sanitized_arguments=call.arguments,
                        )
                        result = self.router.execute(call, context)
                        logger.log(
                            grace_step,
                            "tool_finished",
                            action_type="tool_result",
                            tool_call_id=result.tool_call_id,
                            tool_name=result.tool_name,
                            success=result.success,
                            summary=result.summary,
                            duration_seconds=result.metadata.get("duration_seconds"),
                            exit_code=result.metadata.get("exit_code"),
                            artifact_path=result.artifact_path,
                            error_type=result.error_type.value if result.error_type else None,
                            error_message=result.error_message,
                            payload={"metadata": result.metadata, "terminal_grace_turn": True},
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": result.model_dump_json(),
                            }
                        )
                if stop_condition is not None and stop_condition():
                    status = RunStatus.COMPLETED
                    terminal_signal_recovered = True
                    logger.log(
                        grace_step,
                        "terminal_signal_recovered",
                        action_type="run",
                        success=True,
                        payload={"grace_index": grace_index},
                    )
                    break
                status = RunStatus.TERMINAL_SIGNAL_MISSING
            logger.log(
                self.config.agent.max_steps + terminal_grace_turn_count,
                "terminal_grace_turn_finished",
                action_type="run",
                success=terminal_signal_recovered,
                summary=status.value,
                payload={"terminal_signal_recovered": terminal_signal_recovered},
            )

        if (
            require_external_terminal
            and status in {RunStatus.MAX_STEPS_REACHED, RunStatus.FINAL_WITHOUT_TERMINAL_SIGNAL}
            and completion_evidence is not None
            and completion_evidence()
            and (stop_condition is None or not stop_condition())
        ):
            status = RunStatus.TERMINAL_SIGNAL_MISSING

        finished_at = datetime.now(UTC).isoformat()
        result = RunResult(
            run_id=self.run_id,
            status=status,
            final_answer=final_answer,
            steps=step,
            consecutive_errors=consecutive_errors,
            started_at=started_at,
            finished_at=finished_at,
            event_log_path=str(logger.events_path),
            run_json_path=str(logger.run_json_path),
            tool_call_count=tool_count,
            total_tokens=total_tokens,
            terminal_grace_turn_count=terminal_grace_turn_count,
            terminal_signal_recovered=terminal_signal_recovered,
            tool_argument_protocol_retry_count=tool_argument_protocol_retry_count,
        )
        logger.log(
            result.steps,
            "run_finished",
            action_type="run",
            success=status == RunStatus.COMPLETED,
            summary=status.value,
            payload=result.model_dump(mode="json"),
        )
        logger.save_run(result, {"tool_call_count": tool_count, "total_tokens": total_tokens})
        self._emit("run_finished", result.model_dump(mode="json"))
        return result

    def _generate_with_protocol_retries(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        logger: EventLogger,
        step: int,
    ) -> tuple[ModelResponse | None, int, bool]:
        retries = 0
        while True:
            try:
                response = self.model.generate(messages, tools)
                if retries:
                    logger.log(
                        step,
                        "tool_arguments_protocol_recovered",
                        action_type="model",
                        success=True,
                        payload={"retry_count": retries},
                    )
                    self._emit("tool_arguments_protocol_recovered", {"step": step, "retry_count": retries})
                return response, retries, False
            except ToolArgumentsProtocolError as exc:
                logger.log(
                    step,
                    "tool_arguments_protocol_error",
                    action_type="model",
                    success=False,
                    error_type=ErrorType.PROTOCOL.value,
                    error_message=str(exc),
                    payload={"tool_name": exc.tool_name, "parse_error": exc.parse_error},
                )
                self._emit("tool_arguments_protocol_error", {"step": step, "tool_name": exc.tool_name})
                if retries >= self.config.agent.protocol_retries_per_step:
                    return None, retries, True
                retries += 1
                retry_message = self._tool_arguments_retry_message(exc)
                messages.append({"role": "user", "content": retry_message})
                record_protocol_retry = getattr(self.router, "record_protocol_retry", None)
                if record_protocol_retry:
                    record_protocol_retry()
                logger.log(
                    step,
                    "tool_arguments_protocol_retry",
                    action_type="model",
                    success=True,
                    error_type=ErrorType.PROTOCOL.value,
                    error_message=str(exc),
                    payload={"tool_name": exc.tool_name, "retry": retries},
                )
                self._emit("tool_arguments_protocol_retry", {"step": step, "tool_name": exc.tool_name, "retry": retries})
            except Exception as exc:
                logger.log(
                    step,
                    "provider_error",
                    action_type="model",
                    success=False,
                    error_type=ErrorType.PROVIDER.value,
                    error_message=str(exc),
                )
                return None, retries, False

    def _tool_arguments_retry_message(self, exc: ToolArgumentsProtocolError) -> str:
        schema = None
        tool = getattr(self.router, "tools", {}).get(exc.tool_name)
        if tool is not None:
            schema = tool.input_schema()
        return "\n".join(
            [
                f"Protocol error: tool call arguments for {exc.tool_name} were not valid JSON.",
                f"JSON parse error: {exc.parse_error}",
                f"Expected arguments schema: {json.dumps(schema or {}, sort_keys=True)}",
                "Retry by calling the same intended tool with a valid JSON object for arguments.",
            ]
        )

    def _schemas(self, *, terminal_tools_only: bool) -> list[dict[str, Any]]:
        schemas = self.router.schemas()
        if not terminal_tools_only:
            return schemas
        allowed = {"request_task_completion", "report_blocker"}
        return [schema for schema in schemas if schema.get("function", {}).get("name") in allowed]

    @staticmethod
    def _assistant_tool_message(response: ModelResponse) -> dict[str, Any]:
        if response.raw_metadata.get("message"):
            return response.raw_metadata["message"]
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in response.tool_calls
            ],
        }

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(event_type, payload)
