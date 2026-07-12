from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.agent.prompt import SYSTEM_PROMPT
from longrun_agent.config import AppConfig
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

        logger.log(0, "run_started", payload={"workspace": str(workspace_path), "config": self.config.sanitized()})
        self._emit("run_started", {"run_id": self.run_id, "run_dir": str(run_dir)})

        for step in range(1, self.config.agent.max_steps + 1):
            logger.log(
                step,
                "model_request",
                action_type="model",
                payload={"message_count": len(messages), "tool_count": len(self.router.schemas())},
            )
            if self.config.telemetry.save_prompts:
                logger.save_prompt(step, {"messages": messages, "tools": self.router.schemas()})
            self._emit("model_request", {"step": step})
            try:
                response = self.model.generate(messages, self.router.schemas())
            except Exception as exc:
                status = RunStatus.PROVIDER_ERROR
                consecutive_errors += 1
                logger.log(
                    step,
                    "provider_error",
                    action_type="model",
                    success=False,
                    error_type=ErrorType.PROVIDER.value,
                    error_message=str(exc),
                )
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
            if consecutive_errors >= self.config.agent.max_consecutive_errors:
                status = RunStatus.ABORTED
                break

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
