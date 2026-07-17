from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from longrun_agent.config import AppConfig
from longrun_agent.context.assembler import ContextAssembler
from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.lifecycle import ContextLifecycleManager
from longrun_agent.context.schema import ContextPreparationAction, ContextPreparationResult, TaskContextSeed
from longrun_agent.context_probes.answer_tools import ANSWER_TOOL_NAMES, answer_tool_schema, parse_probe_answer
from longrun_agent.context_probes.config import ProbeEvalConfig, load_mode_app_config
from longrun_agent.context_probes.evaluator import evaluate_prediction
from longrun_agent.context_probes.schemas import ProbeCase, ProbeCaseResult, ProbeMode, ProbePrediction
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.base import ModelProvider
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore

ProviderFactory = Callable[[AppConfig], ModelProvider]


class ProbeExecutor:
    def __init__(
        self,
        eval_config: ProbeEvalConfig,
        *,
        provider_factory: ProviderFactory | None = None,
        fake_provider_script: Path | None = None,
    ):
        self.eval_config = eval_config
        self.provider_factory = provider_factory
        self.fake_provider_script = fake_provider_script

    def run_case(self, case: ProbeCase, mode: ProbeMode) -> ProbeCaseResult:
        force_fake = self.provider_factory is not None or self.fake_provider_script is not None
        app_config = load_mode_app_config(
            self.eval_config,
            mode,
            force_fake_provider=force_fake,
            force_deterministic_compactor=self.eval_config.structured_reset_compactor == "deterministic",
        )
        prediction = self._execute(case, mode, app_config, source_config_path=self.eval_config.mode_configs[mode])
        return evaluate_prediction(case, prediction)

    def _provider(self, app_config: AppConfig) -> ModelProvider:
        if self.provider_factory is not None:
            return self.provider_factory(app_config)
        if self.fake_provider_script is not None:
            return FakeModelProvider(_load_scripted_responses(self.fake_provider_script))
        if app_config.model.provider == "fake":
            return FakeModelProvider([])
        return OpenAICompatibleProvider(app_config.model)

    def _execute(self, case: ProbeCase, mode: ProbeMode, app_config: AppConfig, *, source_config_path: Path) -> ProbePrediction:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="longrun-context-probe-") as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            store = ProjectStateStore(tmp_path / "state", workspace_root=workspace)
            seed = _seed(case)
            assembler = ContextAssembler(app_config.context)
            buffer = ContextBuffer(
                system_message=assembler.system_message(),
                task_anchor_message=assembler.task_anchor_message(seed),
            )
            events: list[dict[str, Any]] = []

            def sink(event_type: str, payload: dict[str, Any]) -> None:
                events.append({"event_type": event_type, **payload})

            runtime_context_config = _runtime_context_config(app_config, source_config_path)
            sink("context_runtime_config_resolved", runtime_context_config)
            provider: ModelProvider | None = None
            if (
                app_config.context.mode == "structured_reset"
                and self.eval_config.structured_reset_compactor == "configured"
                and app_config.context.structured_handoff.use_model
            ):
                provider = self._provider(app_config)
            manager = ContextLifecycleManager(
                app_config.context,
                seed=seed,
                model=provider,
                store=store,
                project_id="context-probe",
                task_id=case.case_id,
                session_id=f"{case.case_id}-{mode}",
                run_id=f"{case.case_id}-{mode}",
                plan_version=0,
                workspace_root=workspace,
                event_sink=sink,
            )
            tools = [answer_tool_schema(case.probe)]
            latest_messages: list[dict[str, Any]] = []
            for event in case.events:
                _replay_event(buffer, event)
                prepared = manager.prepare(buffer, tools, step=event.step)
                latest_messages = prepared.messages
                if prepared.action == ContextPreparationAction.BUDGET_EXHAUSTED:
                    return _handle_budget_exhausted(
                        case=case,
                        mode=mode,
                        manager=manager,
                        buffer=buffer,
                        preparation=prepared,
                        runtime_context_config=runtime_context_config,
                        latest_messages=latest_messages,
                        started=started,
                    )
            required_answer_tool = ANSWER_TOOL_NAMES[case.probe]
            buffer.add_user_reminder(
                f"{case.final_query}\n\n"
                f"Call {required_answer_tool} exactly once with arguments matching its JSON schema. "
                "Do not return a prose answer."
            )
            prepared = manager.prepare(buffer, tools, step=len(case.events) + 1)
            latest_messages = prepared.messages
            latest_estimated = prepared.budget_after.estimated_message_tokens + prepared.budget_after.estimated_tool_schema_tokens
            if prepared.action == ContextPreparationAction.BUDGET_EXHAUSTED:
                return _handle_budget_exhausted(
                    case=case,
                    mode=mode,
                    manager=manager,
                    buffer=buffer,
                    preparation=prepared,
                    runtime_context_config=runtime_context_config,
                    latest_messages=latest_messages,
                    started=started,
                )
            if provider is None:
                provider = self._provider(app_config)
            (
                predicted,
                raw_text,
                protocol_error,
                provider_error,
                usage,
                answer_provider_calls,
                protocol_retry_count,
                retry_budget_exhausted,
            ) = self._ask_model(case, buffer, manager, provider, tools, latest_messages)
            metrics = manager.metrics(buffer)
            input_tokens = int(usage.get("input_tokens") or latest_estimated)
            manager.record_actual_usage(
                step=len(case.events) + 1, estimated_input_tokens=latest_estimated, actual_input_tokens=input_tokens
            )
            return ProbePrediction(
                case_id=case.case_id,
                probe=case.probe,
                mode=mode,
                predicted=predicted,
                raw_final_text=raw_text,
                protocol_valid=predicted is not None and protocol_error is None,
                error_type="context_budget_exhausted" if retry_budget_exhausted else None,
                provider_error=provider_error,
                protocol_error=protocol_error,
                estimated_input_tokens=latest_estimated,
                actual_input_tokens=input_tokens,
                output_tokens=int(usage.get("output_tokens") or 0),
                compactor_input_tokens=int(metrics.get("compactor_input_tokens") or 0),
                compactor_output_tokens=int(metrics.get("compactor_output_tokens") or 0),
                latency_seconds=time.perf_counter() - started,
                context_reset_count=int(metrics.get("context_reset_count") or 0),
                recent_window_trim_count=int(metrics.get("recent_window_trim_count") or 0),
                deterministic_prune_count=int(metrics.get("deterministic_prune_count") or 0),
                structured_reset_count=int(metrics.get("structured_reset_count") or 0),
                structured_compaction_count=int(metrics.get("structured_compaction_count") or 0),
                stale_item_count=int(metrics.get("stale_item_count") or 0),
                superseded_item_count=int(metrics.get("superseded_item_count") or 0),
                estimated_tokens_removed=int(metrics.get("estimated_tokens_removed") or 0),
                context_budget_exhausted=bool(metrics.get("context_budget_exhausted") or retry_budget_exhausted),
                answer_provider_calls=answer_provider_calls,
                protocol_retry_count=protocol_retry_count,
                required_answer_tool=ANSWER_TOOL_NAMES[case.probe],
                compactor_provider_calls=int(getattr(manager.compactor, "model_call_count", 0)),
                runtime_context_config=runtime_context_config,
                prompt_messages=latest_messages if self.eval_config.save_prompts else None,
            )

    def _ask_model(
        self,
        case: ProbeCase,
        buffer: ContextBuffer,
        manager: ContextLifecycleManager,
        provider: ModelProvider,
        tools: list[dict[str, Any]],
        initial_messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None, str | None, str | None, dict[str, int], int, int, bool]:
        messages = initial_messages
        raw_text: str | None = None
        usage: dict[str, int] = {}
        last_error: str | None = None
        answer_provider_calls = 0
        protocol_retry_count = 0
        required_tool_name = ANSWER_TOOL_NAMES[case.probe]
        tool_choice = {"type": "function", "function": {"name": required_tool_name}}
        max_retries = 1
        for retry in range(max_retries + 1):
            try:
                answer_provider_calls += 1
                response = provider.generate(messages, tools, tool_choice=tool_choice)
            except ToolArgumentsProtocolError as exc:
                last_error = str(exc)
                if retry >= max_retries:
                    break
                protocol_retry_count += 1
                buffer.add_protocol_correction(
                    "Your previous response did not call the required answer tool.\n"
                    f"Call {required_tool_name} exactly once with arguments matching\n"
                    "its JSON schema. Do not return a prose answer."
                )
                prepared = manager.prepare(buffer, tools, step=len(case.events) + 2 + retry)
                if prepared.action == ContextPreparationAction.BUDGET_EXHAUSTED:
                    return None, raw_text, "context_budget_exhausted", None, usage, answer_provider_calls, protocol_retry_count, True
                messages = prepared.messages
                continue
            except Exception as exc:
                return None, raw_text, None, str(exc), usage, answer_provider_calls, protocol_retry_count, False
            usage = response.usage
            if response.final_answer is not None:
                raw_text = response.final_answer.content
            predicted, error = parse_probe_answer(case.probe, response)
            if error is None:
                return predicted, raw_text, None, None, usage, answer_provider_calls, protocol_retry_count, False
            last_error = error
            if retry >= max_retries:
                break
            protocol_retry_count += 1
            buffer.add_protocol_correction(
                "Your previous response did not call the required answer tool.\n"
                f"Call {required_tool_name} exactly once with arguments matching\n"
                "its JSON schema. Do not return a prose answer."
            )
            prepared = manager.prepare(buffer, tools, step=len(case.events) + 2 + retry)
            if prepared.action == ContextPreparationAction.BUDGET_EXHAUSTED:
                return None, raw_text, "context_budget_exhausted", None, usage, answer_provider_calls, protocol_retry_count, True
            messages = prepared.messages
        return None, raw_text, last_error, None, usage, answer_provider_calls, protocol_retry_count, False


def _replay_event(buffer: ContextBuffer, event) -> None:
    arguments = json.dumps(event.tool_arguments, ensure_ascii=False, sort_keys=True)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": event.event_id,
                "type": "function",
                "function": {"name": event.tool_name, "arguments": arguments},
            }
        ],
    }
    tool_message = {
        "role": "tool",
        "tool_call_id": event.event_id,
        "name": event.tool_name,
        "content": json.dumps(event.tool_result, ensure_ascii=False, sort_keys=True),
    }
    buffer.add_assistant_tool_turn(assistant_message, step=event.step, metadata=event.metadata)
    buffer.add_tool_result(tool_message)
    buffer.finalize_turn()


def _seed(case: ProbeCase) -> TaskContextSeed:
    return TaskContextSeed(
        project_id="context-probe",
        project_objective="Evaluate context lifecycle behavior with synthetic coding traces.",
        plan_version=0,
        task_id=case.case_id,
        task_title=f"{case.probe} probe",
        task_objective=case.final_query,
        task_status="in_progress",
        acceptance_criteria=["Return the structured probe answer using the provided tool."],
        dependency_summaries=["- none"],
        attempts=0,
        progress_summary=f"Target tokens: {case.target_length_tokens}; estimated tokens: {case.estimated_length_tokens}",
    )


def _handle_budget_exhausted(
    *,
    case: ProbeCase,
    mode: ProbeMode,
    manager: ContextLifecycleManager,
    buffer: ContextBuffer,
    preparation: ContextPreparationResult,
    runtime_context_config: dict[str, Any],
    latest_messages: list[dict[str, Any]],
    started: float,
) -> ProbePrediction:
    metrics = manager.metrics(buffer)
    return ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode=mode,
        protocol_valid=False,
        error_type="context_budget_exhausted",
        protocol_error="context_budget_exhausted",
        estimated_input_tokens=preparation.budget_after.estimated_total_tokens,
        actual_input_tokens=0,
        output_tokens=0,
        latency_seconds=time.perf_counter() - started,
        compactor_input_tokens=int(metrics.get("compactor_input_tokens") or 0),
        compactor_output_tokens=int(metrics.get("compactor_output_tokens") or 0),
        context_reset_count=int(metrics.get("context_reset_count") or 0),
        recent_window_trim_count=int(metrics.get("recent_window_trim_count") or 0),
        deterministic_prune_count=int(metrics.get("deterministic_prune_count") or 0),
        structured_reset_count=int(metrics.get("structured_reset_count") or 0),
        structured_compaction_count=int(metrics.get("structured_compaction_count") or 0),
        stale_item_count=int(metrics.get("stale_item_count") or 0),
        superseded_item_count=int(metrics.get("superseded_item_count") or 0),
        estimated_tokens_removed=int(metrics.get("estimated_tokens_removed") or 0),
        context_budget_exhausted=True,
        answer_provider_calls=0,
        required_answer_tool=ANSWER_TOOL_NAMES[case.probe],
        compactor_provider_calls=int(getattr(manager.compactor, "model_call_count", 0)),
        runtime_context_config=runtime_context_config,
        prompt_messages=latest_messages,
    )


def _runtime_context_config(app_config: AppConfig, source_config_path: Path) -> dict[str, Any]:
    context = app_config.context
    return {
        "mode": context.mode,
        "model_context_limit": context.model_context_limit,
        "reserve_output_tokens": context.reserve_output_tokens,
        "safety_margin_tokens": context.safety_margin_tokens,
        "trigger_ratio": context.trigger_ratio,
        "hard_stop_ratio": context.hard_stop_ratio,
        "recent_full_turns": context.recent_full_turns,
        "source_config_path": str(source_config_path),
    }


def _load_scripted_responses(path: Path) -> list[ModelResponse]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    responses: list[ModelResponse] = []
    for item in raw:
        if item.get("final_answer") is not None:
            responses.append(ModelResponse(final_answer=FinalAnswer(content=item["final_answer"]), usage=item.get("usage") or {}))
        else:
            calls = [ToolCall.model_validate(call) for call in item.get("tool_calls", [])]
            responses.append(ModelResponse(tool_calls=calls, usage=item.get("usage") or {}))
    return responses
