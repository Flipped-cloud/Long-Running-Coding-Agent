from __future__ import annotations

from pathlib import Path
from typing import Any

from longrun_agent.config import ContextConfig
from longrun_agent.context.assembler import ContextAssembler
from longrun_agent.context.budget import ContextBudgetManager
from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.compactor import StructuredContextCompactor
from longrun_agent.context.pruner import DeterministicContextPruner
from longrun_agent.context.schema import ContextPreparationAction, ContextPreparationResult, ContextSegmentRecord, TaskContextSeed
from longrun_agent.context.stale_tracker import StaleTracker
from longrun_agent.context.telemetry import ContextEventSink, budget_payload, prune_payload
from longrun_agent.context.token_counter import build_token_counter
from longrun_agent.model.base import ModelProvider
from longrun_agent.state.store import ProjectStateStore


class ContextLifecycleManager:
    def __init__(
        self,
        config: ContextConfig,
        *,
        seed: TaskContextSeed | None = None,
        model: ModelProvider | None = None,
        store: ProjectStateStore | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        plan_version: int = 0,
        workspace_root: Path | None = None,
        event_sink: ContextEventSink | None = None,
    ):
        self.config = config
        self.seed = seed
        self.project_id = project_id or (seed.project_id if seed else None)
        self.task_id = task_id or (seed.task_id if seed else None)
        self.session_id = session_id
        self.run_id = run_id
        self.plan_version = plan_version
        self.workspace_root = workspace_root
        self.store = store
        self.event_sink = event_sink
        self.counter, self.tokenizer_fallback = build_token_counter(config.token_counter)
        self.budget = ContextBudgetManager(config, self.counter)
        self.tracker = StaleTracker()
        self.pruner = DeterministicContextPruner(config, self.budget)
        self.compactor = StructuredContextCompactor(config, model)
        self.assembler = ContextAssembler(config)
        self.max_estimated_input_tokens = 0
        self.max_actual_input_tokens = 0
        self.max_context_usage_ratio = 0.0
        self.context_reset_count = 0
        self.recent_window_trim_count = 0
        self.deterministic_prune_count = 0
        self.structured_reset_count = 0
        self.structured_compaction_count = 0
        self.pruned_item_count = 0
        self.stale_item_count = 0
        self.superseded_item_count = 0
        self.estimated_tokens_removed = 0
        self.context_budget_exhausted = False
        self.latest_handoff_id: str | None = None
        if self.tokenizer_fallback:
            self._emit("context_tokenizer_fallback", {"reason": "tiktoken unavailable"})

    def initial_task_message(self, fallback_task: str) -> dict[str, str]:
        if self.seed is None:
            return {"role": "user", "content": fallback_task}
        return self.assembler.task_anchor_message(self.seed)

    def prepare(self, buffer: ContextBuffer, tools: list[dict[str, Any]], *, step: int) -> ContextPreparationResult:
        self.tracker.refresh(buffer)
        original_messages = buffer.export_messages()
        before = self.budget.measure(original_messages, tools)
        self._record_budget(before)
        self._emit("context_budget_measured", self._base_payload(step=step, snapshot=before))
        if before.trigger_exceeded:
            self._emit("context_trigger_exceeded", self._base_payload(step=step, snapshot=before))

        if self.config.mode == "full_history":
            if self.budget.should_hard_stop(before):
                return self._budget_exhausted(original_messages, before, step, "full_history hard_stop_ratio exceeded")
            return self._result(ContextPreparationAction.UNCHANGED, original_messages, before, before, step)

        if self.config.mode == "recent_window":
            outcome = self.pruner.recent_window(buffer, tools, before_tokens=before.estimated_total_tokens)
            after = self.budget.measure(outcome.messages, tools)
            self._record_recent_window(outcome.report)
            if self.budget.should_hard_stop(after):
                return self._budget_exhausted(outcome.messages, before, step, "recent_window hard_stop_ratio exceeded", after)
            return self._result(ContextPreparationAction.PRUNED, outcome.messages, before, after, step, pruned=True)

        if self.config.mode == "structured_reset":
            if not before.trigger_exceeded:
                return self._result(ContextPreparationAction.UNCHANGED, original_messages, before, before, step)
            if not buffer.all_turns():
                if self.budget.should_hard_stop(before):
                    return self._budget_exhausted(original_messages, before, step, "structured_reset hard_stop_ratio exceeded")
                return self._result(ContextPreparationAction.UNCHANGED, original_messages, before, before, step)
            return self._perform_structured_reset(buffer, tools, before, step)

        self._emit("context_pruning_started", self._base_payload(step=step, snapshot=before))
        outcome = self.pruner.prune(buffer, tools, before_tokens=before.estimated_total_tokens)
        after_prune = self.budget.measure(outcome.messages, tools)
        self._record_prune(outcome.report)
        self._emit("context_pruning_finished", prune_payload(self._base_payload(step=step, snapshot=after_prune), outcome.report))
        if self.config.mode == "deterministic_prune":
            if self.budget.should_hard_stop(after_prune):
                return self._budget_exhausted(outcome.messages, before, step, "deterministic_prune hard_stop_ratio exceeded", after_prune)
            return self._result(ContextPreparationAction.PRUNED, outcome.messages, before, after_prune, step, pruned=True)

    def _perform_structured_reset(self, buffer: ContextBuffer, tools: list[dict[str, Any]], before, step: int) -> ContextPreparationResult:
        if self.seed is None or self.session_id is None:
            return self._budget_exhausted(
                buffer.export_messages(), before, step, "structured_reset requires context seed and session id", before
            )
        self._emit("context_compaction_requested", self._base_payload(step=step, snapshot=before))
        source_segment = buffer.context_segment_id
        target_segment = source_segment + 1
        handoff = self.compactor.compact(
            seed=self.seed,
            buffer=buffer,
            project_id=self.project_id,
            session_id=self.session_id,
            source_segment_id=source_segment,
            target_segment_id=target_segment,
            plan_version=self.plan_version,
            workspace_root=self.workspace_root,
        )
        self.structured_compaction_count += 1
        self.structured_reset_count += 1
        self.latest_handoff_id = handoff.handoff_id
        if self.compactor.fallback_count:
            self._emit(
                "context_handoff_fallback_created",
                {"handoff_id": handoff.handoff_id, **self._base_payload(step=step, snapshot=before)},
            )
        if self.store and handoff.project_id:
            self.store.save_handoff(handoff)
        self._emit("context_handoff_created", {"handoff_id": handoff.handoff_id, **self._base_payload(step=step, snapshot=before)})
        new_segment = buffer.reset_to(
            task_anchor_message=self.assembler.task_anchor_message(self.seed),
            handoff_message=self.assembler.handoff_message(handoff),
            instruction_message=self.assembler.current_instruction_message(self.seed),
        )
        self.context_reset_count = buffer.context_reset_count
        reset_messages = buffer.export_messages()
        after_reset = self.budget.measure(reset_messages, tools)
        if self.store:
            self.store.append_context_segment(
                ContextSegmentRecord(
                    segment_id=source_segment,
                    run_id=self.run_id or self.session_id or "",
                    project_id=self.project_id,
                    task_id=self.task_id,
                    session_id=self.session_id,
                    end_step=step,
                    reset_reason="trigger_ratio_exceeded",
                    handoff_id=handoff.handoff_id,
                    estimated_tokens_at_close=before.estimated_total_tokens,
                    pruned_item_count=self.pruned_item_count,
                    stale_item_count=self.stale_item_count,
                    compacted_item_count=0,
                )
            )
            self.store.append_context_segment(
                ContextSegmentRecord(
                    segment_id=new_segment,
                    run_id=self.run_id or self.session_id or "",
                    project_id=self.project_id,
                    task_id=self.task_id,
                    session_id=self.session_id,
                    start_step=step,
                    reset_reason="context_reset_start",
                    handoff_id=handoff.handoff_id,
                    estimated_tokens_at_close=after_reset.estimated_total_tokens,
                )
            )
        self._record_budget(after_reset)
        self._emit("context_segment_closed", {"handoff_id": handoff.handoff_id, **self._base_payload(step=step, snapshot=before)})
        self._emit("context_segment_started", {"handoff_id": handoff.handoff_id, **self._base_payload(step=step, snapshot=after_reset)})
        self._emit("context_reset", {"handoff_id": handoff.handoff_id, **self._base_payload(step=step, snapshot=after_reset)})
        if self.budget.should_hard_stop(after_reset):
            return self._budget_exhausted(reset_messages, before, step, "reset context still exceeds hard_stop_ratio", after_reset)
        return ContextPreparationResult(
            action=ContextPreparationAction.RESET,
            messages=reset_messages,
            budget_before=before,
            budget_after=after_reset,
            pruned=True,
            handoff_id=handoff.handoff_id,
            reset_performed=True,
            segment_id=new_segment,
        )

    def record_actual_usage(self, *, step: int, estimated_input_tokens: int, actual_input_tokens: int | None) -> None:
        if actual_input_tokens is None:
            return
        self.max_actual_input_tokens = max(self.max_actual_input_tokens, actual_input_tokens)
        error = actual_input_tokens - estimated_input_tokens
        self._emit(
            "token_estimation_error_recorded",
            {
                "project_id": self.project_id,
                "task_id": self.task_id,
                "session_id": self.session_id,
                "segment_id": None,
                "step": step,
                "estimated_input_tokens": estimated_input_tokens,
                "actual_input_tokens": actual_input_tokens,
                "estimation_error": error,
                "absolute_estimation_error": abs(error),
                "relative_estimation_error": abs(error) / max(1, estimated_input_tokens),
            },
        )

    def metrics(self, buffer: ContextBuffer | None = None) -> dict[str, Any]:
        return {
            "compactor_input_tokens": self.compactor.input_tokens_total,
            "compactor_output_tokens": self.compactor.output_tokens_total,
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
            "max_actual_input_tokens": self.max_actual_input_tokens,
            "max_context_usage_ratio": self.max_context_usage_ratio,
            "context_segment_count": buffer.context_segment_id if buffer is not None else max(1, self.context_reset_count + 1),
            "context_reset_count": self.context_reset_count,
            "recent_window_trim_count": self.recent_window_trim_count,
            "deterministic_prune_count": self.deterministic_prune_count,
            "structured_reset_count": self.structured_reset_count,
            "structured_compaction_count": self.structured_compaction_count,
            "pruned_item_count": self.pruned_item_count,
            "stale_item_count": self.stale_item_count,
            "superseded_item_count": self.superseded_item_count,
            "estimated_tokens_removed": self.estimated_tokens_removed,
            "context_budget_exhausted": self.context_budget_exhausted,
            "latest_context_handoff_id": self.latest_handoff_id,
        }

    def _record_budget(self, snapshot) -> None:
        self.max_estimated_input_tokens = max(
            self.max_estimated_input_tokens, snapshot.estimated_message_tokens + snapshot.estimated_tool_schema_tokens
        )
        self.max_context_usage_ratio = max(self.max_context_usage_ratio, snapshot.usage_ratio)

    def _record_prune(self, report) -> None:
        if report.tokens_removed or report.items_removed:
            self.deterministic_prune_count += 1
        self.pruned_item_count += report.items_removed
        self.stale_item_count += report.stale_items_removed
        self.superseded_item_count += report.superseded_items_removed
        self.estimated_tokens_removed += report.tokens_removed

    def _record_recent_window(self, report) -> None:
        if report.tokens_removed or report.items_removed:
            self.recent_window_trim_count += 1
        self.pruned_item_count += report.items_removed
        self.estimated_tokens_removed += report.tokens_removed

    def _budget_exhausted(self, messages, before, step: int, reason: str, after=None):
        snapshot = after or before
        self.context_budget_exhausted = True
        self._emit("context_budget_exhausted", {"reason": reason, **self._base_payload(step=step, snapshot=snapshot)})
        return ContextPreparationResult(
            action=ContextPreparationAction.BUDGET_EXHAUSTED,
            messages=messages,
            budget_before=before,
            budget_after=snapshot,
            segment_id=1,
            hard_stop_reason=reason,
        )

    def _result(self, action, messages, before, after, step: int, *, pruned: bool = False):
        return ContextPreparationResult(
            action=action, messages=messages, budget_before=before, budget_after=after, pruned=pruned, segment_id=1
        )

    def _base_payload(self, *, step: int, snapshot):
        return budget_payload(
            project_id=self.project_id,
            task_id=self.task_id,
            session_id=self.session_id,
            segment_id=1,
            step=step,
            mode=self.config.mode,
            snapshot=snapshot,
        )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)
