from pathlib import Path

import yaml

from longrun_agent.config import AppConfig
from longrun_agent.context_probes.config import ProbeEvalConfig, load_mode_app_config
from longrun_agent.context_probes.executor import ProbeExecutor
from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall


class AnswerProvider(ModelProvider):
    def __init__(self, arguments):
        self.arguments = arguments
        self.call_count = 0
        self.tool_choices = []

    def generate(self, messages: list[dict], tools: list[dict], tool_choice=None) -> ModelResponse:
        self.call_count += 1
        self.tool_choices.append(tool_choice)
        name = tools[0]["function"]["name"]
        return ModelResponse(tool_calls=[ToolCall(id=f"answer-{self.call_count}", name=name, arguments=self.arguments)])


class ScriptedProvider(ModelProvider):
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.call_count = 0
        self.tool_choices = []

    def generate(self, messages: list[dict], tools: list[dict], tool_choice=None) -> ModelResponse:
        self.call_count += 1
        self.tool_choices.append(tool_choice)
        return self.responses.pop(0)


def test_executor_replays_case_and_calls_provider(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=12)[0]
    executor = ProbeExecutor(config, provider_factory=lambda _app: AnswerProvider(case.expected))

    result = executor.run_case(case, "full_history")

    assert result.exact_match
    assert result.prediction.protocol_valid
    assert result.prediction.actual_input_tokens > 0
    assert result.prediction.required_answer_tool == "submit_position_answer"


def test_executor_forces_named_answer_tool_choice(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=12)[0]
    provider = AnswerProvider(case.expected)
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid
    assert provider.tool_choices == [{"type": "function", "function": {"name": "submit_position_answer"}}]


def test_executor_protocol_error_does_not_guess_free_text(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=13)[0]

    class TextProvider(ModelProvider):
        def generate(self, messages: list[dict], tools: list[dict], tool_choice=None) -> ModelResponse:
            return ModelResponse(final_answer=FinalAnswer(content="the answer is in prose"))

    executor = ProbeExecutor(config, provider_factory=lambda _app: TextProvider())

    result = executor.run_case(case, "full_history")

    assert not result.exact_match
    assert not result.prediction.protocol_valid
    assert result.prediction.protocol_error


def test_executor_retries_once_after_prose_answer_then_accepts_tool_call(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=21)[0]
    provider = ScriptedProvider(
        [
            ModelResponse(final_answer=FinalAnswer(content="prose")),
            ModelResponse(tool_calls=[ToolCall(id="answer-2", name="submit_position_answer", arguments=case.expected)]),
        ]
    )
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid is True
    assert result.prediction.protocol_retry_count == 1
    assert result.prediction.answer_provider_calls == 2


def test_executor_retries_once_then_keeps_protocol_failure(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=22)[0]
    provider = ScriptedProvider(
        [
            ModelResponse(final_answer=FinalAnswer(content="first prose")),
            ModelResponse(final_answer=FinalAnswer(content="second prose")),
        ]
    )
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid is False
    assert result.prediction.protocol_retry_count == 1
    assert result.prediction.answer_provider_calls == 2
    assert result.prediction.predicted is None


def test_executor_retries_after_wrong_answer_tool(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=23)[0]
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall(id="wrong", name="submit_aggregation_answer", arguments={})]),
            ModelResponse(tool_calls=[ToolCall(id="answer", name="submit_position_answer", arguments=case.expected)]),
        ]
    )
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid is True
    assert result.prediction.protocol_retry_count == 1
    assert result.prediction.answer_provider_calls == 2


def test_executor_retries_after_schema_error(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=24)[0]
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall(id="bad", name="submit_position_answer", arguments={"constraints": "not-a-list"})]),
            ModelResponse(tool_calls=[ToolCall(id="answer", name="submit_position_answer", arguments=case.expected)]),
        ]
    )
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid is True
    assert result.prediction.protocol_retry_count == 1
    assert result.prediction.answer_provider_calls == 2


def test_budget_exhausted_before_protocol_retry_skips_second_provider_call(tmp_path):
    measuring_config = _single_mode_eval_config(
        tmp_path / "measure",
        mode="full_history",
        context={
            "model_context_limit": 20000,
            "reserve_output_tokens": 128,
            "safety_margin_tokens": 32,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.95,
        },
    )
    original = generate_cases(probe="position", lengths=[2048], samples=1, seed=25)[0]
    case = original.model_copy(update={"events": [original.events[0]], "final_query": "final query " + "x" * 2500})
    measured = ProbeExecutor(measuring_config, provider_factory=lambda _app: AnswerProvider(case.expected)).run_case(case, "full_history")
    tight_limit = int((measured.prediction.estimated_input_tokens + 180) / 0.95)
    config = _single_mode_eval_config(
        tmp_path / "tight",
        mode="full_history",
        context={
            "model_context_limit": tight_limit,
            "reserve_output_tokens": 128,
            "safety_margin_tokens": 32,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.95,
        },
    )
    provider = ScriptedProvider(
        [
            ModelResponse(final_answer=FinalAnswer(content="prose")),
            ModelResponse(tool_calls=[ToolCall(id="answer", name="submit_position_answer", arguments=case.expected)]),
        ]
    )
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.error_type == "context_budget_exhausted"
    assert result.prediction.protocol_retry_count == 1
    assert result.prediction.answer_provider_calls == 1
    assert provider.call_count == 1


def test_executor_stops_without_provider_when_budget_exhausted(tmp_path):
    config = _single_mode_eval_config(
        tmp_path,
        mode="full_history",
        context={
            "model_context_limit": 4096,
            "reserve_output_tokens": 1024,
            "safety_margin_tokens": 256,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.8,
        },
    )
    case = generate_cases(probe="position", lengths=[8192], samples=1, seed=17)[0]
    provider = AnswerProvider(case.expected)
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert not result.prediction.protocol_valid
    assert result.prediction.error_type == "context_budget_exhausted"
    assert result.prediction.output_tokens == 0
    assert result.prediction.actual_input_tokens == 0
    assert result.prediction.context_budget_exhausted is True
    assert result.prediction.runtime_context_config["model_context_limit"] == 4096
    assert result.prediction.runtime_context_config["reserve_output_tokens"] == 1024
    assert provider.call_count == 0


def test_mode_app_config_is_runtime_context_source(tmp_path):
    config = _single_mode_eval_config(
        tmp_path,
        mode="full_history",
        context={
            "model_context_limit": 4096,
            "reserve_output_tokens": 1024,
            "safety_margin_tokens": 256,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.8,
        },
    )
    config.generation.target_context_limit = 99999

    app_config = load_mode_app_config(config, "full_history", force_fake_provider=True)

    assert app_config.context.model_context_limit == 4096
    assert app_config.context.reserve_output_tokens == 1024


def test_budget_exhausted_after_final_query_does_not_call_provider(tmp_path):
    config = _single_mode_eval_config(
        tmp_path,
        mode="full_history",
        context={
            "model_context_limit": 1200,
            "reserve_output_tokens": 100,
            "safety_margin_tokens": 50,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.8,
        },
    )
    original = generate_cases(probe="position", lengths=[2048], samples=1, seed=18)[0]
    case = original.model_copy(update={"events": [original.events[0]], "final_query": "final query " + "x" * 5000})
    provider = AnswerProvider(case.expected)
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.error_type == "context_budget_exhausted"
    assert provider.call_count == 0


def test_budget_exhausted_during_event_replay_stops_later_events(tmp_path):
    config = _single_mode_eval_config(
        tmp_path,
        mode="full_history",
        context={
            "model_context_limit": 1200,
            "reserve_output_tokens": 100,
            "safety_margin_tokens": 50,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.8,
        },
    )
    case = generate_cases(probe="position", lengths=[8192], samples=1, seed=19)[0]
    provider = AnswerProvider(case.expected)
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")
    prompt_blob = str(result.prediction.prompt_messages)

    assert result.prediction.error_type == "context_budget_exhausted"
    assert provider.call_count == 0
    assert case.events[-1].event_id not in prompt_blob


def test_long_case_with_larger_runtime_budget_still_calls_provider(tmp_path):
    config = _single_mode_eval_config(
        tmp_path,
        mode="full_history",
        context={
            "model_context_limit": 20000,
            "reserve_output_tokens": 512,
            "safety_margin_tokens": 128,
            "trigger_ratio": 0.5,
            "hard_stop_ratio": 0.95,
        },
    )
    case = generate_cases(probe="position", lengths=[8192], samples=1, seed=20)[0]
    provider = AnswerProvider(case.expected)
    executor = ProbeExecutor(config, provider_factory=lambda _app: provider)

    result = executor.run_case(case, "full_history")

    assert result.prediction.protocol_valid is True
    assert result.prediction.error_type is None
    assert provider.call_count == 1


def _eval_config(tmp_path: Path) -> ProbeEvalConfig:
    mode_configs = {}
    for mode in ["full_history", "recent_window", "deterministic_prune", "structured_reset"]:
        path = tmp_path / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(_app_config(mode, tmp_path)), encoding="utf-8")
        mode_configs[mode] = path
    return ProbeEvalConfig(
        seed=7,
        samples=1,
        lengths=[2048],
        modes=["full_history", "recent_window", "deterministic_prune", "structured_reset"],
        mode_configs=mode_configs,
        output_root=tmp_path / "out",
        structured_reset_compactor="deterministic",
    )


def _single_mode_eval_config(tmp_path: Path, *, mode: str, context: dict) -> ProbeEvalConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{mode}.yaml"
    payload = _app_config(mode, tmp_path)
    payload["context"].update(context)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return ProbeEvalConfig(
        seed=42,
        samples=1,
        lengths=[8192],
        modes=[mode],
        mode_configs={mode: path},
        output_root=tmp_path / "out",
        structured_reset_compactor="deterministic",
    )


def _app_config(mode: str, tmp_path: Path) -> dict:
    return AppConfig(
        model={"provider": "fake"},
        workspace={"root": tmp_path / "workspace"},
        state={"root": tmp_path / "state"},
        context={
            "mode": mode,
            "model_context_limit": 3000,
            "reserve_output_tokens": 128,
            "safety_margin_tokens": 32,
            "trigger_ratio": 0.50,
            "hard_stop_ratio": 0.95,
            "recent_full_turns": 1,
            "structured_handoff": {"use_model": False, "fallback_deterministic": True},
        },
    ).model_dump(mode="json")
