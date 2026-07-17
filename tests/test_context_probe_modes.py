from pathlib import Path

import yaml

from longrun_agent.config import AppConfig
from longrun_agent.context_probes.config import ProbeEvalConfig
from longrun_agent.context_probes.executor import ProbeExecutor
from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall


class AnswerProvider(ModelProvider):
    def __init__(self, arguments):
        self.arguments = arguments

    def generate(self, messages: list[dict], tools: list[dict], tool_choice=None) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall(id="answer", name=tools[0]["function"]["name"], arguments=self.arguments)])


def test_same_case_runs_through_all_context_modes(tmp_path):
    config = _eval_config(tmp_path)
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=14)[0]
    executor = ProbeExecutor(config, provider_factory=lambda _app: AnswerProvider(case.expected))

    results = {mode: executor.run_case(case, mode) for mode in config.modes}

    assert set(results) == set(config.modes)
    assert all(result.case.case_id == case.case_id for result in results.values())
    assert results["full_history"].prediction.deterministic_prune_count == 0
    assert results["full_history"].prediction.context_reset_count == 0
    assert results["recent_window"].prediction.estimated_tokens_removed > 0
    assert results["recent_window"].prediction.recent_window_trim_count > 0
    assert results["recent_window"].prediction.deterministic_prune_count == 0
    assert results["recent_window"].prediction.structured_reset_count == 0
    assert results["deterministic_prune"].prediction.deterministic_prune_count > 0
    assert results["deterministic_prune"].prediction.context_reset_count == 0
    assert results["deterministic_prune"].prediction.structured_reset_count == 0
    assert results["structured_reset"].prediction.context_reset_count >= 1
    assert results["structured_reset"].prediction.structured_reset_count >= 1
    assert results["structured_reset"].prediction.structured_compaction_count >= 1


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
