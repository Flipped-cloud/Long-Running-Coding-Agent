#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pytest \
  tests/test_context_budget.py \
  tests/test_context_buffer.py \
  tests/test_context_pruner.py \
  tests/test_context_reset_project_semantics.py \
  tests/test_context_tool_metadata.py \
  tests/test_structured_compactor.py \
  tests/test_context_probe_cli.py \
  tests/test_context_probe_evaluator.py \
  tests/test_context_probe_executor.py \
  tests/test_context_probe_generator.py \
  tests/test_context_probe_modes.py \
  tests/test_context_probe_pairing.py \
  tests/test_context_probe_reporting.py \
  tests/test_context_probe_schemas.py \
  -q

python -m compileall -q src

python evals/context_lifecycle/runner.py \
  --config evals/context_lifecycle/config.yaml \
  --probe all \
  --lengths 2048,4096,8192,12288 \
  --samples 3 \
  --seed 42 \
  --dry-run \
  --output-dir .runs/context_evals/preflight_dry_run

python - <<'PY'
from pathlib import Path

from longrun_agent.config import AppConfig
from longrun_agent.context_probes.runner import run_probe
from longrun_agent.model.base import ModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall


class PreflightProvider(ModelProvider):
    def generate(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        name = tools[0]["function"]["name"]
        if name == "submit_position_answer":
            arguments = {"constraints": ["preflight-wrong"]}
        elif name == "submit_multi_constraint_answer":
            arguments = {"constraints": ["preflight-wrong"]}
        elif name == "submit_state_tracking_answer":
            arguments = {
                "plan_version": -1,
                "file_hash": "preflight",
                "active_task": "preflight",
                "verification_state": "unknown",
                "latest_valid_test": None,
            }
        elif name == "submit_aggregation_answer":
            arguments = {
                "current_test": "preflight",
                "unresolved_failures": [],
                "stale_results_rejected": False,
                "modified_files": [],
                "code_epoch": -1,
            }
        else:
            arguments = {}
        return ModelResponse(
            tool_calls=[ToolCall(id="preflight-answer", name=name, arguments=arguments)],
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def factory(config: AppConfig) -> ModelProvider:
    return PreflightProvider()


run_probe(
    config_path=Path("evals/context_lifecycle/config.yaml"),
    probe="position",
    lengths=[12288],
    samples=1,
    seed=42,
    provider_factory=factory,
    output_dir=Path(".runs/context_evals/preflight_activation"),
)
PY

python evals/context_lifecycle/activation_check.py \
  --predictions .runs/context_evals/preflight_activation/predictions.jsonl \
  --output .runs/context_evals/preflight_activation/activation_report.json

echo "Context Lifecycle Preflight Passed"
