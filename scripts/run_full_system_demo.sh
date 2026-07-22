#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/full_system.yaml"
PROJECT_ID="${LONGRUN_PROJECT_ID:-full-system-demo-$(date +%Y%m%d%H%M%S)-$$}"
RESULT_ROOT=".runs/full_system_demo"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$RESULT_ROOT"

LONGRUN_PROJECT_ID="$PROJECT_ID" "$PYTHON_BIN" - "$CONFIG" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from longrun_agent.config import load_config
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall

project_id = os.environ["LONGRUN_PROJECT_ID"]
config_path = sys.argv[1]
root = Path(".runs/full_system_demo")
workspace = root / "workspaces" / project_id
fixture = Path("examples/verification_bench/full_fix")
if workspace.exists():
    raise SystemExit(f"workspace already exists for project_id={project_id}: {workspace}")
shutil.copytree(fixture, workspace, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".runs"))

config = load_config(config_path)
config.workspace.root = workspace.resolve()
config.state.root = (root / "projects").resolve()
config.telemetry.run_root = (root / "telemetry").resolve()
config.knowledge.root = (root / "knowledge").resolve()
config.verification.store_root = config.state.root

fixed = (
    "def classify_number(value: int) -> str:\n"
    "    if value < 0:\n"
    "        return \"negative\"\n"
    "    if value == 0:\n"
    "        return \"zero\"\n"
    "    return \"positive\"\n"
)
responses = [
    ModelResponse(
        tool_calls=[
            ToolCall(
                id="plan",
                name="submit_plan",
                arguments={
                    "project_summary": "fix negative classification",
                    "tasks": [
                        {
                            "key": "fix",
                            "title": "Fix classifier",
                            "objective": "Make negative values classify as negative while preserving public behavior.",
                            "acceptance_criteria": ["contract verification passes"],
                            "depends_on_keys": [],
                        }
                    ],
                },
            )
        ]
    ),
    ModelResponse(
        tool_calls=[
            ToolCall(
                id="premature-completion",
                name="request_task_completion",
                arguments={
                    "summary": "premature completion without implementation evidence",
                    "acceptance_criteria_addressed": ["contract verification passes"],
                },
            )
        ]
    ),
    ModelResponse(tool_calls=[ToolCall(id="write-fix", name="write_file", arguments={"path": "app.py", "content": fixed})]),
    ModelResponse(
        tool_calls=[
            ToolCall(id="pytest-public", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q", "tests/test_public.py"]})
        ]
    ),
    ModelResponse(
        tool_calls=[
            ToolCall(
                id="complete-fixed",
                name="request_task_completion",
                arguments={
                    "summary": "negative classification fixed and public pytest passes",
                    "acceptance_criteria_addressed": ["contract verification passes"],
                },
            )
        ]
    ),
]

provider = FakeModelProvider(responses)
orchestrator = ProjectOrchestrator(config, provider, project_id=project_id)
first = orchestrator.start((workspace / "TASK.md").read_text(encoding="utf-8"))
second = orchestrator.resume(project_id) if first.status != "verified" else first

print(f"project_id: {project_id}")
print(f"status: {second.status}")
print(f"verification_verdict: {second.verification_verdict}")
print(f"state_path: {second.state_path}")
print(f"result_root: {root.resolve()}")
if second.status != "verified":
    raise SystemExit(1)
PY
