# Long-horizon architecture video demo

This demonstration complements the short core-loop inventory demo. It uses a staged `workflow_service` fixture and a five-task dependency DAG to exercise the higher-level project architecture within a bounded recording window. The tested foundation is pre-completed; the Agent must still implement the service lifecycle, transitive propagation, persistent idempotency, CLI, and final integration.

## What it demonstrates

- deterministic dependency-aware project planning;
- persisted project/task state across multiple Agent Sessions;
- focused per-task tests and final project verification;
- deterministic structured handoff between Sessions and context reset if needed;
- episode recording plus conditional reflection, memory, and skill lifecycle recording;
- a frozen project verification contract;
- isolated public and hidden tests plus test/configuration integrity checks;
- independent post-run oracle validation and a machine-readable summary.

The static plan is intentional: it keeps the demonstration reproducible while still exercising scheduling, dependency release, Session handoff, persistence, and verification. The expected runtime is about 30 minutes with the configured model. The 55-minute project budget and 60-minute shell watchdog are safety limits, not duration targets. Actual duration still depends on model and endpoint latency. Adaptive decomposition remains available when a task repeatedly fails.

## Ubuntu run

Prepare the environment before recording so the API key is never visible:

```bash
cd /path/to/Coding-Agent
source ~/venvs/coding-agent/bin/activate
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export MODEL_NAME="your-tool-calling-model"
clear
```

Run the complete case with a fresh project ID:

```bash
PROJECT_ID=architecture-bounded-01 bash scripts/run_long_horizon_real_api.sh
```

To use a specific Python executable or project ID:

```bash
LONGRUN_PYTHON_BIN=/path/to/python \
PROJECT_ID=architecture-bounded-01 \
bash scripts/run_long_horizon_real_api.sh
```

The run creates an isolated workspace under `.runs/long_horizon_real_api/workspaces/<project-id>` and never edits the fixture. The final console output prints paths for persisted state, telemetry, the independent oracle, and `summary.json`.

## Resume after interruption

Use the exact resume command printed by the runner. For a known project ID, the equivalent command is:

```bash
PROJECT_ID=architecture-bounded-01 \
RESUME_PROJECT=1 \
MINIMUM_SECONDS=0 \
bash scripts/run_long_horizon_real_api.sh
```

The resume mode does not copy or reset the fixture. It reuses the persisted workspace, task states, Session history, plan, and knowledge records, then regenerates the independent oracle and summary after the resumed run.

## Suggested edited sequence

1. Show the baseline (`47 passed, 11 failed`) and the five-node plan.
2. Show the project tree after several tasks have completed.
3. Show two adjacent Sessions and one structured handoff/context reset.
4. Interrupt and resume once to demonstrate state recovery.
5. Show knowledge episodes, any generated reflection/memory artifacts, and project metrics.
6. Show frozen-contract verification in an isolated copy.
7. End on `oracle_verified: true`, `integrity_passed: true`, and the run summary.

Do not attempt to show the complete raw run in the submitted video. Record the long run separately, then edit representative checkpoints together with a short architecture narration.
