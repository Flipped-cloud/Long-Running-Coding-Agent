# Context Lifecycle Probes

Synthetic model probes for the v0.3 context lifecycle harness.

The probes are not a full RULER benchmark and do not evaluate the whole Coding Agent. They only evaluate whether the context harness preserves task-local coding evidence under different context policies.

## Probes

- `position`: Lost-in-the-Middle style active coding constraint recall at beginning, middle, and end positions.
- `multi_constraint`: RULER-style retrieval of 2, 3, or 5 active constraints with similar superseded distractors.
- `state_tracking`: current plan version, file hash, active task, and stale test state after read/write/bash chains.
- `aggregation`: effective test result, unresolved failure, stale rejection, modified files, and current code epoch.

## Context Modes

Each generated base case is replayed unchanged under:

- `full_history`
- `recent_window`
- `deterministic_prune`
- `structured_reset`

The design is paired: `case-001` has identical events, expected answer, target length, and relevant position in every mode. The only intended variable is the context policy.

## Run

Dry-run validates generation, schema, token length, and output writing without calling a provider:

```bash
longrun-agent eval context \
  --config evals/context_lifecycle/config.yaml \
  --probe all \
  --lengths 2048 \
  --samples 3 \
  --seed 42 \
  --dry-run
```

Real provider run:

```bash
longrun-agent eval context \
  --config evals/context_lifecycle/config.yaml \
  --probe position \
  --lengths 2048,4096,8192 \
  --samples 20 \
  --seed 42 \
  --modes full_history,recent_window,deterministic_prune,structured_reset \
  --output-dir .runs/context_evals/position_main
```

Fake Provider scripts can return deterministic native tool calls for tests:

```bash
longrun-agent eval context \
  --config evals/context_lifecycle/config.yaml \
  --probe position \
  --lengths 2048 \
  --samples 1 \
  --modes full_history \
  --fake-provider-script path/to/script.json
```

## Outputs

The output directory contains:

- `cases.jsonl`
- `predictions.jsonl`
- `results.jsonl`
- `summary.json`
- `summary.csv`

`summary.json` includes overall totals and grouped metrics by probe, mode, length, position, probe-mode, probe-mode-length, and probe-mode-length-position. Ratio metrics include numerator, denominator, and rate.

`activation_check.py` validates that the four modes actually changed context lifecycle behavior:

```bash
python evals/context_lifecycle/activation_check.py \
  --predictions .runs/context_evals/position_main/predictions.jsonl \
  --output .runs/context_evals/position_main/activation_report.json
```

Before a real run, use:

```bash
bash scripts/context_preflight.sh
```

## Metrics

Position and multi-constraint probes use exact normalized string-set matching. There is no fuzzy semantic matching and no free-text JSON parsing. State and aggregation probes score individual fields so stale or partially correct state can be diagnosed.

Provider and answer-protocol errors are counted as failures. If the model does not call the required answer tool, the executor adds protocol correction turns up to the configured retry limit; it never guesses a structured answer from natural language.
