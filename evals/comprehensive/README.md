# Comprehensive real-model qualification case

This profile extends the bounded long-horizon demonstration with one focused
Agent-generated F2P transition and adaptive recovery search. It reuses the
already validated workflow fixture so the run measures Agent behavior rather
than uncertainty in a second fixture.

The run demonstrates the native model/tool loop, a persisted dependency DAG,
multiple Sessions, structured handoffs, context lifecycle management,
evidence-grounded knowledge, generated-test registration and F2P validation,
adaptive decomposition/recovery when needed, frozen public/hidden contract
checks, integrity enforcement, resume, an independent Oracle, and a final
machine-readable summary.

Expected runtime is approximately 30 minutes for a typical tool-calling model.
The 55-minute project budget and default 70-minute shell watchdog are safety
limits, not duration targets.

Run on Ubuntu/WSL2 from the repository root:

```bash
PROJECT_ID=comprehensive-agent-01 \
CASE_PROFILE=comprehensive \
bash scripts/run_long_horizon_real_api.sh
```

Resume the same persisted project after an interruption:

```bash
PROJECT_ID=comprehensive-agent-01 \
CASE_PROFILE=comprehensive \
RESUME_PROJECT=1 \
bash scripts/run_long_horizon_real_api.sh
```
