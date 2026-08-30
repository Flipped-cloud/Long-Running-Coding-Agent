# Comprehensive Agent Qualification Task

Complete the advanced service and CLI layers of the file-backed workflow
scheduler. The lower-level model, serialization, storage, graph, retry, audit,
and metrics modules are a tested foundation. Preserve them unless focused test
evidence proves that a narrow integration correction is necessary.

## Working rules

- Modify implementation files under `workflow_service/`, generated tests under
  `agent_tests/`, and `README.md` only.
- Do not modify public tests, this task file, or project configuration.
- Do not skip, rename, delete, or weaken tests.
- Preserve existing public imports and CLI behavior.
- In the execution task, add one focused issue-reproduction test under
  `agent_tests/`, run it, register it with `register_test_candidate`, and inspect
  the returned transition. Later tasks may reuse normal focused tests.
- A valid generated test supplements but never replaces the frozen verification
  contract, public regression tests, or hidden verification.
- Run focused tests before completing each task and `python -m pytest -q` before
  completing final integration.
- Do not create scratch files inside the workspace; use `/tmp` for disposable
  diagnostics.

## Required behavior

1. `run` requires all dependencies to have succeeded and records legal,
   persisted running-to-terminal transitions.
2. Retry enforces attempt limits, preserves deterministic policy behavior, and
   emits a complete retry audit event.
3. Failure and cancellation propagate through every transitive dependent while
   preserving the originating reason.
4. Create and cancel idempotency survives service reconstruction; identical
   replays do not duplicate state or audit events, while conflicting key reuse
   is rejected.
5. Every state-changing operation emits exactly one monotonically sequenced,
   append-only audit event.
6. Metrics report status counts, attempts, retries, terminal count, and terminal
   rate with deterministic empty values.
7. CLI `run`, `retry`, and `cancel` commands call the service and return canonical
   JSON on success or a clear non-zero stderr failure.
8. CLI import validates the full graph before replacing persisted data; export
   is canonical, creates parent directories, and round-trips without data loss.
9. CLI `metrics` exposes the complete deterministic metrics payload.
10. README documents architecture, persistence, lifecycle, propagation,
    idempotency, retry, audit, metrics, import/export, and executable examples.
11. All public tests and frozen hidden checks pass without integrity violations.
