# Long-Horizon Workflow Service Task

Complete the Python workflow and task scheduling service in this workspace.
The repository is intentionally incomplete: preserve its existing behavior while
implementing every requirement below.

## Scope and constraints

- Modify only implementation files under `workflow_service/` and `README.md`.
- Do not modify tests, this task specification, or project configuration.
- Do not delete, skip, rename, or weaken tests.
- Keep backward compatibility for existing public imports and CLI commands.
- Before completing each plan task, run its focused tests.
- Before completing the final plan task, run `python -m pytest -q`.
- Do not report completion solely from inspection; provide executed test evidence.

## Required behavior

1. The `Task` model strictly validates identifiers, names, status, counters,
   dependencies, metadata, and JSON-compatible result values.
2. Task JSON serialization is lossless, rejects malformed/unknown data, and
   produces deterministic output.
3. File persistence creates parent directories, uses same-directory atomic
   replacement, leaves no temporary files, and rejects duplicate/corrupt data.
4. Lifecycle transitions are legal and explicit. Terminal tasks cannot silently
   return to mutable states.
5. Dependency graphs reject missing nodes and cycles with useful diagnostics.
6. Ready-task selection is deterministic and only returns tasks whose
   dependencies succeeded.
7. Retry policy uses deterministic capped exponential backoff and enforces
   attempt limits without sleeping.
8. Failures and cancellations propagate through all transitive dependents while
   preserving the originating reason.
9. Create, run, retry, and cancel operations are idempotent where an
   idempotency key or identical terminal request is supplied; conflicting reuse
   is rejected.
10. Every state-changing operation emits an append-only, monotonically
    sequenced audit event.
11. Metrics include status counts, attempts, retry count, terminal count, and
    terminal rate, including well-defined empty values.
12. The CLI supports `create`, `get`, `list`, `run`, `retry`, `cancel`,
    `import`, `export`, and `metrics`, returns JSON on success, and returns
    non-zero with a clear stderr message on invalid operations.
13. Import/export round trips preserve tasks and validate the complete graph
    before replacing persisted data.
14. Public and integration tests pass, and README documents storage, lifecycle,
    retry semantics, command examples, and JSON import/export.

