Investigate task-name validation without applying a fix.

Acceptance criteria:

- Inspect the current implementation.
- Run the focused pytest suite and capture the failing whitespace-only input evidence.
- Diagnose that whitespace-only names are incorrectly accepted.
- Do not modify source files in this learning probe.
- Report the issue as a blocker with the observed verification evidence.
