Fix task-name validation.

Acceptance criteria:

- Make the minimal change so whitespace-only task names are rejected.
- Preserve valid non-empty task names.
- Run `python -m pytest -q`.
- Request task completion only after pytest passes.
