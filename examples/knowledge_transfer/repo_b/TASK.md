Fix shell command validation.

Acceptance criteria:

- `python -m pytest -q` passes.
- Commands containing only whitespace are rejected.
- Surrounding whitespace is normalized before command validation.
- Keep the implementation minimal and verify with pytest before completion.
