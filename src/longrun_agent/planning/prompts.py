def initial_planner_prompt(min_tasks: int, max_tasks: int) -> str:
    return f"""Create a coarse-grained engineering plan.

You must call submit_plan exactly once.

Create between {min_tasks} and {max_tasks} complete tasks.
Do not create one task for every minor requirement.
Combine related requirements so the whole project can be completed within this range.
For fast static planning, prefer 2-3 coarse tasks when allowed.

Merge related requirements when practical:
- validation and retry behavior may be combined;
- persistence and CLI work may be combined;
- tests, docs, and integration checks may be combined.

Every task must contain:
- key
- title
- objective
- acceptance_criteria
- depends_on_keys

Each acceptance criterion must be observable and testable.
Dependencies must appear only as strings inside depends_on_keys.
Do not emit dependency-only objects.
Do not claim any task is already complete.
"""


DECOMPOSER_PROMPT = """Decompose the blocked task only when needed.
Use only submit_decomposition. Children must be more specific than the parent and have acceptance criteria."""

RECOVERY_GENERATOR_PROMPT = """Generate bounded recovery candidates for one blocked task.
Use only submit_recovery_candidates. Do not branch the repository or propose full tree search."""

RECOVERY_EVALUATOR_PROMPT = """Select exactly one valid recovery candidate.
Use only select_recovery_candidate and score all candidates."""
