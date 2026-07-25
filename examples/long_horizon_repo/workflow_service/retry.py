from __future__ import annotations


def deterministic_backoff(attempt: int, *, base_seconds: int = 2, maximum_seconds: int = 60) -> int:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    return min(base_seconds * attempt, maximum_seconds)


def can_retry(attempts: int, max_attempts: int) -> bool:
    return attempts < max_attempts
