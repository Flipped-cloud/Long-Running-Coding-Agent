from __future__ import annotations


def deterministic_backoff(attempt: int, *, base_seconds: int = 2, maximum_seconds: int = 60) -> int:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if isinstance(base_seconds, bool) or not isinstance(base_seconds, int) or base_seconds < 1:
        raise ValueError("base_seconds must be a positive integer")
    if isinstance(maximum_seconds, bool) or not isinstance(maximum_seconds, int) or maximum_seconds < 1:
        raise ValueError("maximum_seconds must be a positive integer")
    return min(base_seconds * 2 ** (attempt - 1), maximum_seconds)


def can_retry(attempts: int, max_attempts: int) -> bool:
    return attempts < max_attempts
