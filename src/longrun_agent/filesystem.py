from __future__ import annotations

import os
import time
from pathlib import Path


def atomic_replace(source: str | Path, target: str | Path, *, attempts: int = 5) -> None:
    """Atomically replace a file, retrying transient Windows sharing violations."""

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.01 * (attempt + 1))
