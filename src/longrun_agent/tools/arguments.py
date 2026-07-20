from __future__ import annotations

import math
import shlex
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


class ToolArgumentError(ValueError):
    """Raised when tool arguments cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class ArgumentNormalization:
    field: str
    index: int
    original_type: str
    normalized_type: str
    reason: str

    def model_dump(self) -> dict[str, str | int]:
        return asdict(self)


_ALLOWED_TYPES = "string, integer, finite number, or boolean"


def normalize_command_argv(
    values: Sequence[Any],
    *,
    field: str = "argv",
) -> tuple[list[str], list[ArgumentNormalization]]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ToolArgumentError(f"{field} must be an array of {_ALLOWED_TYPES}; received {_type_name(values)}")
    if not values:
        raise ToolArgumentError(f"{field} must not be empty; allowed item types: {_ALLOWED_TYPES}")

    normalized: list[str] = []
    records: list[ArgumentNormalization] = []
    for index, value in enumerate(values):
        value_type = type(value)
        if value_type is str:
            normalized.append(value)
            continue
        if value_type is bool:
            normalized.append("true" if value else "false")
        elif value_type is int:
            normalized.append(str(value))
        elif value_type is float and math.isfinite(value):
            normalized.append(str(value))
        elif value_type is float:
            raise ToolArgumentError(
                f"{field}[{index}] must be a string or finite JSON scalar; received non-finite number; allowed types: {_ALLOWED_TYPES}"
            )
        else:
            raise ToolArgumentError(
                f"{field}[{index}] must be a string or JSON scalar; received {_type_name(value)}; allowed types: {_ALLOWED_TYPES}"
            )
        records.append(
            ArgumentNormalization(
                field=field,
                index=index,
                original_type=value_type.__name__,
                normalized_type="str",
                reason="json_scalar_to_string",
            )
        )

    if not normalized[0].strip():
        raise ToolArgumentError(f"{field}[0] must be a non-empty command string; allowed types: string")
    return normalized, records


def render_command(argv: Sequence[str]) -> str:
    if isinstance(argv, (str, bytes, bytearray)) or not isinstance(argv, Sequence) or not argv:
        raise ToolArgumentError("argv must be a non-empty normalized list of strings")
    for index, value in enumerate(argv):
        if type(value) is not str:
            raise ToolArgumentError(f"argv[{index}] must be a normalized string; received {_type_name(value)}; allowed types: string")
    return shlex.join(list(argv))


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple, set)):
        return "array"
    if isinstance(value, bytes):
        return "bytes"
    return type(value).__name__
