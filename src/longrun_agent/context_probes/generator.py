from __future__ import annotations

import json
import random
from typing import Any

from longrun_agent.config import ContextTokenCounterConfig
from longrun_agent.context.token_counter import ApproximateTokenCounter
from longrun_agent.context_probes.schemas import ProbeCase, ProbeEvent, ProbeName, ProbePosition

POSITIONS: list[ProbePosition] = ["beginning", "middle", "end"]
POSITION_RATIOS: dict[ProbePosition, float] = {"beginning": 0.10, "middle": 0.50, "end": 0.90}
DEFAULT_POSITION_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]
FILLER_MIN_TOKENS = 128
FILLER_MAX_TOKENS = 256


def generate_cases(
    *,
    probe: ProbeName,
    lengths: list[int],
    samples: int,
    seed: int = 42,
    positions: list[ProbePosition] | None = None,
    position_ratios: list[float] | None = None,
    constraints_per_case: list[int] | None = None,
    token_length_tolerance: float = 0.08,
) -> list[ProbeCase]:
    rng = random.Random(seed)
    selected_positions = positions or POSITIONS
    selected_position_ratios = position_ratios or [POSITION_RATIOS[position] for position in selected_positions]
    selected_counts = constraints_per_case or [2, 3, 5]
    cases: list[ProbeCase] = []
    for length in lengths:
        for index in range(samples):
            if probe == "position":
                ratio = selected_position_ratios[index % len(selected_position_ratios)]
                position = _position_from_ratio(ratio)
                cases.append(_position_case(length, seed, index, position, ratio, rng, token_length_tolerance))
            elif probe == "multi_constraint":
                count = selected_counts[index % len(selected_counts)]
                cases.append(_multi_case(length, seed, index, count, rng, token_length_tolerance))
            elif probe == "state_tracking":
                cases.append(_state_case(length, seed, index, token_length_tolerance))
            elif probe == "aggregation":
                cases.append(_aggregation_case(length, seed, index, token_length_tolerance))
            else:
                raise ValueError(f"unknown probe: {probe}")
    return cases


def _position_case(
    length: int, seed: int, index: int, position: ProbePosition, ratio: float, rng: random.Random, tolerance: float
) -> ProbeCase:
    token = _unique_token(rng)
    constraint = f"Do not modify API symbol public_process_{token}."
    case_id = _case_id("position", length, index, seed)
    events: list[ProbeEvent] = []
    _add_filler_events(events, target_tokens=max(1, int(length * ratio)), label=f"{case_id}-before", completion_ratio=0.99)
    events.append(
        _event(
            len(events) + 1,
            "read_file",
            {"path": "src/service.py", "start_line": 1, "end_line": 80},
            {
                "summary": "active coding constraint observed",
                "output": f"Developer note: {constraint}",
                "metadata": {
                    "path": "src/service.py",
                    "content_sha256": f"constraint-{token}",
                    "start_line": 1,
                    "end_line": 80,
                    "constraint": constraint,
                },
            },
            token_role="constraint",
            is_relevant=True,
        )
    )
    current = _estimate_events(events)
    if current < length:
        _add_filler_events(events, target_tokens=length - current, label=f"{case_id}-after")
    return _finalize_case(
        case_id=case_id,
        probe="position",
        length=length,
        seed=seed,
        events=events,
        final_query="Return every currently active coding constraint from the observed interaction history.",
        expected={"constraints": [constraint]},
        position=position,
        metadata={"constraint": constraint, "target_position_ratio": ratio, "needle_position": ratio},
        tolerance=tolerance,
    )


def _multi_case(length: int, seed: int, index: int, count: int, rng: random.Random, tolerance: float) -> ProbeCase:
    case_id = _case_id("multi_constraint", length, index, seed)
    categories = ["file path", "test requirement", "API name", "version", "configuration"]
    constraints = [_constraint_for_category(categories[item % len(categories)], rng) for item in range(max(3, count))]
    events: list[ProbeEvent] = []
    _add_filler_events(events, target_tokens=150, label=f"{case_id}-pre")
    for constraint_index, constraint in enumerate(constraints):
        distractor = f"Previous note, now superseded: modifying public_process_{_unique_token(rng)} was once considered."
        events.append(
            _event(
                len(events) + 1,
                "read_file",
                {"path": f"src/module_{constraint_index}.py", "start_line": 1, "end_line": 60},
                {
                    "summary": "constraint and superseded note observed",
                    "output": f"{distractor}\nCurrent active constraint: {constraint}",
                    "metadata": {
                        "path": f"src/module_{constraint_index}.py",
                        "content_sha256": f"multi-{constraint_index}-{seed}-{index}",
                        "start_line": 1,
                        "end_line": 60,
                        "constraint": constraint,
                        "distractor": distractor,
                    },
                },
                token_role="constraint",
                is_relevant=True,
            )
        )
        if constraint_index != len(constraints) - 1:
            _add_filler_events(events, target_tokens=150, label=f"{case_id}-gap-{constraint_index}")
    if _estimate_events(events) < length:
        _add_filler_events(events, target_tokens=length - _estimate_events(events), label=f"{case_id}-post")
    return _finalize_case(
        case_id=case_id,
        probe="multi_constraint",
        length=length,
        seed=seed,
        events=events,
        final_query="Return every currently active coding constraint from the observed interaction history.",
        expected={"constraints": constraints},
        metadata={"constraint_count": len(constraints), "constraint_categories": categories[: len(constraints)]},
        tolerance=tolerance,
    )


def _state_case(length: int, seed: int, index: int, tolerance: float) -> ProbeCase:
    rng = random.Random(f"state-{seed}-{length}-{index}")
    case_id = _case_id("state_tracking", length, index, seed)
    file_path = f"src/{_slug(rng, 'module')}.py"
    task_initial = f"task_{rng.randrange(10, 99)}"
    task_final = f"task_{rng.randrange(100, 999)}"
    branch = f"feature/{_slug(rng, 'branch')}"
    test_name = f"test_{_slug(rng, 'flow')}"
    hash_a = _commit_hash(rng)
    hash_b = _commit_hash(rng)
    hash_c = _commit_hash(rng)
    hash_d = _commit_hash(rng)
    events: list[ProbeEvent] = []
    _add_filler_events(events, target_tokens=max(1, length // 5), label=f"{case_id}-pre")
    events.extend(
        [
            _event(
                len(events) + 1,
                "read_file",
                {"path": file_path, "start_line": 1, "end_line": 40},
                {
                    "summary": f"read {file_path} at {hash_a}",
                    "output": f"{file_path} content from commit {hash_a} on branch {branch}",
                    "metadata": {"path": file_path, "content_sha256": hash_a, "start_line": 1, "end_line": 40, "branch": branch},
                },
                token_role="state",
                is_relevant=True,
            ),
            _event(
                len(events) + 2,
                "write_file",
                {"path": file_path, "content": f"{hash_b} content"},
                {
                    "summary": f"write {hash_b}",
                    "output": f"{file_path} updated from {hash_a} to {hash_b}",
                    "metadata": {"path": file_path, "status": "updated", "before_sha256": hash_a, "after_sha256": hash_b},
                },
                token_role="state",
                is_relevant=True,
            ),
            _bash_event(len(events) + 3, exit_code=1, code_epoch=1, output=f"pytest failed: {test_name}", summary="verification failed"),
            _event(
                len(events) + 4,
                "write_file",
                {"path": file_path, "content": f"{hash_c} content"},
                {
                    "summary": f"write {hash_c}",
                    "output": f"{file_path} updated from {hash_b} to {hash_c}",
                    "metadata": {"path": file_path, "status": "updated", "before_sha256": hash_b, "after_sha256": hash_c},
                },
                token_role="state",
                is_relevant=True,
            ),
            _bash_event(len(events) + 5, exit_code=0, code_epoch=2, output=f"pytest passed: {test_name}", summary="verification passed"),
            _event(
                len(events) + 6,
                "write_file",
                {"path": file_path, "content": f"{hash_d} content"},
                {
                    "summary": f"write {hash_d} after passing test",
                    "output": f"{file_path} updated to {hash_d}; previous passing test for {test_name} is stale",
                    "metadata": {"path": file_path, "status": "updated", "before_sha256": hash_c, "after_sha256": hash_d},
                },
                token_role="state",
                is_relevant=True,
            ),
        ]
    )
    events.append(
        _progress_event(
            len(events) + 1,
            f"plan_version=2; active_task={task_final}; previous_task={task_initial}; branch={branch}",
            {"plan_version": 2, "active_task": task_final, "previous_task": task_initial, "branch": branch},
        )
    )
    if _estimate_events(events) < length:
        _add_filler_events(events, target_tokens=length - _estimate_events(events), label=f"{case_id}-post")
    return _finalize_case(
        case_id=case_id,
        probe="state_tracking",
        length=length,
        seed=seed,
        events=events,
        final_query=(
            "Return the current plan version, current file hash, current active task, current verification state, "
            "and latest non-stale test result."
        ),
        expected={
            "plan_version": 2,
            "file_hash": hash_d,
            "active_task": task_final,
            "verification_state": "stale_after_write",
            "latest_valid_test": None,
            "stale_test_label": f"pytest passed: {test_name}",
        },
        metadata={
            "state_chain": "read, write, fail, write, pass, write, plan update",
            "file_path": file_path,
            "branch": branch,
            "initial_task": task_initial,
            "final_task": task_final,
            "test_name": test_name,
        },
        tolerance=tolerance,
    )


def _aggregation_case(length: int, seed: int, index: int, tolerance: float) -> ProbeCase:
    rng = random.Random(f"aggregation-{seed}-{length}-{index}")
    case_id = _case_id("aggregation", length, index, seed)
    changed_files = [f"src/{_slug(rng, 'component')}.py", f"src/{_slug(rng, 'adapter')}.py"]
    stale_failures = [f"test_{_slug(rng, 'old')}", f"test_{_slug(rng, 'legacy')}"]
    current_failure = f"test_{_slug(rng, 'current')}"
    first_passed = rng.randrange(7, 15)
    first_failed = len(stale_failures)
    second_passed = first_passed + rng.randrange(1, 5)
    final_passed = second_passed + rng.randrange(0, 4)
    warning_count = rng.randrange(1, 7)
    error_count = rng.randrange(1, 4)
    events: list[ProbeEvent] = []
    events.append(
        _bash_event(
            1,
            exit_code=1,
            code_epoch=0,
            output=f"pytest: {first_passed} passed, {first_failed} failed: {' '.join(stale_failures)}",
            summary="pytest failed",
        )
    )
    _add_filler_events(events, target_tokens=160, label=f"{case_id}-gap-1")
    events.append(
        _event(
            len(events) + 1,
            "write_file",
            {"path": changed_files[0], "content": "epoch 1"},
            {
                "summary": f"modified {changed_files[0]}",
                "output": f"{changed_files[0]} modified; warnings={warning_count}",
                "metadata": {"path": changed_files[0], "status": "updated", "before_sha256": "epoch0-a", "after_sha256": "epoch1-a"},
            },
            token_role="aggregation",
            is_relevant=True,
        )
    )
    events.append(
        _bash_event(len(events) + 1, exit_code=0, code_epoch=1, output=f"pytest: {second_passed} passed", summary="pytest passed")
    )
    _add_filler_events(events, target_tokens=160, label=f"{case_id}-gap-2")
    events.append(
        _progress_event(
            len(events) + 1,
            f"progress: {changed_files[0]} fixed; continuing {changed_files[1]}; warnings={warning_count}",
            {"code_epoch": 1, "warning_count": warning_count},
        )
    )
    events.append(
        _event(
            len(events) + 1,
            "write_file",
            {"path": changed_files[1], "content": "epoch 2"},
            {
                "summary": f"modified {changed_files[1]}",
                "output": f"{changed_files[1]} modified; errors={error_count}",
                "metadata": {"path": changed_files[1], "status": "updated", "before_sha256": "epoch1-b", "after_sha256": "epoch2-b"},
            },
            token_role="aggregation",
            is_relevant=True,
        )
    )
    events.append(
        _bash_event(
            len(events) + 1,
            exit_code=1,
            code_epoch=2,
            output=f"pytest: {final_passed} passed, 1 failed: {current_failure}; errors={error_count}; warnings={warning_count}",
            summary="latest pytest failed",
        )
    )
    if _estimate_events(events) < length:
        _add_filler_events(events, target_tokens=length - _estimate_events(events), label=f"{case_id}-post")
    return _finalize_case(
        case_id=case_id,
        probe="aggregation",
        length=length,
        seed=seed,
        events=events,
        final_query=(
            "Aggregate the current effective test result, unresolved failures, stale test rejections, recently modified files, "
            "and current code epoch."
        ),
        expected={
            "current_test": f"pytest: {final_passed} passed, 1 failed: {current_failure}; errors={error_count}; warnings={warning_count}",
            "unresolved_failures": [current_failure],
            "stale_results_rejected": True,
            "modified_files": changed_files,
            "code_epoch": 2,
        },
        metadata={
            "current_failure": current_failure,
            "stale_failures": stale_failures,
            "warning_count": warning_count,
            "error_count": error_count,
        },
        tolerance=tolerance,
    )


def _finalize_case(
    *,
    case_id: str,
    probe: ProbeName,
    length: int,
    seed: int,
    events: list[ProbeEvent],
    final_query: str,
    expected: dict[str, Any],
    tolerance: float,
    position: ProbePosition | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProbeCase:
    _renumber(events)
    estimated = _estimate_case(events, final_query)
    if estimated < length and abs(estimated - length) / length > tolerance:
        _add_filler_events(events, target_tokens=length - estimated, label=f"{case_id}-adjust")
        _renumber(events)
        estimated = _estimate_case(events, final_query)
    meta = dict(metadata or {})
    if position is not None:
        relevant_index = next((index for index, event in enumerate(events) if event.is_relevant), 0)
        before_tokens = _estimate_events(events[:relevant_index])
        meta["actual_position_ratio"] = before_tokens / max(1, estimated)
        meta["position_error"] = abs(meta["actual_position_ratio"] - float(meta.get("needle_position", POSITION_RATIOS[position])))
    return ProbeCase(
        case_id=case_id,
        probe=probe,
        target_length_tokens=length,
        estimated_length_tokens=estimated,
        position=position,
        events=events,
        final_query=final_query,
        expected=expected,
        seed=seed,
        metadata=meta,
    )


def _event(
    step: int,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    *,
    token_role: str,
    is_relevant: bool = False,
) -> ProbeEvent:
    payload = {
        "tool_name": tool_name,
        "success": result.get("success", True),
        "summary": result.get("summary", ""),
        "output": result.get("output", ""),
        "metadata": result.get("metadata", {}),
    }
    return ProbeEvent(
        event_id=f"probe-{step:04d}",
        step=step,
        tool_name=tool_name,
        tool_arguments=arguments,
        tool_result=payload,
        token_role=token_role,
        is_relevant=is_relevant,
        metadata=dict(payload["metadata"]),
    )


def _bash_event(step: int, *, exit_code: int, code_epoch: int, output: str, summary: str) -> ProbeEvent:
    return _event(
        step,
        "bash",
        {"argv": ["python", "-m", "pytest", "-q"]},
        {
            "success": exit_code == 0,
            "summary": summary,
            "output": output,
            "metadata": {
                "normalized_command": "python -m pytest -q",
                "code_epoch": code_epoch,
                "exit_code": exit_code,
                "verification_kind": "pytest",
            },
        },
        token_role="verification",
        is_relevant=True,
    )


def _progress_event(step: int, output: str, metadata: dict[str, Any]) -> ProbeEvent:
    return _event(
        step,
        "report_progress",
        {"summary": output},
        {"success": True, "summary": "progress reported", "output": output, "metadata": metadata},
        token_role="state",
        is_relevant=True,
    )


def _add_filler_events(events: list[ProbeEvent], *, target_tokens: int, label: str, completion_ratio: float = 0.92) -> None:
    if target_tokens <= 0:
        return
    start = len(events)
    while _estimate_events(events[start:]) < target_tokens * completion_ratio:
        index = len(events) + 1
        remaining = target_tokens - _estimate_events(events[start:])
        chunk_target = max(FILLER_MIN_TOKENS, min(FILLER_MAX_TOKENS, remaining))
        repeated = _filler_word_count(index, label, chunk_target)
        if repeated <= 0:
            repeated = 1
        text = " ".join([f"{label} neutral log {index}-{part}; no durable fact." for part in range(repeated)])
        events.append(
            _event(
                index,
                "read_file" if index % 2 else "bash",
                {"path": f"logs/filler_{index}.txt"} if index % 2 else {"argv": ["python", "-c", "print('noop')"]},
                {
                    "success": True,
                    "summary": "neutral filler observation",
                    "output": text,
                    "metadata": {
                        "path": f"logs/filler_{index}.txt",
                        "content_sha256": f"filler-{label}-{index}",
                        "start_line": 1,
                        "end_line": repeated,
                    }
                    if index % 2
                    else {
                        "normalized_command": "python -c print-noop",
                        "code_epoch": 0,
                        "exit_code": 0,
                    },
                },
                token_role="filler",
            )
        )


def _estimate_case(events: list[ProbeEvent], final_query: str) -> int:
    return _estimate_events(events) + _counter().count_text(final_query)


def _estimate_events(events: list[ProbeEvent]) -> int:
    if not events:
        return 0
    messages: list[dict[str, Any]] = []
    for event in events:
        messages.extend(_event_messages(event))
    return _counter().count_messages(messages)


def _event_messages(event: ProbeEvent) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": event.event_id,
                    "type": "function",
                    "function": {
                        "name": event.tool_name,
                        "arguments": json.dumps(event.tool_arguments, ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": event.event_id,
            "name": event.tool_name,
            "content": json.dumps(event.tool_result, ensure_ascii=False, sort_keys=True),
        },
    ]


def _filler_word_count(index: int, label: str, remaining_tokens: int) -> int:
    low = 0
    high = max(1, remaining_tokens * 8)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = _event(
            index,
            "read_file" if index % 2 else "bash",
            {"path": f"logs/filler_{index}.txt"} if index % 2 else {"argv": ["python", "-c", "print('noop')"]},
            {
                "success": True,
                "summary": "neutral filler observation",
                "output": " ".join([f"{label} neutral log {index}-{part}; no durable fact." for part in range(mid)]),
                "metadata": {
                    "path": f"logs/filler_{index}.txt",
                    "content_sha256": f"filler-{label}-{index}",
                    "start_line": 1,
                    "end_line": mid,
                }
                if index % 2
                else {"normalized_command": "python -c print-noop", "code_epoch": 0, "exit_code": 0},
            },
            token_role="filler",
        )
        tokens = _estimate_events([candidate])
        if tokens <= remaining_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _counter() -> ApproximateTokenCounter:
    return ApproximateTokenCounter(ContextTokenCounterConfig(type="approximate"))


def _renumber(events: list[ProbeEvent]) -> None:
    for step, event in enumerate(events, 1):
        event.step = step
        event.event_id = f"probe-{step:04d}"


def _unique_token(rng: random.Random) -> str:
    return f"{rng.randrange(16**4):04x}"


def _case_id(probe: ProbeName, length: int, index: int, seed: int) -> str:
    return f"{probe}_{length}_{index:03d}_seed{seed}"


def _position_from_ratio(ratio: float) -> ProbePosition:
    if ratio < 0.34:
        return "beginning"
    if ratio < 0.67:
        return "middle"
    return "end"


def _constraint_for_category(category: str, rng: random.Random) -> str:
    token = _unique_token(rng)
    if category == "file path":
        return f"Do not modify file src/{_slug(rng, 'stable')}_{token}.py."
    if category == "test requirement":
        return f"Keep test_{_slug(rng, 'contract')}_{token} passing before completion."
    if category == "API name":
        return f"Do not modify API symbol public_process_{token}."
    if category == "version":
        return f"Preserve schema version v{rng.randrange(1, 5)}.{rng.randrange(0, 9)}-{token}."
    return f"Keep configuration key feature.{_slug(rng, 'flag')}.{token} unchanged."


def _slug(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(1000, 9999)}"


def _commit_hash(rng: random.Random) -> str:
    return f"{rng.randrange(16**8):08x}"
