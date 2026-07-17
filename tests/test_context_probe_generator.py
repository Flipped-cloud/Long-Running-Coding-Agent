from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.stale_tracker import StaleTracker
from longrun_agent.context_probes.executor import _replay_event
from longrun_agent.context_probes.generator import _estimate_events, generate_cases


def test_all_probe_types_generate_token_controlled_cases():
    for probe in ["position", "multi_constraint", "state_tracking", "aggregation"]:
        cases = generate_cases(probe=probe, lengths=[2048], samples=3, seed=3)
        assert cases
        for case in cases:
            assert not hasattr(case, "mode")
            error = abs(case.estimated_length_tokens - case.target_length_tokens) / case.target_length_tokens
            assert error <= 0.08
            filler_sizes = [_estimate_events([event]) for event in case.events if event.token_role == "filler"]
            assert filler_sizes
            assert max(filler_sizes) <= 256


def test_position_cases_cover_beginning_middle_end():
    cases = generate_cases(probe="position", lengths=[2048], samples=3, seed=4)

    assert {case.position for case in cases} == {"beginning", "middle", "end"}
    assert all(case.metadata["position_error"] < 0.08 for case in cases)
    assert all("needle_position" in case.metadata for case in cases)


def test_multi_constraint_counts_and_scattered_events():
    cases = generate_cases(probe="multi_constraint", lengths=[2048], samples=3, seed=5, constraints_per_case=[3, 4, 5])

    assert [len(case.expected["constraints"]) for case in cases] == [3, 4, 5]
    for case in cases:
        relevant_steps = [event.step for event in case.events if event.is_relevant]
        assert len(relevant_steps) == len(case.expected["constraints"])
        assert all(right - left > 1 for left, right in zip(relevant_steps, relevant_steps[1:], strict=False))


def test_state_tracking_metadata_marks_stale_turns():
    case = generate_cases(probe="state_tracking", lengths=[2048], samples=1, seed=6)[0]
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    for event in case.events:
        _replay_event(buffer, event)

    StaleTracker().refresh(buffer)

    assert any(turn.status == "stale" for turn in buffer.all_turns())
    assert case.expected["file_hash"] != "hash-D"
    assert case.expected["active_task"].startswith("task_")


def test_aggregation_contains_write_test_code_epoch():
    case = generate_cases(probe="aggregation", lengths=[2048], samples=1, seed=7)[0]

    assert any(event.tool_name == "write_file" for event in case.events)
    assert any(event.tool_name == "bash" and event.metadata.get("code_epoch") == 2 for event in case.events)
