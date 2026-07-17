from longrun_agent.context_probes.generator import generate_cases


def test_base_cases_are_mode_independent_for_paired_comparison():
    cases = generate_cases(probe="position", lengths=[2048], samples=2, seed=15)
    snapshots = [case.model_dump(mode="json") for case in cases]

    for mode in ["full_history", "recent_window", "deterministic_prune", "structured_reset"]:
        paired = [case.model_dump(mode="json") for case in cases]
        assert paired == snapshots
        assert all("mode" not in row for row in paired)
        assert mode in {"full_history", "recent_window", "deterministic_prune", "structured_reset"}
