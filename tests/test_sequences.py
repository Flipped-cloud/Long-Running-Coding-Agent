from longrun_agent.sequences import dedupe_preserving_order


def test_dedupe_preserving_order_drops_empty_and_duplicate_values() -> None:
    assert dedupe_preserving_order(["b", "", "a", "b", "a", "c"]) == ["b", "a", "c"]
