import pytest
from warehouse import InsufficientStock, Inventory


def test_successful_reservation_updates_all_items() -> None:
    inventory = Inventory({"keyboard": 5, "mouse": 4})

    remaining = inventory.reserve([("keyboard", 2), ("mouse", 1)])

    assert remaining == {"keyboard": 3, "mouse": 3}
    assert inventory.snapshot() == remaining


def test_failed_reservation_is_atomic() -> None:
    inventory = Inventory({"keyboard": 5, "mouse": 1})

    with pytest.raises(InsufficientStock, match="mouse"):
        inventory.reserve([("keyboard", 2), ("mouse", 2)])

    assert inventory.snapshot() == {"keyboard": 5, "mouse": 1}


def test_duplicate_skus_are_aggregated_before_validation() -> None:
    inventory = Inventory({"keyboard": 3})

    with pytest.raises(InsufficientStock, match=r"requested=4, available=3"):
        inventory.reserve([("keyboard", 2), ("keyboard", 2)])

    assert inventory.snapshot() == {"keyboard": 3}


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_invalid_quantity_does_not_mutate_inventory(quantity: object) -> None:
    inventory = Inventory({"keyboard": 5})

    with pytest.raises(ValueError, match="quantity must be positive"):
        inventory.reserve([("keyboard", 1), ("mouse", quantity)])

    assert inventory.snapshot() == {"keyboard": 5}


def test_snapshot_is_not_a_mutable_view() -> None:
    inventory = Inventory({"keyboard": 5})
    snapshot = inventory.snapshot()

    snapshot["keyboard"] = 0

    assert inventory.snapshot() == {"keyboard": 5}
