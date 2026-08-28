import os
from pathlib import Path

BROKEN_IMPLEMENTATION = '''from __future__ import annotations


class InsufficientStock(ValueError):
    pass


class Inventory:
    def __init__(self, stock: dict[str, int]):
        self._stock = dict(stock)

    def reserve(self, items: list[tuple[str, int]]) -> dict[str, int]:
        """Reserve requested items and return the remaining stock."""

        for sku, quantity in items:
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            available = self._stock.get(sku, 0)
            if available < quantity:
                raise InsufficientStock(f"insufficient stock for {sku}: requested={quantity}, available={available}")
            self._stock[sku] = available - quantity
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        return dict(self._stock)
'''


def main() -> None:
    target = Path(__file__).with_name("warehouse.py")
    temporary = target.with_suffix(".py.tmp")
    temporary.write_text(BROKEN_IMPLEMENTATION, encoding="utf-8")
    os.replace(temporary, target)
    print(f"reset {target}")


if __name__ == "__main__":
    main()
