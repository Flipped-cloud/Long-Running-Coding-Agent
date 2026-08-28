from __future__ import annotations


class InsufficientStock(ValueError):
    pass


class Inventory:
    def __init__(self, stock: dict[str, int]):
        self._stock = dict(stock)

    def reserve(self, items: list[tuple[str, int]]) -> dict[str, int]:
        """Reserve requested items and return the remaining stock."""

        totals: dict[str, int] = {}
        for sku, quantity in items:
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
                raise ValueError("quantity must be positive")
            totals[sku] = totals.get(sku, 0) + quantity
        for sku, quantity in totals.items():
            available = self._stock.get(sku, 0)
            if available < quantity:
                raise InsufficientStock(f"insufficient stock for {sku}: requested={quantity}, available={available}")
        for sku, quantity in totals.items():
            self._stock[sku] = self._stock[sku] - quantity
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        return dict(self._stock)
