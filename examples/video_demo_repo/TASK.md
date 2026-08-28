# Make inventory reservations transactional

`Inventory.reserve()` currently mutates stock while it is still validating the request. A later failure can therefore leave a partially applied reservation, and repeated entries for one SKU are not validated as a combined request.

Implement the following behavior without changing the public API:

1. A reservation is all-or-nothing: any validation or stock failure leaves inventory unchanged.
2. Repeated SKU entries are aggregated before stock validation and deduction.
3. Every quantity must be a positive integer; invalid quantities leave inventory unchanged.
4. Preserve the existing exception types and useful stock error details.

Inspect the existing implementation and tests, make the smallest appropriate change in `warehouse.py`, and run the complete test suite before finishing. Do not modify tests, test configuration, or other project files.
