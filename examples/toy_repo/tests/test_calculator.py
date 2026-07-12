import pytest
from calculator import divide


def test_divide_positive_numbers():
    assert divide(8, 2) == 4


def test_divide_negative_number():
    assert divide(-9, 3) == -3


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)
