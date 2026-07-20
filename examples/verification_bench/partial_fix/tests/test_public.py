from app import classify_number


def test_nonnegative_regression():
    assert classify_number(1) == "nonnegative"
