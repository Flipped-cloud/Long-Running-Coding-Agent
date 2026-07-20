from app import classify_number


def test_positive_and_zero_regression():
    assert classify_number(2) == "positive"
    assert classify_number(0) == "zero"
