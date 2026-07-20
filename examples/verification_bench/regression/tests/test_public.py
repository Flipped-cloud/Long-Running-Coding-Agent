from app import classify_number


def test_positive_regression():
    assert classify_number(2) == "positive"
