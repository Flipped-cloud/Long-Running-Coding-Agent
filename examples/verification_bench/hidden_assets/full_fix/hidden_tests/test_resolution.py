from app import classify_number


def test_negative_value():
    assert classify_number(-2) == "negative"
