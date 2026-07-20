from app import classify_number


def test_negative_odd():
    assert classify_number(-1) == "negative"


def test_negative_even():
    assert classify_number(-2) == "negative"
