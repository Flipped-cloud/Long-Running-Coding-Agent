from app import classify_number


def test_negative():
    assert classify_number(-1) == "negative"
