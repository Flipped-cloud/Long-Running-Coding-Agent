from app import value


def test_type_regression():
    assert isinstance(value(), int)
