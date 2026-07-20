from pathlib import Path

Path("app.py").write_text("def value():\n    return 0\n", encoding="utf-8")
Path("tests/test_trusted.py").write_text("from app import value\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
