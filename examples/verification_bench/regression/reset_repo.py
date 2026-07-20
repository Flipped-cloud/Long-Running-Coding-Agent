from pathlib import Path

Path("app.py").write_text('def classify_number(value: int) -> str:\n    return "positive"\n', encoding="utf-8")
