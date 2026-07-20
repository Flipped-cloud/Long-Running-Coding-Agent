from pathlib import Path

Path("app.py").write_text(
    'def classify_number(value: int) -> str:\n    if value == 0:\n        return "zero"\n    return "positive"\n',
    encoding="utf-8",
)
