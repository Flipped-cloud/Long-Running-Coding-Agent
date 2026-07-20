from pathlib import Path

Path("app.py").write_text("def value():\n    return 0\n", encoding="utf-8")
