from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    (ROOT / "task_app.py").write_text(
        "def validate_task_name(name: str) -> bool:\n    return bool(name)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
