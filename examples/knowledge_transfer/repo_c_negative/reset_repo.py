from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    (ROOT / "format_app.py").write_text("def preserve_command(command: str) -> str:\n    return command\n", encoding="utf-8")


if __name__ == "__main__":
    main()
