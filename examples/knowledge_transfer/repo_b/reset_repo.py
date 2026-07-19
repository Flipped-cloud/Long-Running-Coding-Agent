from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    (ROOT / "command_app.py").write_text(
        "def normalize_command(command: str) -> str:\n    return command\n\n\n"
        "def is_valid_command(command: str) -> bool:\n    return bool(command)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
