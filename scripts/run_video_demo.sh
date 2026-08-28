#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:-${PYTHON:-python3}}"
script_dir="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
demo_root="$repo_root/examples/video_demo_repo"
config="$repo_root/configs/video_demo_real.yaml"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 1
fi

for name in OPENAI_API_KEY OPENAI_BASE_URL MODEL_NAME; do
    if [[ -z "${!name:-}" ]]; then
        printf 'Missing required environment variable: %s\n' "$name" >&2
        exit 1
    fi
done

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

trusted_manifest() {
    "$python_bin" - "$demo_root" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
ignored_parts = {".pytest_cache", "__pycache__"}
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root)
    if relative.as_posix() == "warehouse.py" or path.suffix == ".pyc" or ignored_parts.intersection(relative.parts):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{relative.as_posix()} {digest}")
PY
}

printf '[1/4] Resetting the demo repository\n'
"$python_bin" "$demo_root/reset_repo.py"
trusted_before="$(trusted_manifest)"

printf '[2/4] Confirming the baseline test suite fails\n'
set +e
(
    cd -- "$demo_root"
    "$python_bin" -m pytest -q
)
baseline_status=$?
set -e
if [[ $baseline_status -eq 0 ]]; then
    printf 'The intentionally broken baseline unexpectedly passed\n' >&2
    exit 1
fi
if [[ $baseline_status -ne 1 ]]; then
    printf 'Baseline pytest exited unexpectedly with status %s\n' "$baseline_status" >&2
    exit "$baseline_status"
fi

printf '[3/4] Running longrun-agent with the real model\n'
task="$(<"$demo_root/TASK.md")"
"$python_bin" -m longrun_agent.cli run \
    --config "$config" \
    --workspace "$demo_root" \
    --task "$task"

trusted_after="$(trusted_manifest)"
if [[ "$trusted_after" != "$trusted_before" ]]; then
    printf 'The agent modified files outside warehouse.py; refusing final verification\n' >&2
    exit 1
fi
printf 'Trusted tests and configuration are unchanged\n'

printf '[4/4] Independently verifying the finished repository\n'
(
    cd -- "$demo_root"
    "$python_bin" -m pytest -q
)

printf 'Video demo completed successfully.\n'
