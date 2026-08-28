# Transactional inventory demo

This intentionally broken repository is the real-model video demonstration for `longrun-agent`.

The task asks the agent to repair partial inventory mutations and duplicate-SKU handling. Reset it with `python reset_repo.py`, then run the repository tests to show the failing baseline.

## Run on Ubuntu

From the main `Long-Running-Coding-Agent` repository:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export MODEL_NAME="your-tool-calling-model"

bash scripts/run_video_demo.sh
```

If Python is installed elsewhere, pass its command or absolute path as the first argument:

```bash
bash scripts/run_video_demo.sh /path/to/python
```

The script performs four visible stages: restore the broken implementation, prove that the baseline tests fail, run `longrun-agent` with the real model, and independently require all tests to pass. It never reads `.env` automatically.

For a clean retake, run the same script again; the first stage always restores the intentional bug.
