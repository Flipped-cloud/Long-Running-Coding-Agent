#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

missing_env=0
for variable_name in MODEL_NAME OPENAI_BASE_URL OPENAI_API_KEY; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "missing required environment variable: $variable_name" >&2
    missing_env=1
  fi
done
if (( missing_env != 0 )); then
  exit 2
fi

PYTHON_BIN="${LONGRUN_PYTHON_BIN:-python}"
AGENT_BIN="${LONGRUN_AGENT_BIN:-longrun-agent}"
PROJECT_ID="${PROJECT_ID:-long-horizon-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MINIMUM_SECONDS="${MINIMUM_SECONDS:-1800}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3900}"
CONFIG="$REPO_ROOT/configs/long_horizon_real_api.yaml"
FIXTURE="$REPO_ROOT/examples/long_horizon_repo"
TASK_FILE="$REPO_ROOT/evals/long_horizon/TASK.md"
HIDDEN_TESTS="$REPO_ROOT/evals/long_horizon/hidden_tests"
RUN_ROOT="$REPO_ROOT/.runs/long_horizon_real_api"
WORKSPACE="$RUN_ROOT/workspaces/$PROJECT_ID"
RESULT_DIR="$RUN_ROOT/results/$PROJECT_ID"
STATE_ROOT="$RUN_ROOT/projects"
TELEMETRY_ROOT="$RUN_ROOT/telemetry"
ORACLE_PATH="$RESULT_DIR/oracle.json"
SUMMARY_PATH="$RESULT_DIR/summary.json"
CONSOLE_LOG="$RESULT_DIR/console.log"

if [[ -e "$WORKSPACE" || -e "$RESULT_DIR" || -e "$STATE_ROOT/$PROJECT_ID" ]]; then
  echo "project output already exists; choose a new PROJECT_ID: $PROJECT_ID" >&2
  exit 2
fi

mkdir -p "$WORKSPACE" "$RESULT_DIR"
cp -R "$FIXTURE/." "$WORKSPACE/"
cp "$TASK_FILE" "$WORKSPACE/TASK.md"

export LONGRUN_WORKSPACE="$WORKSPACE"
export LONGRUN_PLAN_FILE="$REPO_ROOT/evals/long_horizon/plan.json"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"
CLI_EXIT_CODE=1

finalize_run() {
  original_exit_code=$?
  trap - EXIT INT TERM
  set +e

  ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ENDED_EPOCH="$(date +%s)"
  ELAPSED_SECONDS=$((ENDED_EPOCH - STARTED_EPOCH))

  "$PYTHON_BIN" "$REPO_ROOT/scripts/validate_long_horizon_result.py" \
    --workspace "$WORKSPACE" \
    --fixture "$FIXTURE" \
    --task-file "$TASK_FILE" \
    --config "$CONFIG" \
    --hidden-tests "$HIDDEN_TESTS" \
    --output "$ORACLE_PATH"
  oracle_exit_code=$?

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_long_horizon_run.py" \
    --project-id "$PROJECT_ID" \
    --state-root "$STATE_ROOT" \
    --telemetry-root "$TELEMETRY_ROOT" \
    --workspace "$WORKSPACE" \
    --console-log "$CONSOLE_LOG" \
    --oracle "$ORACLE_PATH" \
    --started-at "$STARTED_AT" \
    --ended-at "$ENDED_AT" \
    --elapsed-seconds "$ELAPSED_SECONDS" \
    --minimum-target-seconds "$MINIMUM_SECONDS" \
    --configured-project-limit-seconds 5400 \
    --cli-exit-code "$CLI_EXIT_CODE" \
    --output "$SUMMARY_PATH"
  summary_exit_code=$?

  echo "project_id: $PROJECT_ID"
  echo "workspace: $WORKSPACE"
  echo "state: $STATE_ROOT/$PROJECT_ID/project_state.json"
  echo "telemetry: $TELEMETRY_ROOT"
  echo "console_log: $CONSOLE_LOG"
  echo "oracle: $ORACLE_PATH"
  echo "summary: $SUMMARY_PATH"
  echo "elapsed_seconds: $ELAPSED_SECONDS"
  echo "resume: LONGRUN_WORKSPACE=\"$WORKSPACE\" LONGRUN_PLAN_FILE=\"$LONGRUN_PLAN_FILE\" $AGENT_BIN project resume --config \"$CONFIG\" --project-id \"$PROJECT_ID\""

  result_exit_code=$original_exit_code
  if (( CLI_EXIT_CODE != 0 || oracle_exit_code != 0 || summary_exit_code != 0 || ELAPSED_SECONDS < MINIMUM_SECONDS )); then
    result_exit_code=1
  fi
  exit "$result_exit_code"
}

trap finalize_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command_args=(
  "$AGENT_BIN" project start
  --config "$CONFIG"
  --task-file "$TASK_FILE"
  --project-id "$PROJECT_ID"
)

set +e
if command -v timeout >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT_SECONDS" "${command_args[@]}" 2>&1 | tee "$CONSOLE_LOG"
  CLI_EXIT_CODE=${PIPESTATUS[0]}
else
  "${command_args[@]}" 2>&1 | tee "$CONSOLE_LOG"
  CLI_EXIT_CODE=${PIPESTATUS[0]}
fi
set -e

exit "$CLI_EXIT_CODE"
