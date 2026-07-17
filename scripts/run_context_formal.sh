#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 自动加载项目根目录下的 .env
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=".runs/context_evals/formal_${STAMP}"

PROBES=(
  position
  multi_constraint
  state_tracking
  aggregation
)

LENGTHS=(
  2048
  4096
  8192
  12288
)

MODES="full_history,recent_window,deterministic_prune,structured_reset"
SAMPLES=20

mkdir -p "$ROOT"

echo "Formal context experiment"
echo "Output root: $ROOT"
echo "Samples per cell: $SAMPLES"
echo "Modes: $MODES"
echo

for probe in "${PROBES[@]}"; do
  for length in "${LENGTHS[@]}"; do
    OUT="${ROOT}/${probe}/${length}"
    mkdir -p "$OUT"

    echo "=================================================="
    echo "Running probe=${probe}, length=${length}"
    echo "Output: ${OUT}"
    echo "=================================================="

    python evals/context_lifecycle/runner.py \
      --config evals/context_lifecycle/config.yaml \
      --probe "$probe" \
      --lengths "$length" \
      --samples "$SAMPLES" \
      --modes "$MODES" \
      --output-dir "$OUT" \
      2>&1 | tee "${OUT}/console.log"
  done
done

echo
echo "Formal experiment complete: $ROOT"
