#!/usr/bin/env bash
# A 组 — baseline：无 tracing
# 用法：./run_A_baseline.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
echo "=== Group A: baseline, $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"
    "${RECORD_CMD[@]}" --timing_tag="A_${run}"
    echo ""
    sleep 30
done
echo "=== Group A done ==="
