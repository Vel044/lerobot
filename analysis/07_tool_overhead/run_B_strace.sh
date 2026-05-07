#!/usr/bin/env bash
# B 组 — strace
# 用法：./run_B_strace.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
echo "=== Group B: strace, $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"
    strace -f -ttT -o "$RAW_DIR/strace_B_${run}.log" \
        "${RECORD_CMD[@]}" --timing_tag="B_${run}"

    trace_mb=$(du -m "$RAW_DIR/strace_B_${run}.log" | cut -f1)
    echo "  trace: ${trace_mb}MB"
    echo ""
    sleep 30
done
echo "=== Group B done ==="
