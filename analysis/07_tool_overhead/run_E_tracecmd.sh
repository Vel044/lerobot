#!/usr/bin/env bash
# E 组 — trace-cmd record
# 用法：./run_E_tracecmd.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
echo "=== Group E: trace-cmd, $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"
    trace-cmd record -o "$RAW_DIR/trace_E_${run}.dat" \
        $(for tp in "${TRACEPOINTS[@]}"; do echo "-e $tp"; done) \
        -- "${RECORD_CMD[@]}" --timing-tag="E_${run}"

    trace_mb=$(du -m "$RAW_DIR/trace_E_${run}.dat" | cut -f1)
    echo "  trace: ${trace_mb}MB"
    echo ""
    sleep 30
done
echo "=== Group E done ==="
