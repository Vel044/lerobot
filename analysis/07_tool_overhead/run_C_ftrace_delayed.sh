#!/usr/bin/env bash
# C 组 — ftrace 延迟导出：episode 期间只写 ring buffer，结束后才读出
# 用法：./run_C_ftrace_delayed.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
echo "=== Group C: ftrace delayed, $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"
    ftrace_enable
    sudo sh -c "echo 1 > $TRACEFS/tracing_on"

    "${RECORD_CMD[@]}" --timing-tag="C_${run}"

    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    sudo cat "$TRACEFS/trace" > "$RAW_DIR/ftrace_C_${run}.txt"
    ftrace_disable

    trace_mb=$(du -m "$RAW_DIR/ftrace_C_${run}.txt" | cut -f1)
    lost=$(grep -c "lost" "$RAW_DIR/ftrace_C_${run}.txt" 2>/dev/null || echo "0")
    echo "  trace: ${trace_mb}MB, lost events: $lost"
    echo ""
    sleep 30
done
echo "=== Group C done ==="
