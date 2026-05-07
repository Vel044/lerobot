#!/usr/bin/env bash
# D 组 — ftrace 实时导出：后台 cat trace_pipe 持续写文件
# 用法：./run_D_ftrace_realtime.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
echo "=== Group D: ftrace realtime, $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"
    ftrace_enable
    sudo sh -c "echo 1 > $TRACEFS/tracing_on"
    sudo cat "$TRACEFS/trace_pipe" > "$RAW_DIR/ftrace_D_${run}.txt" &
    pipe_pid=$!

    "${RECORD_CMD[@]}" --timing_tag="D_${run}"

    sudo kill "$pipe_pid" 2>/dev/null || true
    wait "$pipe_pid" 2>/dev/null || true
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    ftrace_disable

    trace_mb=$(du -m "$RAW_DIR/ftrace_D_${run}.txt" | cut -f1)
    lost=$(grep -c "lost" "$RAW_DIR/ftrace_D_${run}.txt" 2>/dev/null || echo "0")
    echo "  trace: ${trace_mb}MB, lost events: $lost"
    echo ""
    sleep 30
done
echo "=== Group D done ==="
