#!/usr/bin/env bash
# D 组 — ftrace 实时导出：收到信号后开始/结束 tracing
# trace_pipe 实时流式导出到文件，只在 episode 期间有效
# 用法：./run_D_ftrace_realtime.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
FIFO="/tmp/lerobot_inference_$$"
READY_MARKER="/tmp/lerobot_listener_ready_$$"

cleanup() {
    rm -f "$FIFO" "$READY_MARKER"
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Group D: ftrace realtime (episode windows), $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"

    rm -f "$FIFO" "$READY_MARKER"
    mkfifo "$FIFO"

    # 设置 tracepoint，清空 ring buffer，tracing 初始关闭
    ftrace_enable
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    # 清空 buffer（确保 episode 开始前是干净的）
    sudo sh -c "echo > $TRACEFS/trace"

    # 启动 FIFO 监听器：
    #   START → trace_pipe 先启动读 → 再开启 tracing_on
    #   END   → 关闭 tracing_on，kill trace_pipe
    (
        exec 3<>"$FIFO"
        > "$READY_MARKER"
        while IFS= read -r signal <&3; do
            case "$signal" in
                START)
                    # 先启动 trace_pipe 读端，确保在 tracing_on 之前就开始消费
                    sudo cat "$TRACEFS/trace_pipe" > "$RAW_DIR/ftrace_D_${run}.txt" &
                    # 再开启 tracing
                    sudo sh -c "echo 1 > $TRACEFS/tracing_on"
                    echo "[ftrace] tracing started (realtime)" >&2
                    ;;
                END)
                    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
                    sudo kill "$(pgrep -f "cat.*trace_pipe")" 2>/dev/null || true
                    echo "[ftrace] tracing stopped (realtime)" >&2
                    ;;
            esac
        done
    ) &
    fifo_pid=$!

    # 等待 listener 准备好
    for i in {1..60}; do
        if [[ -f "$READY_MARKER" ]] && kill -0 "$fifo_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    # 运行 record.py
    PYTHONPATH=/home/vel/lerobot/src:${PYTHONPATH:-} LEROBOT_INFERENCE_FIFO="$FIFO" "${RECORD_CMD[@]}" --timing_tag="D_${run}"

    # 等待 listener 处理完
    kill "$fifo_pid" 2>/dev/null || true
    wait "$fifo_pid" 2>/dev/null || true

    # 确保 tracing 关闭
    sudo sh -c "echo 0 > $TRACEFS/tracing_on" 2>/dev/null || true
    sudo kill "$(pgrep -f "cat.*trace_pipe")" 2>/dev/null || true

    ftrace_disable

    trace_mb=$(du -m "$RAW_DIR/ftrace_D_${run}.txt" 2>/dev/null | cut -f1 || echo "0")
    lost=$(grep -c "lost" "$RAW_DIR/ftrace_D_${run}.txt" 2>/dev/null || echo "0")
    echo "  trace: ${trace_mb}MB, lost events: $lost"
    echo ""
    sleep 1
done
echo "=== Group D done ==="
