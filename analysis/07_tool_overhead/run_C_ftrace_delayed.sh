#!/usr/bin/env bash
# C 组 — ftrace 延迟导出（精确推理窗口版）
# 推理开始/结束由 record.py 通过 LEROBOT_INFERENCE_FIFO 通知
# 用法：./run_C_ftrace_delayed.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
FIFO="/tmp/lerobot_inference_$$"
PID_FILE="/tmp/lerobot_listener_$$.pid"
READY_MARKER="/tmp/lerobot_listener_ready_$$"

cleanup() {
    rm -f "$FIFO" "$PID_FILE" "$READY_MARKER"
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Group C: ftrace delayed (inference windows), $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"

    rm -f "$FIFO" "$PID_FILE" "$READY_MARKER"
    mkfifo "$FIFO"

    ftrace_enable
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"

    # 启动 FIFO 监听器：收到 START → 开 tracing，收到 END → 关 tracing
    # 监听器独立运行，不阻塞主脚本
    (
        exec 3<>"$FIFO"
        echo $$ > "$PID_FILE"
        > "$READY_MARKER"   # 通知主脚本：读端已打开
        tracing_on=0
        while IFS= read -r signal <&3; do
            case "$signal" in
                START)
                    if [[ "$tracing_on" -eq 0 ]]; then
                        sudo sh -c "echo 1 > $TRACEFS/tracing_on"
                        tracing_on=1
                        echo "[ftrace] tracing started" >&2
                    fi
                    ;;
                END)
                    if [[ "$tracing_on" -eq 1 ]]; then
                        sudo sh -c "echo 0 > $TRACEFS/tracing_on"
                        tracing_on=0
                        echo "[ftrace] tracing stopped" >&2
                    fi
                    ;;
            esac
        done
    ) &
    fifo_pid=$!

    # 等待 listener 准备好（PID_FILE 存在表示子进程的 exec 已完成 fd 打开）
    # 使用 poll 轮询而非 sleep，避免过早或过晚启动 record.py
    for i in {1..60}; do
        if [[ -f "$READY_MARKER" ]] && kill -0 "$fifo_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    # 运行 record.py
    LEROBOT_INFERENCE_FIFO="$FIFO" "${RECORD_CMD[@]}" --timing_tag="C_${run}"

    # 等待 listener 处理完所有信号
    kill "$fifo_pid" 2>/dev/null || true
    wait "$fifo_pid" 2>/dev/null || true

    # 关闭 tracing
    sudo sh -c "echo 0 > $TRACEFS/tracing_on" 2>/dev/null || true

    # 导出 ring buffer 内容
    sudo cat "$TRACEFS/trace" > "$RAW_DIR/ftrace_C_${run}.txt"
    ftrace_disable

    trace_mb=$(du -m "$RAW_DIR/ftrace_C_${run}.txt" | cut -f1)
    lost=$(grep -c "lost" "$RAW_DIR/ftrace_C_${run}.txt" 2>/dev/null || echo "0")
    echo "  trace: ${trace_mb}MB, lost events: $lost"
    echo ""
    sleep 30
done
echo "=== Group C done ==="