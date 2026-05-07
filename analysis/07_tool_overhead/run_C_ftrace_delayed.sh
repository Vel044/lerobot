#!/usr/bin/env bash
# C 组 — ftrace 延迟导出（精确推理窗口版）
# episode 期间 ring buffer 一直开着，但 tracing_on 只在推理阶段开启
# 推理开始/结束由 record.py 通过 LEROBOT_INFERENCE_FIFO 通知
# 用法：./run_C_ftrace_delayed.sh [NUM_RUNS]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NUM_RUNS=${1:-$NUM_RUNS}

mkdir -p "$RAW_DIR"
# 信号 FIFO：record.py -> listener
FIFO="/tmp/lerobot_inference_$$"
# 就绪 FIFO：listener -> script（确认 listener 已打开信号 FIFO 的读端）
READY_FIFO="/tmp/lerobot_inference_ready_$$"

cleanup() {
    rm -f "$FIFO" "$READY_FIFO"
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Group C: ftrace delayed (inference windows), $NUM_RUNS runs ==="

for ((run=1; run<=NUM_RUNS; run++)); do
    echo "--- run $run ---"

    rm -f "$FIFO" "$READY_FIFO"
    mkfifo "$FIFO"
    mkfifo "$READY_FIFO"

    # 一次性设置好 tracepoint（ring buffer 一直分配，但 tracing_on 初始关闭）
    ftrace_enable
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"

    # 启动 FIFO 监听器：收到 START → 开 tracing，收到 END → 关 tracing
    # 打开 READY_FIFO 写端来通知主脚本：信号 FIFO 的读端已就绪
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    (
        tracing_on=0
        exec 3<>"$READY_FIFO"  # 打开就绪管道（写端），通知主脚本可以继续

        while IFS= read -r signal; do
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
        done < "$FIFO"
    ) &
    fifo_pid=$!

    # 等待 listener 打开 READY_FIFO（表示信号 FIFO 读端已就绪）
    # 这确保 record.py 打开 FIFO 写端时不会阻塞
    read -r < "$READY_FIFO"

    # 运行 record.py，传入 FIFO 路径
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
