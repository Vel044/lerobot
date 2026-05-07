#!/usr/bin/env bash
# 实验 03：开启摄像头链路相关的 ftrace tracepoint（layer 1/2）
# 用法：sudo ./start_ftrace.sh [LEROBOT_PID]
set -euo pipefail

TRACE=/sys/kernel/debug/tracing
PID=${1:-}

# 清空 + 配置
echo > "$TRACE/trace"
echo 16384 > "$TRACE/buffer_size_kb"

# 启用 tracepoint
echo 1 > "$TRACE/events/syscalls/sys_enter_pselect6/enable"
echo 1 > "$TRACE/events/syscalls/sys_exit_pselect6/enable"
echo 1 > "$TRACE/events/syscalls/sys_enter_ioctl/enable"
echo 1 > "$TRACE/events/syscalls/sys_exit_ioctl/enable"

# 可选：限定到 LeRobot 进程
if [[ -n "$PID" ]]; then
    echo "$PID" > "$TRACE/set_event_pid"
    echo "[info] ftrace pid filter set to $PID"
fi

echo 1 > "$TRACE/tracing_on"
echo "[ok] ftrace started; run profile_camera.py --use-ftrace, then ./stop_ftrace.sh"
