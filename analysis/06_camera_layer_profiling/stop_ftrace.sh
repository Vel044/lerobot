#!/usr/bin/env bash
# 实验 06：停止 ftrace 并导出 trace log
set -euo pipefail

TRACE=/sys/kernel/debug/tracing
TS=$(date +%s)
OUT=/tmp/lerobot_ftrace_${TS}.log

echo 0 > "$TRACE/tracing_on"
cp "$TRACE/trace" "$OUT"

# 关闭 tracepoint
echo 0 > "$TRACE/events/syscalls/sys_enter_pselect6/enable"
echo 0 > "$TRACE/events/syscalls/sys_exit_pselect6/enable"
echo 0 > "$TRACE/events/syscalls/sys_enter_ioctl/enable"
echo 0 > "$TRACE/events/syscalls/sys_exit_ioctl/enable"
echo > "$TRACE/set_event_pid" || true

echo "[ok] ftrace log: $OUT"
echo "[info] use parse_ftrace.py to extract layer 1 (pselect6) and layer 2 (VIDIOC_DQBUF=0xc0585611)"
