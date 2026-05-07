#!/usr/bin/env bash
# _common.sh — 工具开销校准实验共享配置
# 被 run_A/B/C/D/E.sh source，不单独运行
set -euo pipefail

NUM_RUNS=${NUM_RUNS:-1}
OVERHEAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$OVERHEAD_DIR/raw"
TRACEFS=/sys/kernel/debug/tracing

# record 公共参数
RECORD_CMD=(
    python -m lerobot.record
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM0
    --robot.id=R12254705
    --teleop.type=so101_leader
    --teleop.port=/dev/ttyACM1
    --teleop.id=R07254705
    --robot.disable_torque_on_disconnect=true
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}"
    --dataset.single_task="Put the bottle into the black basket."
    --policy.path=/home/vel/so101-bottle/last/pretrained_model
    --dataset.repo_id="${HF_USER}/eval_so101_bottle"
    --dataset.push_to_hub=false
    --dataset.num_episodes=1
    --dataset.episode_time_s=30
    --dataset.reset_time_s=1
    --resume=true
)

# ftrace 需要的 tracepoint（目录格式：category/event_name）
TRACEPOINTS=(
    syscalls/sys_enter_ioctl   syscalls/sys_exit_ioctl
    syscalls/sys_enter_pselect6 syscalls/sys_exit_pselect6
    syscalls/sys_enter_read    syscalls/sys_exit_read
    syscalls/sys_enter_write   syscalls/sys_exit_write
    syscalls/sys_enter_clock_nanosleep syscalls/sys_exit_clock_nanosleep
    syscalls/sys_enter_futex   syscalls/sys_exit_futex
)

ftrace_enable() {
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    sudo sh -c "echo > $TRACEFS/trace"
    sudo sh -c "echo 32768 > $TRACEFS/buffer_size_kb"
    for tp in "${TRACEPOINTS[@]}"; do
        sudo sh -c "echo 1 > $TRACEFS/events/$tp/enable"
    done
}

ftrace_disable() {
    sudo sh -c "echo 0 > $TRACEFS/tracing_on"
    for tp in "${TRACEPOINTS[@]}"; do
        sudo sh -c "echo 0 > $TRACEFS/events/$tp/enable"
    done
}
