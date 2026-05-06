#!/usr/bin/env python3
import os
import re
import sys


TRACE_TIMES_FILE = os.path.expanduser("~/lerobot/syscall-analyze/.lerobot_trace_times.txt")


def get_episode_time():
    """获取 episode 时间（绝对时间）"""
    if not os.path.exists(TRACE_TIMES_FILE):
        return None, None
    
    episode_starts = []
    episode_ends = []
    
    with open(TRACE_TIMES_FILE, 'r') as f:
        for line in f:
            match = re.search(r'episode_start\|([\d.]+)', line)
            if match:
                episode_starts.append(float(match.group(1)))
            
            match = re.search(r'episode_end\|([\d.]+)', line)
            if match:
                episode_ends.append(float(match.group(1)))
    
    return episode_starts, episode_ends


def is_in_episode_window(timestamp, episode_starts, episode_ends):
    """检查时间戳是否在 episode 时间窗口内"""
    if not episode_starts or not episode_ends:
        return True
    for start, end in zip(episode_starts, episode_ends):
        if start <= timestamp <= end:
            return True
    return False


def parse_ps6_log(filename, episode_starts=None, episode_ends=None):
    """解析ps6日志文件，提取关键数据"""
    total_context = 0
    total_sleep = 0
    total_cpu = 0
    call_count = 0

    with open(filename, 'r') as f:
        for line in f:
            m = re.search(r'pselect6:.*?total=(\d+) ns.*?sleep=(\d+) ns.*?context=(\d+) ns', line)
            if m:
                timestamp_match = re.search(r'\s+([\d.]+):.*?pselect6:', line)
                timestamp = float(timestamp_match.group(1)) if timestamp_match else None
                
                total = int(m.group(1))
                sleep = int(m.group(2))
                context = int(m.group(3))
                cpu = total - sleep

                if timestamp and is_in_episode_window(timestamp, episode_starts, episode_ends):
                    call_count += 1
                    total_context += context
                    total_sleep += sleep
                    total_cpu += cpu

    return {
        'call_count': call_count,
        'total_context': total_context,
        'total_sleep': total_sleep,
        'total_cpu': total_cpu,
    }


def main():
    filename = "/home/vel/lerobot/syscall-analyze/full_trace.txt"
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    print(f"分析文件: {filename}")
    print("=" * 50)

    episode_starts, episode_ends = get_episode_time()
    
    if episode_starts and episode_ends:
        episode_duration = episode_ends[-1] - episode_starts[0]
        print(f"【Episode 时间】: {episode_starts[0]:.3f} - {episode_ends[-1]:.3f} (持续 {episode_duration:.2f}s)")
    else:
        print("未找到 episode 时间记录")
        return

    stats = parse_ps6_log(filename, episode_starts, episode_ends)
    episode_duration_ns = (episode_ends[-1] - episode_starts[0]) * 1e9

    print(f"\n【Python pselect6 调用统计】(Episode 内)")
    print(f"  调用次数: {stats['call_count']}")
    print(f"  总context时间: {stats['total_context']:,} ns ({stats['total_context']/1e6:.2f} ms)")
    print(f"  总sleep时间: {stats['total_sleep']:,} ns ({stats['total_sleep']/1e6:.2f} ms)")
    print(f"  总CPU时间: {stats['total_cpu']:,} ns ({stats['total_cpu']/1e6:.2f} ms)")
    
    print(f"\n【时间占比分析】")
    cpu_ratio = (stats['total_cpu'] / episode_duration_ns) * 100
    sleep_ratio = (stats['total_sleep'] / episode_duration_ns) * 100
    context_ratio = (stats['total_context'] / episode_duration_ns) * 100
    print(f"  ★ CPU时间占Episode时间比例: {cpu_ratio:.4f}%")
    print(f"  ★ Sleep时间占Episode时间比例: {sleep_ratio:.4f}%")
    print(f"  ★ Context时间占Episode时间比例: {context_ratio:.4f}%")


if __name__ == "__main__":
    main()
