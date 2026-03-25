#!/usr/bin/env python3
import os
import re
import sys
from collections import defaultdict


TRACE_TIMES_FILE = os.path.expanduser("~/lerobot/syscall-analyze/.lerobot_trace_times.txt")


def get_episode_time():
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
    if not episode_starts or not episode_ends:
        return True
    for start, end in zip(episode_starts, episode_ends):
        if start <= timestamp <= end:
            return True
    return False


def parse_futex_log(filename, episode_starts=None, episode_ends=None):
    thread_stats = defaultdict(lambda: {'count': 0, 'total_ns': 0, 'cpu_ns': 0, 'sleep_ns': 0})

    with open(filename, 'r') as f:
        for line in f:
            if 'futex:' not in line:
                continue
            
            timestamp_match = re.search(r'\s+([\d.]+):', line)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else None
            
            if timestamp and not is_in_episode_window(timestamp, episode_starts, episode_ends):
                continue
            
            thread_match = re.search(r'^\s+(\S+)', line)
            thread_name = thread_match.group(1) if thread_match else "unknown"
            
            m = re.search(r'total=(\d+) ns.*?cpu=(\d+) ns.*?sleep=(\d+) ns', line)
            if m:
                total = int(m.group(1))
                cpu = int(m.group(2))
                sleep = int(m.group(3))
                
                thread_stats[thread_name]['count'] += 1
                thread_stats[thread_name]['total_ns'] += total
                thread_stats[thread_name]['cpu_ns'] += cpu
                thread_stats[thread_name]['sleep_ns'] += sleep

    return thread_stats


def main():
    filename = "/home/vel/lerobot/syscall-analyze/full_trace.txt"
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    print(f"分析文件: {filename}")
    print("=" * 70)

    episode_starts, episode_ends = get_episode_time()
    
    if episode_starts and episode_ends:
        episode_duration = episode_ends[-1] - episode_starts[0]
        print(f"【Episode 时间】: {episode_starts[0]:.3f} - {episode_ends[-1]:.3f} (持续 {episode_duration:.2f}s)")
    else:
        print("未找到 episode 时间记录")
        return

    thread_stats = parse_futex_log(filename, episode_starts, episode_ends)
    episode_duration_ns = (episode_ends[-1] - episode_starts[0]) * 1e9

    def get_thread_prefix(name):
        if name.startswith('python'):
            return 'python'
        elif name.startswith('node'):
            return 'node'
        elif name.startswith('libuv'):
            return 'libuv'
        elif name.startswith('ckg_server'):
            return 'ckg_server'
        elif name.startswith('tokio'):
            return 'tokio'
        elif name.startswith('t:ttnet'):
            return 'ttnet'
        else:
            return 'other'

    grouped_stats = defaultdict(lambda: {'count': 0, 'total_ns': 0, 'cpu_ns': 0, 'sleep_ns': 0, 'threads': []})
    for thread_name, stats in thread_stats.items():
        prefix = get_thread_prefix(thread_name)
        grouped_stats[prefix]['count'] += stats['count']
        grouped_stats[prefix]['total_ns'] += stats['total_ns']
        grouped_stats[prefix]['cpu_ns'] += stats['cpu_ns']
        grouped_stats[prefix]['sleep_ns'] += stats['sleep_ns']
        grouped_stats[prefix]['threads'].append((thread_name, stats))

    sorted_groups = sorted(grouped_stats.items(), key=lambda x: x[1]['cpu_ns'], reverse=True)

    total_count = sum(s['count'] for _, s in grouped_stats.items())
    total_time = sum(s['total_ns'] for _, s in grouped_stats.items())
    total_cpu = sum(s['cpu_ns'] for _, s in grouped_stats.items())
    total_sleep = sum(s['sleep_ns'] for _, s in grouped_stats.items())

    cpu_ratio = (total_cpu / episode_duration_ns) * 100 if episode_duration_ns else 0
    sleep_ratio = (total_sleep / episode_duration_ns) * 100 if episode_duration_ns else 0
    total_ratio = (total_time / episode_duration_ns) * 100 if episode_duration_ns else 0

    print(f"\n【Futex 按线程统计】(Episode 内, 按CPU时间降序)")
    print("-" * 85)
    print(f"{'线程名称':<25} {'调用次数':>10} {'Futex总时间':>12} {'占Episode%':>10} {'CPU(s)':>10} {'Sleep(s)':>12}")
    print("-" * 85)
    
    for group_name, group_stats in sorted_groups:
        sorted_threads = sorted(group_stats['threads'], key=lambda x: x[1]['cpu_ns'], reverse=True)
        group_ratio = (group_stats['total_ns'] / episode_duration_ns * 100) if episode_duration_ns else 0
        total_line = f"{group_name} (total):"
        print(f"{total_line:<25} {group_stats['count']:>10,} {group_stats['total_ns']/1e9:>12.4f} {group_ratio:>10.2f} {group_stats['cpu_ns']/1e9:>10.6f} {group_stats['sleep_ns']/1e9:>12.4f}")
        for i, (thread_name, stats) in enumerate(sorted_threads):
            ratio = (stats['total_ns'] / episode_duration_ns * 100) if episode_duration_ns else 0
            cpu_s = stats['cpu_ns'] / 1e9
            sleep_s = stats['sleep_ns'] / 1e9
            print(f"{'':25} {stats['count']:>10,} {stats['total_ns']/1e9:>12.4f} {'':>10} {cpu_s:>10.6f} {sleep_s:>12.4f}")
    
    print("-" * 85)
    print(f"{'总计':<25} {total_count:>10,} {total_time/1e9:>12.4f} {(total_time/episode_duration_ns)*100 if episode_duration_ns else 0:>10.2f} {total_cpu/1e9:>10.6f} {total_sleep/1e9:>12.4f}")
    
    print(f"\n【Episode 时间分析】(总时长: {episode_duration:.2f}s)")
    print("-" * 85)
    print(f"  ★ Futex总时间占Episode时间: {total_time/1e9:.4f}s ({total_ratio:.4f}%)")
    print(f"  ★ CPU时间占Episode时间: {total_cpu/1e9:.4f}s ({cpu_ratio:.4f}%)")
    print(f"  ★ 睡眠时间占Episode时间: {total_sleep/1e9:.4f}s ({sleep_ratio:.4f}%)")


if __name__ == "__main__":
    main()
