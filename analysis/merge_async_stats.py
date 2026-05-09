"""合并 robot_client 和 policy_server 的异步推理统计 CSV。

用法:
    python merge_async_stats.py [--client_csv PATH] [--server_csv PATH] [--output PATH]

默认路径:
    client: ~/lerobot/analysis/async_client_stats.csv
    server: ~/lerobot/analysis/async_server_stats.csv
    output: ~/lerobot/analysis/async_merged_stats.csv

合并逻辑:
    1. server 端按 episode_idx 分组，计算 T_infer 的 mean/std
    2. 与 client 端按 episode_idx join
    3. 输出合并后的 CSV（含 t_infer_mean, t_infer_std 列）
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="合并异步推理统计 CSV")
    parser.add_argument("--client_csv", default=os.path.expanduser("~/lerobot/analysis/async_client_stats.csv"))
    parser.add_argument("--server_csv", default=os.path.expanduser("~/lerobot/analysis/async_server_stats.csv"))
    parser.add_argument("--output", default=os.path.expanduser("~/lerobot/analysis/async_merged_stats.csv"))
    args = parser.parse_args()

    client = pd.read_csv(args.client_csv)
    server = pd.read_csv(args.server_csv)

    # server 端按 episode 汇总 T_infer 均值/std
    tinfer = server.groupby("episode_idx")["t_infer_s"].agg(
        t_infer_mean="mean", t_infer_std="std"
    ).reset_index()

    merged = client.merge(tinfer, on="episode_idx", how="left")
    merged.to_csv(args.output, index=False)
    print(f"合并完成: {len(merged)} 行 → {args.output}")


if __name__ == "__main__":
    main()
