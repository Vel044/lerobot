#!/usr/bin/env python3
"""实验 08：合并 robot_client 和 policy_server 的异步推理统计 CSV。

默认输出到本实验目录下的 async_merged_stats.csv。

对齐模式：
    auto      优先按 episode_idx 对齐，其次按 timestep reset，最后按采集顺序等分
    episode   强制按 episode_idx join
    timestep-reset 强制按 timestep 回到 0 的位置切分 episode
    row-split 强制按 client 行数等分 server 推理记录

当前 hold 扫描数据中 policy_server 持续运行，server 端 episode_idx 全为 24，
但每组开始时 timestep 会回到 0，因此 auto 会选择 timestep-reset。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize(values: pd.Series) -> dict[str, float | int]:
    values = values.astype(float)
    return {
        "t_infer_count": int(values.count()),
        "t_infer_mean": float(values.mean()) if len(values) else np.nan,
        "t_infer_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "t_infer_min": float(values.min()) if len(values) else np.nan,
        "t_infer_max": float(values.max()) if len(values) else np.nan,
    }


def can_join_by_episode(client: pd.DataFrame, server: pd.DataFrame) -> bool:
    client_eps = set(client["episode_idx"].dropna().astype(int))
    server_eps = set(server["episode_idx"].dropna().astype(int))
    return len(server_eps) > 1 and client_eps.issubset(server_eps)


def split_by_timestep_reset(server: pd.DataFrame) -> list[pd.DataFrame]:
    server = server.reset_index(drop=True)
    timesteps = server["timestep"].astype(int).to_numpy()
    starts = [0]
    for i in range(1, len(timesteps)):
        if timesteps[i] < timesteps[i - 1]:
            starts.append(i)
    starts.append(len(server))
    return [server.iloc[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


def can_split_by_timestep_reset(client: pd.DataFrame, server: pd.DataFrame) -> bool:
    return len(split_by_timestep_reset(server)) == len(client)


def summarize_by_episode(client: pd.DataFrame, server: pd.DataFrame, skip_warmup: int) -> pd.DataFrame:
    rows = []
    for row in client.itertuples(index=False):
        ep_data = server[server["episode_idx"] == row.episode_idx].copy()
        if skip_warmup > 0 and len(ep_data) > skip_warmup:
            ep_data = ep_data.iloc[skip_warmup:]
        item = {"episode_idx": row.episode_idx}
        item.update(_summarize(ep_data["t_infer_s"]))
        rows.append(item)
    return pd.DataFrame(rows)


def summarize_by_row_split(client: pd.DataFrame, server: pd.DataFrame, skip_warmup: int) -> pd.DataFrame:
    server = server.reset_index(drop=True)
    chunks = [server.iloc[idx] for idx in np.array_split(np.arange(len(server)), len(client))]
    rows = []
    for client_row, chunk in zip(client.itertuples(index=False), chunks):
        if skip_warmup > 0 and len(chunk) > skip_warmup:
            chunk = chunk.iloc[skip_warmup:]
        item = {"episode_idx": client_row.episode_idx}
        item.update(_summarize(chunk["t_infer_s"]))
        rows.append(item)
    return pd.DataFrame(rows)


def summarize_by_timestep_reset(client: pd.DataFrame, server: pd.DataFrame, skip_warmup: int) -> pd.DataFrame:
    chunks = split_by_timestep_reset(server)
    rows = []
    for client_row, chunk in zip(client.itertuples(index=False), chunks):
        if skip_warmup > 0 and len(chunk) > skip_warmup:
            chunk = chunk.iloc[skip_warmup:]
        item = {"episode_idx": client_row.episode_idx}
        item.update(_summarize(chunk["t_infer_s"]))
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并异步推理统计 CSV")
    here = Path(__file__).resolve().parent
    parser.add_argument("--client-csv", "--client_csv", type=Path, default=here / "async_client_stats.csv")
    parser.add_argument("--server-csv", "--server_csv", type=Path, default=here / "async_server_stats.csv")
    parser.add_argument("--output", type=Path, default=here / "async_merged_stats.csv")
    parser.add_argument("--match-mode", choices=["auto", "episode", "timestep-reset", "row-split"], default="auto")
    parser.add_argument("--skip-warmup", "--skip_warmup", type=int, default=2)
    args = parser.parse_args()

    client = pd.read_csv(args.client_csv).sort_values("hold").reset_index(drop=True)
    # server CSV 可能无 header（policy_server 在文件被删后若进程未重启会漏写 header）
    with open(args.server_csv, newline="") as f:
        first_line = f.readline().strip()
    has_header = not first_line[0].isdigit() if first_line else True
    if has_header:
        server = pd.read_csv(args.server_csv)
    else:
        server = pd.read_csv(args.server_csv, header=None, names=["episode_idx", "timestep", "t_infer_s"])

    mode = args.match_mode
    if mode == "auto":
        if can_join_by_episode(client, server):
            mode = "episode"
        elif can_split_by_timestep_reset(client, server):
            mode = "timestep-reset"
        else:
            mode = "row-split"

    if mode == "episode":
        tinfer = summarize_by_episode(client, server, args.skip_warmup)
    elif mode == "timestep-reset":
        tinfer = summarize_by_timestep_reset(client, server, args.skip_warmup)
    else:
        tinfer = summarize_by_row_split(client, server, args.skip_warmup)

    merged = client.merge(tinfer, on="episode_idx", how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, float_format="%.4f")
    print(f"对齐模式: {mode}")
    print(f"合并完成: {len(merged)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
