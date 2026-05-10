#!/usr/bin/env python3
"""实验 08：异步推理 hold 阈值扫描 - 数据分析与画图。

输入：
    - async_client_stats.csv：robot_client 每个 episode 的统计
    - async_server_stats.csv：policy_server 每次推理的耗时
    - ../04_tool_overhead/timing_stats.csv：同步 baseline 的 fps

输出：
    - async_hold_summary.csv
    - chart1_fps_vs_sync.png
    - chart2_must_go.png
    - chart3_frame_breakdown.png
    - chart4_tinfer_boxplot.png

用法：
    python plot_async_sweep.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_ASYNC = "#4C72B0"
C_SYNC = "#C44E52"
C_EXEC = "#55A868"
C_EXPIRE = "#E8C547"
C_STALL = "#C44E52"
C_TINF = "#8172B2"


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_sync_fps(sync_csv: Path, sync_tag: str) -> float:
    sync = pd.read_csv(sync_csv)
    rows = sync[sync["timing_tag"] == sync_tag]
    if rows.empty:
        raise ValueError(f"{sync_csv} 中没有 timing_tag={sync_tag!r} 的同步 baseline")
    return float(rows.iloc[0]["fps_mean"])


def split_by_timestep_reset(server: pd.DataFrame) -> list[pd.DataFrame]:
    server = server.reset_index(drop=True)
    timesteps = server["timestep"].astype(int).to_numpy()
    starts = [0]
    for i in range(1, len(timesteps)):
        if timesteps[i] < timesteps[i - 1]:
            starts.append(i)
    starts.append(len(server))
    return [server.iloc[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


def split_server_tinfer(server: pd.DataFrame, client: pd.DataFrame, skip_warmup: int) -> list[np.ndarray]:
    """按 timestep reset 切分 server 日志，必要时退回按行等分。"""
    chunks = split_by_timestep_reset(server)
    if len(chunks) != len(client):
        server = server.reset_index(drop=True)
        chunks = [server.iloc[idx] for idx in np.array_split(np.arange(len(server)), len(client))]
    out: list[np.ndarray] = []
    for chunk in chunks:
        if skip_warmup > 0 and len(chunk) > skip_warmup:
            chunk = chunk.iloc[skip_warmup:]
        out.append(chunk["t_infer_s"].to_numpy(dtype=float))
    return out


def build_summary(
    client: pd.DataFrame,
    tinfer_data: list[np.ndarray],
    sync_fps: float,
) -> pd.DataFrame:
    summary_cols = [
        "episode_idx",
        "hold",
        "eff_fps",
        "fps_mean",
        "fps_p5",
        "fps_p95",
        "stall_frames",
        "stall_pct",
        "expire_frames",
        "expire_pct",
        "must_go_count",
        "action_frames",
        "episode_total_s",
        "success",
        "cpu_temp_start_c",
        "cpu_temp_end_c",
        "cpu_freq_mean_mhz",
    ]
    summary = client[[c for c in summary_cols if c in client.columns]].copy()
    summary["sync_baseline_fps"] = sync_fps
    summary["fps_gain_pct"] = (summary["eff_fps"] - sync_fps) / sync_fps * 100
    summary["t_infer_count"] = [len(x) for x in tinfer_data]
    summary["t_infer_mean"] = [np.mean(x) if len(x) else np.nan for x in tinfer_data]
    summary["t_infer_std"] = [np.std(x, ddof=1) if len(x) > 1 else np.nan for x in tinfer_data]
    summary["t_infer_min"] = [np.min(x) if len(x) else np.nan for x in tinfer_data]
    summary["t_infer_max"] = [np.max(x) if len(x) else np.nan for x in tinfer_data]
    return summary


def plot_fps(client: pd.DataFrame, labels: list[str], sync_fps: float, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    x_all = client["hold"].to_numpy(dtype=float)
    y_all = client["eff_fps"].to_numpy(dtype=float)

    ax.plot(
        x_all, y_all, color=C_ASYNC, linewidth=2.5, marker="o", markersize=10,
        markerfacecolor="white", markeredgewidth=2, zorder=3,
    )
    for x, y in zip(x_all, y_all):
        ax.annotate(
            f"{y:.1f}", (x, y), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=10, color=C_ASYNC,
        )

    ax.plot(
        [0.0], [sync_fps], color=C_SYNC, marker="s", markersize=8,
        markerfacecolor="white", markeredgewidth=2, linestyle="None",
        zorder=4, label="同步 baseline",
    )
    ax.annotate(
        f"{sync_fps:.1f}", (0.0, sync_fps), textcoords="offset points",
        xytext=(0, 12), ha="center", fontsize=10, color=C_SYNC,
    )

    mask_left = (x_all >= 0.10) & (x_all <= 0.30)
    x_left = np.concatenate(([0.0], x_all[mask_left]))
    y_left = np.concatenate(([sync_fps], y_all[mask_left]))
    coeffs = np.polyfit(x_left, y_left, deg=1)
    x_fit = np.linspace(0.0, 0.30, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(
        x_fit, y_fit, color="#D95F02", linewidth=2,
        linestyle="-.", alpha=0.75, zorder=2.5,
        label="线性趋势",
    )

    # 平台区：从 0.30 开始横着，取 hold=0.30 的实测值（到达满帧）
    plateau_value = float(y_all[x_all == 0.30][0])
    ax.hlines(
        plateau_value, xmin=0.30, xmax=0.62,
        color="#8172B2", linewidth=2,
        linestyle="-", alpha=0.75, zorder=2.5,
        label=f"满帧运行 {plateau_value:.1f} FPS",
    )

    ax.axhline(y=sync_fps, color=C_SYNC, linestyle="--", linewidth=1.5, alpha=0.25, zorder=1)

    ax.set_xlabel("hold (chunk_size_threshold)", fontsize=12)
    ax.set_ylabel("有效 FPS (action_frames / episode_total_s)", fontsize=12)
    ax.set_title(f"异步推理 hold 扫描：有效 FPS 与同步 baseline 对比（60s/组）", fontsize=13)
    ax.set_xlim(-0.03, 0.65)
    ax.set_ylim(0, max(y_all.max(), sync_fps) * 1.15)
    ax.set_xticks([0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
    ax.set_xticklabels(["baseline", "0.10", "0.20", "0.25", "0.30", "0.35", "0.40", "0.50", "0.60"])
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def plot_must_go(client: pd.DataFrame, labels: list[str], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(client))

    ax.plot(x, client["must_go_count"], marker="o", linewidth=2.5, color=C_SYNC, markersize=8, zorder=3)
    for xi, row in zip(x, client.itertuples()):
        ax.annotate(
            f"{int(row.must_go_count)}",
            (xi, row.must_go_count),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=11,
            color=C_SYNC,
        )

    avg_mg = client["must_go_count"].mean()
    ax.axhline(y=avg_mg, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label=f"均值 {avg_mg:.0f}")
    ax.set_xlabel("hold", fontsize=12)
    ax.set_ylabel("must_go 触发次数（每 60s）", fontsize=12)
    ax.set_title("hold 与真实 stall 触发次数", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def plot_frame_breakdown(client: pd.DataFrame, labels: list[str], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(client))
    bottom = np.zeros(len(client))

    for col, label, color in [
        ("action_frames", "执行帧", C_EXEC),
        ("expire_frames", "过期帧", C_EXPIRE),
        ("stall_frames", "stall 帧", C_STALL),
    ]:
        values = client[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.6, zorder=2)
        for xi, val, bot in zip(x, values, bottom):
            if val > 20:
                ax.text(
                    xi,
                    bot + val / 2,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom += values

    ax.set_xlabel("hold", fontsize=12)
    ax.set_ylabel("帧数（60s episode）", fontsize=12)
    ax.set_title("hold 下的帧组成：执行 / 过期 / stall", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def plot_tinfer(tinfer_data: list[np.ndarray], labels: list[str], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(tinfer_data, positions=range(len(tinfer_data)), widths=0.5, patch_artist=True, zorder=2)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_TINF)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.5)

    for i, data in enumerate(tinfer_data):
        if len(data) == 0:
            continue
        mean_val = float(np.mean(data))
        ax.scatter([i], [mean_val], color="white", s=40, zorder=4, marker="D")
        ax.annotate(
            f"均值={mean_val:.2f}s",
            (i, mean_val),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            color=C_TINF,
        )

    ax.set_xlabel("hold 配置", fontsize=12)
    ax.set_ylabel("T_infer (s)", fontsize=12)
    ax.set_title("policy_server 推理耗时分布（每组跳过前 2 次 warm-up）", fontsize=13)
    ax.set_xticks(range(len(tinfer_data)))
    ax.set_xticklabels([f"hold={x}" for x in labels])
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(1.2, 2.4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--client-csv", "--client_csv", type=Path, default=here / "async_client_stats.csv")
    parser.add_argument("--server-csv", "--server_csv", type=Path, default=here / "async_server_stats.csv")
    parser.add_argument("--sync-csv", "--sync_csv", type=Path, default=here.parent / "04_tool_overhead" / "timing_stats.csv")
    parser.add_argument("--sync-tag", "--sync_tag", default="A_1")
    parser.add_argument("--outdir", type=Path, default=here)
    parser.add_argument("--skip-warmup", "--skip_warmup", type=int, default=2)
    args = parser.parse_args()

    _setup_fonts()
    # 保持原始采集顺序先对齐 server 日志；对齐完成后再按 hold 排序用于绘图。
    client_raw = pd.read_csv(args.client_csv).reset_index(drop=True)
    # server CSV 可能无 header
    with open(args.server_csv, newline="") as f:
        first_line = f.readline().strip()
    has_header = not first_line[0].isdigit() if first_line else True
    if has_header:
        server = pd.read_csv(args.server_csv)
    else:
        server = pd.read_csv(args.server_csv, header=None, names=["episode_idx", "timestep", "t_infer_s"])
    sync_fps = load_sync_fps(args.sync_csv, args.sync_tag)

    client_raw["eff_fps"] = client_raw["action_frames"] / client_raw["episode_total_s"]
    tinfer_raw = split_server_tinfer(server, client_raw, args.skip_warmup)

    order = np.argsort(client_raw["hold"].to_numpy(dtype=float))
    client = client_raw.iloc[order].reset_index(drop=True)
    tinfer_data = [tinfer_raw[i] for i in order]
    labels = [f"{h:.2f}" for h in client["hold"]]

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(client, tinfer_data, sync_fps)
    out_csv = args.outdir / "async_hold_summary.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"saved {out_csv}")

    plot_fps(client, labels, sync_fps, args.outdir / "chart1_fps_vs_sync.png")
    plot_must_go(client, labels, args.outdir / "chart2_must_go.png")
    plot_frame_breakdown(client, labels, args.outdir / "chart3_frame_breakdown.png")
    plot_tinfer(tinfer_data, labels, args.outdir / "chart4_tinfer_boxplot.png")


if __name__ == "__main__":
    main()
