#!/usr/bin/env python3
"""
实验 04：实验环境与测量工具校准 - 数据分析与画图

输入：lerobot/analysis/timing_stats.csv 中 timing_tag 形如 "A_n"/"B_n"/"C_n"/"D_n" 的行
输出：
    - tool_overhead_summary.csv   各组指标 + 相对 baseline 漂移
    - fps_drop.png                ① FPS 与相对 baseline 下降
    - context_switches.png        ② 上下文切换次数（log 轴）
    - phase_drift.png             ③ 阶段占比相对 A 组的漂移

(原 user/system CPU 时间面板已并入表格，删除以避免与上下文切换信息重复。)

用法：
    python lerobot/analysis/04_tool_overhead/analyze.py \
        --csv lerobot/analysis/timing_stats.csv \
        --raw-dir lerobot/analysis/04_tool_overhead/raw \
        --outdir lerobot/analysis/04_tool_overhead
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ["A", "B", "C", "D"]
GROUP_LABEL = {
    "A": "A baseline",
    "B": "B strace",
    "C": "C ftrace 延迟导出",
    "D": "D ftrace 实时导出",
}
GROUP_COLOR = {"A": "#4C72B0", "B": "#C44E52", "C": "#55A868", "D": "#DD8452"}


def load_groups(csv_path: Path) -> pd.DataFrame:
    """读取 timing_stats.csv，按 timing_tag 前缀（A/B/C/D）取出本实验数据，按组聚合。

    返回 DataFrame：index = group ('A'/'B'/'C'/'D')，列 = 各指标的均值。
    n=1 时均值即原值；n>1 时取算数平均。
    """
    df = pd.read_csv(csv_path)
    df = df[df["timing_tag"].str.match(r"^[ABCD]_\d+$", na=False)].copy()
    df["group"] = df["timing_tag"].str[0]
    metric_cols = [
        "frames", "episode_total_s", "fps_mean",
        "obs_pct", "inference_pct", "action_pct", "wait_pct",
        "user_time_s", "system_time_s", "max_rss_kb",
        "voluntary_cs", "involuntary_cs",
        "cpu_temp_start_c", "cpu_temp_end_c", "cpu_freq_mean_mhz",
    ]
    grouped = df.groupby("group")[metric_cols].mean()
    grouped["n"] = df.groupby("group").size()
    return grouped.reindex(GROUPS).dropna(how="all")


def build_summary(grouped: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """以 A 组为基准，计算各组相对漂移；附加 trace 文件大小 (MB)。"""
    base = grouped.loc["A"]
    out = grouped.copy()
    out["fps_drop_pct"] = (base["fps_mean"] - out["fps_mean"]) / base["fps_mean"] * 100
    out["system_time_ratio"] = out["system_time_s"] / base["system_time_s"]
    out["rss_ratio"] = out["max_rss_kb"] / base["max_rss_kb"]
    for ph in ("obs_pct", "inference_pct", "action_pct", "wait_pct"):
        out[f"{ph}_drift_pp"] = out[ph] - base[ph]
    out["voluntary_cs_ratio"] = out["voluntary_cs"] / base["voluntary_cs"]
    out["involuntary_cs_ratio"] = out["involuntary_cs"] / base["involuntary_cs"]

    trace_size = {"A": np.nan}  # baseline 无 trace 文件
    name_map = {"B": "strace_B_1.log", "C": "ftrace_C_1.txt", "D": "ftrace_D_1.txt"}
    for g, fn in name_map.items():
        p = raw_dir / fn
        trace_size[g] = p.stat().st_size / 1e6 if p.exists() else np.nan
    out["trace_file_mb"] = pd.Series(trace_size)
    return out


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_fps_drop(grouped: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    """① FPS 与相对 baseline 下降比例。每条柱顶部标注 fps 数值和下降百分比。"""
    _setup_fonts()
    groups = list(grouped.index)
    labels = [GROUP_LABEL[g] for g in groups]
    colors = [GROUP_COLOR[g] for g in groups]
    fps = grouped["fps_mean"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fps, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(grouped.loc["A", "fps_mean"], color="gray", ls="--", lw=1.0, label="A baseline")
    ax.set_ylabel("fps_mean", fontsize=12)
    ax.set_title("FPS 与相对 baseline 下降", fontsize=13)
    for b, g in zip(bars, groups):
        drop = summary.loc[g, "fps_drop_pct"]
        tag = "baseline" if g == "A" else f"-{drop:.1f}%"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{b.get_height():.2f}\n({tag})", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(fps) * 1.25)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def plot_context_switches(grouped: pd.DataFrame, out_png: Path) -> None:
    """② 上下文切换次数（log 轴）。展现 strace ptrace 拦截带来的几何级放大。"""
    _setup_fonts()
    groups = list(grouped.index)
    labels = [GROUP_LABEL[g] for g in groups]
    x = np.arange(len(groups))
    w = 0.38
    vol = grouped["voluntary_cs"].values
    invol = grouped["involuntary_cs"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, vol, w, label="voluntary_cs",
           color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, invol, w, label="involuntary_cs",
           color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("次数 (log scale)", fontsize=12)
    ax.set_title("上下文切换次数", fontsize=13)
    for xi, v in zip(x - w / 2, vol):
        ax.text(xi, v * 1.15, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    for xi, v in zip(x + w / 2, invol):
        ax.text(xi, v * 1.15, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def plot_phase_drift(summary: pd.DataFrame, out_png: Path) -> None:
    """③ 各阶段占比相对 A 组 baseline 的漂移 (pp)。"""
    _setup_fonts()
    groups = list(summary.index)
    labels = [GROUP_LABEL[g] for g in groups]
    x = np.arange(len(groups))
    phases = ["obs_pct", "inference_pct", "action_pct", "wait_pct"]
    phase_label = {"obs_pct": "obs", "inference_pct": "inference",
                   "action_pct": "action", "wait_pct": "wait"}
    bw = 0.2

    fig, ax = plt.subplots(figsize=(8, 5.2))
    drift_all = []
    for i, ph in enumerate(phases):
        drift = summary[f"{ph}_drift_pp"].values
        drift_all.extend(drift)
        bars = ax.bar(x + (i - 1.5) * bw, drift, bw, label=phase_label[ph])
        for b, v in zip(bars, drift):
            if abs(v) >= 1:
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.8 if v >= 0 else -0.8),
                        f"{v:+.1f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(3, color="gray", ls=":", lw=0.8, label="±3 pp 验收阈值")
    ax.axhline(-3, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("阶段占比漂移 (pp)", fontsize=12)
    ax.set_title("阶段占比相对 A 组的漂移", fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    # 给标签留出 padding，避免最低/最高柱的数字被裁切
    lo = min(drift_all + [-3]) - 4
    hi = max(drift_all + [3]) + 5
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=Path, default=here.parent / "timing_stats.csv")
    parser.add_argument("--raw-dir", type=Path, default=here / "raw")
    parser.add_argument("--outdir", type=Path, default=here)
    args = parser.parse_args()

    grouped = load_groups(args.csv)
    summary = build_summary(grouped, args.raw_dir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "tool_overhead_summary.csv"
    summary.to_csv(out_csv, float_format="%.4f")
    print(f"saved {out_csv}")

    # 终端速览：核心指标
    cols = ["n", "fps_mean", "fps_drop_pct", "user_time_s", "system_time_s",
            "system_time_ratio", "voluntary_cs", "involuntary_cs",
            "obs_pct_drift_pp", "wait_pct_drift_pp", "trace_file_mb"]
    print("\n=== 各组核心指标 (相对 A 组 baseline) ===")
    print(summary[cols].round(3).to_string())

    plot_fps_drop(grouped, summary, args.outdir / "fps_drop.png")
    plot_context_switches(grouped, args.outdir / "context_switches.png")
    plot_phase_drift(summary, args.outdir / "phase_drift.png")


if __name__ == "__main__":
    main()
