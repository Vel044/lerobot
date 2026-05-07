#!/usr/bin/env python3
"""
实验 06：ACT 推理管线逐模块耗时剖析 - 数据分析与画图

输入：inference_timing.csv（由 ACT 模型在 LEROBOT_PROFILING=1 时直接写入）
输出：
    - inference_summary.csv   各模块均值/标准差/占比
    - inference_breakdown.png 水平柱状图（逐模块耗时 + 占比）

CSV 格式：run_id,path_type,module,time_ms

用法（在树莓派 5 上跑完后，Mac 上分析）：
    python analyze.py --csv inference_timing.csv --outdir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DISPLAY_MODULES = [
    "full",
    "resnet_backbone",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
    "fast",
]

MODULE_LABEL = {
    "full": "完整推理 (full)",
    "resnet_backbone": "ResNet18 + 投影",
    "transformer_encoder": "Transformer Encoder",
    "transformer_decoder": "Transformer Decoder",
    "action_head": "action_head",
    "fast": "快速路径 (fast)",
}


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计算各模块的均值/标准差/占比。"""
    stats = df.groupby("module")["time_ms"].agg(["mean", "std", "count"])
    stats = stats.rename(columns={"mean": "mean_ms", "std": "std_ms", "count": "n"})

    full = df[df.path_type == "full"]
    if len(full) > 0:
        total_mean = full[full.module == "full"]["time_ms"].mean()
        stats["pct"] = stats["mean_ms"] / total_mean * 100

    fast = df[df.path_type == "fast"]
    if len(fast) > 0:
        print(f"\n=== fast path 对照 ===")
        print(f"fast_path: mean={stats.loc['fast', 'mean_ms']:.3f} ms, std={stats.loc['fast', 'std_ms']:.3f} ms")

    print(f"\n=== full path 统计 ===")
    for m in ["full", "resnet_backbone", "transformer_encoder", "transformer_decoder", "action_head"]:
        if m in stats.index:
            r = stats.loc[m]
            print(f"  {m:25s}  mean={r['mean_ms']:7.1f} ms  std={r['std_ms']:6.1f} ms  {r['pct']:5.1f}%")

    return stats


def plot_breakdown(stats: pd.DataFrame, out_png: Path) -> None:
    """水平柱状图：各模块平均耗时 + 占比标注。"""
    _setup_fonts()

    modules = [m for m in DISPLAY_MODULES if m in stats.index]
    means = [stats.loc[m, "mean_ms"] for m in modules]
    stds = [stats.loc[m, "std_ms"] for m in modules]
    pcts = [stats.loc[m, "pct"] if "pct" in stats.columns else 0 for m in modules]
    labels = [MODULE_LABEL.get(m, m) for m in modules]

    modules = modules[::-1]
    means = means[::-1]
    stds = stds[::-1]
    pcts = pcts[::-1]
    labels = labels[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "#C44E52" if m in ("full", "resnet_backbone", "transformer_encoder", "transformer_decoder", "action_head")
        else "#4C72B0"
        for m in modules
    ]
    bars = ax.barh(labels, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=3)
    for bar, mean, pct in zip(bars, means, pcts):
        ax.text(bar.get_width() + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1f} ms ({pct:.1f}%)",
                va="center", fontsize=10)

    ax.set_xlabel("平均耗时 (ms)", fontsize=12)
    ax.set_title("ACT 推理管线逐模块耗时分解", fontsize=13)
    ax.set_xlim(0, max(means) * 1.45)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C44E52", edgecolor="black", label="完整推理路径模块"),
        Patch(facecolor="#4C72B0", edgecolor="black", label="快速路径"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=Path, default=here / "inference_timing.csv")
    parser.add_argument("--outdir", type=Path, default=here)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    stats = build_summary(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "inference_summary.csv"
    stats.to_csv(out_csv, float_format="%.3f")
    print(f"saved {out_csv}")

    plot_breakdown(stats, args.outdir / "inference_breakdown.png")


if __name__ == "__main__":
    main()
