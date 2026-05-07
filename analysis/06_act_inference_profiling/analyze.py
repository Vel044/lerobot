#!/usr/bin/env python3
"""
实验 06：ACT 推理管线逐模块耗时剖析 - 数据分析与画图

输入：inference_timing.csv（由 profile_act_inference.py 生成）
输出：
    - inference_summary.csv   各模块均值/标准差/占比
    - inference_breakdown.png 水平柱状图（逐模块耗时 + 占比）

用法：
    python analyze.py --csv inference_timing.csv --outdir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ── full path 顶层模块（按 predict_action 内执行顺序）─────────────
# 子模块（resnet_backbone 等）在图中单独展示，不计入顶层柱
TOP_MODULES = [
    "input_prepare",
    "preprocessor",
    "select_action",
    "postprocess",
]

# select_action 内部的子模块（由 modeling_act.py 的 forward 计时产出）
SUB_MODULES = [
    "resnet_backbone",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
    "queue_extend",
    "predict_action_chunk",  # 仅用于交叉验证，不单独画柱
]

# 图中展示的所有模块（从上到下）
DISPLAY_MODULES = [
    "input_prepare",
    "preprocessor",
    "resnet_backbone",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
    "queue_extend",
    "postprocess",
]

MODULE_LABEL = {
    "input_prepare": "输入整理\n(numpy→tensor)",
    "preprocessor": "preprocessor\n(mean-std 归一化)",
    "resnet_backbone": "ResNet18 + 投影",
    "transformer_encoder": "Transformer Encoder",
    "transformer_decoder": "Transformer Decoder",
    "action_head": "action_head",
    "queue_extend": "队列 extend/popleft",
    "postprocess": "后处理\n(反归一化+squeeze+cpu)",
    "predict_action_chunk": "predict_action_chunk",
    "select_action": "select_action",
}


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计算各模块的均值/标准差/占比，输出论文表格。"""
    full = df[df.path_type == "full"].copy()
    fast = df[df.path_type == "fast"].copy()

    # 按 (run_id, module) 去重取均值（每个 run_id 每个模块只应出现一次）
    per_run = full.groupby(["run_id", "module"])["time_ms"].first().reset_index()

    # 计算每次完整推理的总时间（用顶层模块求和）
    top_per_run = per_run[per_run["module"].isin(TOP_MODULES)]
    totals = top_per_run.groupby("run_id")["time_ms"].sum()
    total_mean = totals.mean()

    # 各模块统计
    stats = per_run.groupby("module")["time_ms"].agg(["mean", "std", "count"])
    stats["pct"] = stats["mean"] / total_mean * 100
    stats = stats.rename(columns={"mean": "mean_ms", "std": "std_ms", "count": "n"})

    # fast path 统计
    if len(fast) > 0:
        fast_stats = fast.groupby("module")["time_ms"].agg(["mean", "std"])
        print(f"\n=== fast path 对照 ===")
        print(f"fast_path 总耗时: mean={fast_stats.loc['fast_path', 'mean']:.3f} ms, "
              f"std={fast_stats.loc['fast_path', 'std']:.3f} ms")

    # 交叉验证：select_action ≈ sum(sub_modules)
    if "select_action" in stats.index:
        sa_mean = stats.loc["select_action", "mean_ms"]
        sub_sum = stats.loc[stats.index.isin(SUB_MODULES), "mean_ms"].sum()
        print(f"\n[交叉验证] select_action = {sa_mean:.1f} ms, "
              f"子模块之和 = {sub_sum:.1f} ms, "
              f"差值 = {sa_mean - sub_sum:.1f} ms")

    return stats


def plot_breakdown(stats: pd.DataFrame, fast_mean: float | None, out_png: Path) -> None:
    """水平柱状图：各模块平均耗时 + 占比标注。"""
    _setup_fonts()

    # 只画 DISPLAY_MODULES 中存在的模块
    modules = [m for m in DISPLAY_MODULES if m in stats.index]
    means = [stats.loc[m, "mean_ms"] for m in modules]
    stds = [stats.loc[m, "std_ms"] for m in modules]
    pcts = [stats.loc[m, "pct"] for m in modules]
    labels = [MODULE_LABEL.get(m, m) for m in modules]

    # 反转使第一个模块在最上方
    modules = modules[::-1]
    means = means[::-1]
    stds = stds[::-1]
    pcts = pcts[::-1]
    labels = labels[::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [
        "#4C72B0" if m in ("input_prepare", "preprocessor", "postprocess", "queue_extend")
        else "#C44E52"
        for m in modules
    ]
    bars = ax.barh(labels, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=3)
    for bar, mean, pct in zip(bars, means, pcts):
        ax.text(bar.get_width() + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1f} ms ({pct:.1f}%)",
                va="center", fontsize=10)

    ax.set_xlabel("平均耗时 (ms)", fontsize=12)
    ax.set_title("ACT 完整推理逐模块耗时分解", fontsize=13)
    ax.set_xlim(0, max(means) * 1.4)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    # 图例：蓝色=非神经网络，红色=神经网络
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", edgecolor="black", label="数据转换 / 队列"),
        Patch(facecolor="#C44E52", edgecolor="black", label="神经网络计算"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    # fast path 对照注释
    if fast_mean is not None:
        ax.annotate(f"fast path 对照: {fast_mean:.3f} ms",
                    xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

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

    # 保存汇总表
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "inference_summary.csv"
    stats.to_csv(out_csv, float_format="%.3f")
    print(f"saved {out_csv}")

    # 打印论文表格格式
    print("\n=== 论文表格 ===")
    for m in DISPLAY_MODULES:
        if m in stats.index:
            row = stats.loc[m]
            print(f"  {m:25s}  {row['mean_ms']:8.1f}  {row['std_ms']:7.1f}  {row['pct']:5.1f}%")

    # fast path 均值
    fast = df[df.path_type == "fast"]
    fast_mean = fast["time_ms"].mean() if len(fast) > 0 else None

    plot_breakdown(stats, fast_mean, args.outdir / "inference_breakdown.png")


if __name__ == "__main__":
    main()
