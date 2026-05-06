#!/usr/bin/env python3
"""
实验 06：摄像头感知层级耗时分布图

输入：layer_timing.csv（profile_camera.py 产出）+ 可选 ftrace 解析后的 layer 1/2 数据
输出：水平堆叠条形 + 误差棒，按线程归属配色，标注"端到端时延"和"主线程感知耗时"两条参考线
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 线程归属配色
LAYER_META = [
    (1, "pselect6_wait",          "kernel",  "#e89441"),
    (2, "vidioc_dqbuf",           "kernel",  "#e89441"),
    (3, "libjpeg_decode",         "bg_thread", "#3b82c4"),
    (4, "ndarray_alloc",          "bg_thread", "#3b82c4"),
    (5, "postprocess_bgr2rgb",    "bg_thread", "#3b82c4"),
    (6, "async_read_lock_copy",   "main",    "#5fb56a"),
    (7, "observation_processor",  "main",    "#5fb56a"),
    (8, "normalize_processor",    "main",    "#5fb56a"),
    (9, "device_processor",       "preproc", "#9aa1ac"),
]


def load_csv(path: Path):
    """返回 layer_id -> [t_ms, ...]"""
    bucket = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            bucket[int(row["layer_id"])].append(float(row["t_ms"]))
    return bucket


def plot(csv_path: Path, out_path: Path):
    bucket = load_csv(csv_path)
    layers = LAYER_META
    means = np.array([np.mean(bucket.get(lid, [0])) for lid, *_ in layers])
    stds  = np.array([np.std(bucket.get(lid, [0])) for lid, *_ in layers])
    names = [name for _, name, *_ in layers]
    colors = [c for *_, c in layers]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y_pos = np.arange(len(layers))
    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"L{lid}: {n}" for (lid, n, *_), _ in zip(layers, layers)])
    ax.invert_yaxis()
    ax.set_xlabel("耗时 (ms)")
    ax.set_title("摄像头感知 9 层耗时分布（n=3 ep × 30 frames，640×480 @30fps，MJPG）")

    # 参考线：端到端时延 = layer 1+2+3+4+5
    e2e = sum(np.mean(bucket.get(i, [0])) for i in (1, 2, 3, 4, 5))
    main = sum(np.mean(bucket.get(i, [0])) for i in (6, 7, 8, 9))
    ax.axvline(e2e,  color="#e89441", linestyle="--", alpha=0.6,
               label=f"端到端时延 (L1+2+3+4+5) ≈ {e2e:.1f} ms")
    ax.axvline(main, color="#5fb56a", linestyle="--", alpha=0.6,
               label=f"主线程感知 (L6+7+8+9) ≈ {main:.2f} ms")
    ax.legend(loc="lower right")

    # 配色图例
    from matplotlib.patches import Patch
    legend2 = [
        Patch(color="#e89441", label="内核"),
        Patch(color="#3b82c4", label="后台线程"),
        Patch(color="#5fb56a", label="主线程"),
        Patch(color="#9aa1ac", label="预处理"),
    ]
    ax.add_artist(ax.legend(handles=legend2, loc="upper right", title="归属"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[ok] saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    plot(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
