#!/usr/bin/env python3
"""
实验 06：ACT 推理管线逐模块耗时剖析

用法（在树莓派 5 上）：
    LEROBOT_PROFILING=1 python profile_act_inference.py \
        --policy-path /home/vel/so101-bottle/last/pretrained_model \
        --dataset-repo vel/so101_bottle \
        --warmup 5 --full-runs 50 --fast-runs 50 \
        --out lerobot/analysis/06_act_inference_profiling/inference_timing.csv

设计：复用 predict_action() 内的 _PROFILING 守卫计时点，
离线 replay 一帧 observation，采集 full/fast 两种路径的逐模块耗时。
"""
from __future__ import annotations

import argparse
import csv
import os
import time

# 必须在 import lerobot 之前设置环境变量
os.environ["LEROBOT_PROFILING"] = "1"

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action


def load_observation(dataset: LeRobotDataset, frame_idx: int) -> dict[str, np.ndarray]:
    """从数据集取一帧，构造 predict_action 所需的 observation dict。

    predict_action 期望的格式：
      - "observation.state": np.ndarray (state_dim,) float32
      - "observation.images.xxx": np.ndarray (H, W, C) uint8
    """
    from lerobot.datasets.utils import build_dataset_frame

    item = dataset[frame_idx]
    # item 中 observation 相关的 key 形如 "observation.state", "observation.images.handeye" 等
    # 先把 tensor 转 numpy，再按 build_dataset_frame 格式组装
    obs = {}
    for key in dataset.features:
        if not key.startswith("observation"):
            continue
        val = item[key]
        if isinstance(val, torch.Tensor):
            val = val.numpy()
        obs[key] = val
    return obs


def main(args: argparse.Namespace) -> None:
    # ── 1. 加载策略 ──────────────────────────────────────────────
    print(f"[1/4] 加载策略: {args.policy_path}")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    device = torch.device(policy.config.device)
    policy.to(device)
    policy.eval()

    # ── 2. 加载数据集，取一帧 observation ────────────────────────
    print(f"[2/4] 加载数据集: {args.dataset_repo}")
    dataset = LeRobotDataset(repo_id=args.dataset_repo)
    # 取第一个 episode 第一帧作为固定输入
    frame_idx = 0
    obs = load_observation(dataset, frame_idx)
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # ── 3. 构建 preprocessor / postprocessor ─────────────────────
    print("[3/4] 构建预处理器/后处理器")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
    )

    # ── 4. 注入全局计时列表 ──────────────────────────────────────
    import lerobot.utils.control_utils as cu_mod
    import lerobot.policies.act.modeling_act as ma_mod

    records: list[tuple] = []
    cu_mod._profiling_records = records
    ma_mod._profiling_records = records

    # ── 5. 采集循环 ──────────────────────────────────────────────
    print(f"[4/4] 采集: warmup={args.warmup}, full_runs={args.full_runs}, fast_runs={args.fast_runs}")
    rows: list[tuple] = []

    # 5a. full path: 队列为空时的完整推理
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    for i in range(args.warmup + args.full_runs):
        # 确保 action_queue 为空（每轮都重新 reset → queue 清空 → 走 full path）
        policy._action_queue.clear()
        records.clear()

        _ = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
        )

        if i < args.warmup:
            continue  # 预热帧不记录

        run_id = i - args.warmup + 1
        for module, t_ms in records:
            rows.append((run_id, "full", module, f"{t_ms:.3f}"))

    # 5b. fast path: 队列非空时只取缓存动作
    # 做一次完整推理填充队列，然后连续调用 popleft
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    # 先做一次完整推理让队列里有动作
    _ = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
    )
    # warmup
    for _ in range(args.warmup):
        records.clear()
        if len(policy._action_queue) == 0:
            # 队列空了，再填充
            _ = predict_action(obs, policy, device, preprocessor, postprocessor, policy.config.use_amp)
        _ = predict_action(obs, policy, device, preprocessor, postprocessor, policy.config.use_amp)

    # 正式采集 fast path
    for i in range(args.fast_runs):
        # 确保队列非空；如果空了就先做一次完整推理（不计时）
        if len(policy._action_queue) == 0:
            _ = predict_action(obs, policy, device, preprocessor, postprocessor, policy.config.use_amp)

        records.clear()
        _ = predict_action(obs, policy, device, preprocessor, postprocessor, policy.config.use_amp)

        # 验证走的是 fast path
        for module, t_ms in records:
            path_type = "fast" if module == "fast_path" else "full"
            rows.append((i + 1, path_type, module, f"{t_ms:.3f}"))

    # ── 6. 写 CSV ────────────────────────────────────────────────
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "path_type", "module", "time_ms"])
        w.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {out_path}")

    # ── 7. 速览统计 ──────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(out_path)
    full = df[df.path_type == "full"]
    if len(full) > 0:
        print("\n=== full path 逐模块统计 ===")
        stats = full.groupby("module")["time_ms"].agg(["mean", "std", "count"])
        total_mean = full.groupby("run_id")["time_ms"].sum().mean()
        stats["pct"] = stats["mean"] / total_mean * 100
        stats = stats.sort_values("mean", ascending=False)
        print(stats.to_string(float_format="%.2f"))
        print(f"\nfull path 总耗时均值: {total_mean:.1f} ms")

    fast = df[df.path_type == "fast"]
    if len(fast) > 0:
        print(f"\n=== fast path ===")
        print(f"mean: {fast.time_ms.mean():.3f} ms, std: {fast.time_ms.std():.3f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ACT 推理管线逐模块耗时剖析")
    p.add_argument("--policy-path", required=True, help="ACT checkpoint 路径")
    p.add_argument("--dataset-repo", required=True, help="LeRobot dataset repo ID")
    p.add_argument("--warmup", type=int, default=5, help="预热次数（不计入统计）")
    p.add_argument("--full-runs", type=int, default=50, help="完整推理采集次数")
    p.add_argument("--fast-runs", type=int, default=50, help="快速路径采集次数")
    p.add_argument("--out", required=True, help="输出 CSV 路径")
    main(p.parse_args())
