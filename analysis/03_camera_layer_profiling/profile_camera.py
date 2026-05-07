#!/usr/bin/env python3
"""
实验 03：摄像头感知层级耗时采集脚本

用法（在树莓派 5 上）：
    python profile_camera.py --episodes 3 --frames 30 \
        --out robotdoc/实验/03_摄像头感知层级耗时剖析/layer_timing.csv

设计：通过 monkey-patch 切入 OpenCVCamera 的 read / _postprocess_image / async_read
和 RobotProcessor 的预处理三件套，分别记录 9 层耗时；不修改主代码库。
对应论文第 3.1 节、实验设计文档见同目录下《实验设计.md》。
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

# 这里只放骨架；实际执行需在树莓派环境下安装 lerobot 后运行。
# Mac 开发机上仅作语法校验，不实际启动摄像头。

LAYER_NAMES = [
    (1, "pselect6_wait"),          # 内核 - ftrace 提供
    (2, "vidioc_dqbuf"),            # 内核 - ftrace 提供
    (3, "libjpeg_decode"),          # 后台线程 - perf_counter
    (4, "ndarray_alloc"),           # 后台线程 - perf_counter
    (5, "postprocess_bgr2rgb"),     # 后台线程 - perf_counter
    (6, "async_read_lock_copy"),    # 主线程 - perf_counter
    (7, "observation_processor"),   # 主线程 - perf_counter
    (8, "normalize_processor"),     # 主线程 - perf_counter
    (9, "device_processor"),        # 主线程 - perf_counter
]


def install_camera_patches(records: list[tuple]):
    """
    Monkey-patch OpenCVCamera 上的 read / _postprocess_image / async_read，
    将 layer 3/4/5/6 的耗时写入 records (episode, frame, layer_id, layer_name, t_ms)。
    """
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    orig_postprocess = OpenCVCamera._postprocess_image
    orig_async_read = OpenCVCamera.async_read

    def patched_read(self):
        # 拆分 grab() 与 retrieve() 以分离 layer 2 (ioctl) 和 layer 3+4 (decode+alloc)
        t0 = time.perf_counter_ns()
        ok = self.videocapture.grab()
        t1 = time.perf_counter_ns()
        ret, frame = self.videocapture.retrieve()
        t2 = time.perf_counter_ns()
        ep, fi = getattr(self, "_prof_ep", 0), getattr(self, "_prof_fi", 0)
        # grab 内部已含 ioctl(VIDIOC_DQBUF)，在 ftrace 中由 layer 2 直接量化；
        # 这里 (t1-t0) 主要是 ioctl 等待路径，仅作交叉验证
        records.append((ep, fi, 3, "libjpeg_decode_plus_alloc", (t2 - t1) / 1e6))
        return ret, frame

    def patched_postprocess(self, frame):
        t0 = time.perf_counter_ns()
        out = orig_postprocess(self, frame)
        ep, fi = getattr(self, "_prof_ep", 0), getattr(self, "_prof_fi", 0)
        records.append((ep, fi, 5, "postprocess_bgr2rgb", (time.perf_counter_ns() - t0) / 1e6))
        return out

    def patched_async_read(self, timeout_ms: float = 200):
        t0 = time.perf_counter_ns()
        out = orig_async_read(self, timeout_ms)
        ep, fi = getattr(self, "_prof_ep", 0), getattr(self, "_prof_fi", 0)
        records.append((ep, fi, 6, "async_read_lock_copy", (time.perf_counter_ns() - t0) / 1e6))
        return out

    OpenCVCamera.read = patched_read
    OpenCVCamera._postprocess_image = patched_postprocess
    OpenCVCamera.async_read = patched_async_read


def install_processor_patches(records: list[tuple], cam_handle):
    """
    包裹 observation/normalize/device 三个 processor 的 __call__，
    分别记录 layer 7/8/9。
    """
    from lerobot.processor import (
        observation_processor as obs_mod,
        normalize_processor as norm_mod,
        device_processor as dev_mod,
    )

    def make_wrapper(orig, layer_id, layer_name):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter_ns()
            out = orig(*args, **kwargs)
            ep, fi = getattr(cam_handle, "_prof_ep", 0), getattr(cam_handle, "_prof_fi", 0)
            records.append((ep, fi, layer_id, layer_name, (time.perf_counter_ns() - t0) / 1e6))
            return out
        return wrapped

    # 各 processor 的具体入口名以实际 API 为准；这里给出占位
    if hasattr(obs_mod, "make_observation_processor"):
        original = obs_mod.make_observation_processor
        obs_mod.make_observation_processor = lambda *a, **k: make_wrapper(
            original(*a, **k), 7, "observation_processor"
        )


def run(args):
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig

    records: list[tuple] = []
    install_camera_patches(records)

    cfg = OpenCVCameraConfig(
        index_or_path=args.cam_path,
        fps=30,
        width=640,
        height=480,
    )
    cam = OpenCVCamera(cfg)
    cam.connect()
    install_processor_patches(records, cam)

    try:
        for ep in range(args.episodes):
            cam._prof_ep = ep
            # 跳过冷启动
            for _ in range(args.warmup):
                _ = cam.async_read()
            for fi in range(args.frames):
                cam._prof_fi = fi
                _ = cam.async_read()
                # frame 间留 ~33ms 让 fps=30 自然驱动
                time.sleep(1 / 30)
    finally:
        cam.disconnect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "frame", "layer_id", "layer_name", "t_ms"])
        for row in records:
            w.writerow(row)
    print(f"[ok] wrote {len(records)} rows to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--frames", type=int, default=30, help="每 episode 计入的帧数")
    p.add_argument("--warmup", type=int, default=30, help="每 episode 冷启动跳过帧数")
    p.add_argument("--cam-path", default="/dev/video0", help="USB 摄像头设备路径")
    p.add_argument("--out", required=True, help="输出 csv 路径")
    p.add_argument("--use-ftrace", action="store_true", help="标记本次运行已配合 ftrace（对照 B 组）")
    run(p.parse_args())


if __name__ == "__main__":
    main()
