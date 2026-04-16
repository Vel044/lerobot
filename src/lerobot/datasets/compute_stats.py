#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from lerobot.datasets.utils import load_image_as_numpy


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """根据数据集大小估算采样数量（用于计算归一化统计量时采样帧数）。
    power 控制采样数随数据集大小的增长速率，默认 0.75 次方。

    默认参数下的典型值：
    - 1 ~ 500 帧：采 100 帧
    - 1000 帧 → 177，2000 帧 → 299，10000 帧 → 1000，20000 帧 → 1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    """均匀采样索引：在 [0, data_len-1] 之间均匀取 estimate_num_samples 个点。"""
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    """自动降采样图像：短边大于 max_size_threshold 时，按比例缩小到 target_size 左右。
    降采样是为了减少计算统计量时的内存消耗，不影响最终 std/mean 的精度。
    """
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # 小图不需要降采样
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    """从图像路径列表中均匀采样部分图像，返回 (N, C, h, w) uint8 数组。"""
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # uint8 加载节省内存；channel_first=True → (C, H, W)
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        # 降采样减少内存
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    """对单个特征的一个数据块计算五项统计量。
    返回 {"min", "max", "mean", "std", "count"}，沿指定 axis 归约。
    - state/action: axis=0, keepdims=按原ndim → 输出 shape=(dim,)
    - image: axis=(0,2,3), keepdims=True → 输出 shape=(C,1,1) 只保留通道维
    """
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),   # ← std 的诞生地
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    """计算单个 episode 内所有特征的统计量。
    episode_data 结构：
      - image/video 字段 → list[str]（图像文件路径列表）
      - 数值字段 → np.ndarray，shape=(T, dim)
    """
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            # 图像：采样部分帧，降采样后堆叠成 (N, C, h, w)
            ep_ft_array = sample_images(data)
            # 沿 batch/height/width 三轴归约，只保留通道维 → mean/std shape=(C,1,1)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            # state/action：直接用 ndarray，沿时间轴归约 → mean/std shape=(dim,)
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1  # 1D 向量保持 shape，2D+ 降维

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # 图像统计量：uint8 [0,255] → 归一化到 [0,1] 再计算，所以要先 /255
        # squeeze 去掉 batch 维：(1, C, 1, 1) → (C, 1, 1)
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """将多个 episode 的统计量聚合为一个全局统计量（用于合并多个 episode 的 mean/std）。
    使用并行方差算法（parallel variance algorithm），等价于把所有原始数据拼起来重算，
    但只需要每个 episode 的 mean/std/count，不需要原始数据。

    算法：
      total_mean = Σ(mean_i × count_i) / Σ(count_i)
      total_var  = Σ((var_i + (mean_i - total_mean)²) × count_i) / Σ(count_i)
      total_std  = √(total_var)
    """
    means = np.stack([s["mean"] for s in stats_ft_list])      # 各 episode 均值堆叠
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])  # std² → 方差
    counts = np.stack([s["count"] for s in stats_ft_list])    # 各 episode 样本数
    total_count = counts.sum(axis=0)

    # counts 广播对齐维度：counts (N,1) → means (N, dim) 的维度
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # 加权均值：大 episode（样本多）权重大
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # 并行方差：不能用简单平均 std，必须考虑各 episode 均值与全局均值的偏差
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),   # ← 全局 std，最终写入 checkpoint
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """将多组统计量（通常来自多个 episode 或多个 dataset）聚合为一组全局统计量。
    对每个特征 key 独立调用 aggregate_feature_stats 做并行方差合并。
    最终结果会保存到 checkpoint，推理时通过 load_state_dict 加载到 NormalizerProcessorStep。

    合并规则：
    - min = 所有组的最小值
    - max = 所有组的最大值
    - mean = 加权均值（按样本数加权）
    - std = 并行方差算法算出的全局标准差
    """
    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats
