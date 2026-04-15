#!/usr/bin/env python

# 版权所有 2024 Tony Z. Zhao 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（"许可证"）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件
# 按"原样"提供，不附带任何明示或暗示的保证或条件。
# 有关许可证下特定语言的权限和限制，请参阅许可证。

"""ACT 策略的预处理器和后处理器模块

本模块负责：
    1. 预处理（Preprocessor）：将原始环境观测转换为模型输入格式
        - 重命名观测键（兼容性）
        - 添加批次维度
        - 移动到正确设备（CPU/GPU）
        - 归一化数值特征

    2. 后处理（Postprocessor）：将模型输出转换为环境可执行格式
        - 反归一化到原始尺度
        - 移动到 CPU（环境通常在 CPU 上运行）

数据流：
    环境观测 (raw)
         │
         ▼
    ┌─────────────────────────────────────┐
    │       Preprocessor Pipeline          │
    │  1. RenameObservationsProcessorStep │
    │  2. AddBatchDimensionProcessorStep  │
    │  3. DeviceProcessorStep             │
    │  4. NormalizerProcessorStep         │
    └────────┬────────────────────────────┘
         │
         ▼
    模型输入 (normalized, on device)
         │
         ▼
    ┌─────────────────────────────────────┐
    │         ACT Policy (推理)            │
    └────────┬────────────────────────────┘
         │
         ▼
    模型输出 (normalized actions)
         │
         ▼
    ┌─────────────────────────────────────┐
    │       Postprocessor Pipeline         │
    │  1. UnnormalizerProcessorStep        │
    │  2. DeviceProcessorStep (to CPU)    │
    └────────┬────────────────────────────┘
         │
         ▼
    环境动作 (raw, on CPU)
"""

from typing import Any

import torch

from lerobot.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建 ACT 策略的预处理和后处理流水线。

    预处理流水线功能：
        1. 重命名观测键（如 observation.images.top → observation.images）
           兼容性处理，允许不同数据集使用不同的命名约定
        2. 添加批次维度
           环境可能返回 (C,H,W)，但模型期望 (B,C,H,W)
        3. 移动到正确设备
           将数据从 CPU 移动到 GPU（如果可用）
        4. 归一化数值特征
           使用数据集统计信息（均值、标准差）进行标准化

    后处理流水线功能：
        1. 反归一化到原始尺度
           将模型输出的归一化动作转回原始物理单位（如关节角度）
        2. 移动到 CPU
           PyTorch 模型通常在 GPU 上，但环境在 CPU 上运行

    Args:
        config: ACTConfig 实例
            包含所有预处理/后处理所需的配置信息：
            - config.input_features: 输入特征配置
            - config.output_features: 输出特征配置
            - config.normalization_mapping: 归一化模式映射
            - config.device: 目标设备

        dataset_stats: 数据集统计信息字典（可选）
            结构：{feature_name: {"mean": Tensor, "std": Tensor, ...}}
            用于归一化和反归一化
            如果为 None，则使用配置中的默认值（通常为 0,1 归一化）

    Returns:
        tuple: 包含两个流水线的元组
            - 第一个：预处理器流水线，处理环境观测
            - 第二个：后处理器流水线，处理策略输出

    示例用法：
        >>> config = ACTConfig(...)
        >>> dataset_stats = load_dataset_stats("path/to/stats")
        >>> preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats)
        >>>
        >>> # 推理时
        >>> raw_obs = env.reset()
        >>> processed_obs = preprocessor(raw_obs)
        >>> action = policy.select_action(processed_obs)
        >>> raw_action = postprocessor(action)
        >>> env.step(raw_action)
    """

    # ── 预处理器流水线 ──────────────────────────────────────────────

    # 步骤 1：重命名观测键
    # rename_map: 空字典表示不重命名任何键
    # 这个步骤主要用于数据集兼容性处理
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
    ]

    # 步骤 2：添加批次维度
    # 环境返回的观测通常是 (C,H,W) 或 (state_dim,)
    # 模型期望 (B,C,H,W) 或 (B,state_dim)
    # 这个步骤自动添加 batch 维度
    input_steps.append(AddBatchDimensionProcessorStep())

    # 步骤 3：移动到正确设备
    # 将数据从 CPU 移动到 config.device（通常是 "cuda" 或 "mps"）
    input_steps.append(DeviceProcessorStep(device=config.device))

    # 步骤 4：归一化数值特征
    # 使用 config 中的特征配置和数据集统计信息
    # 会根据 normalization_mapping 对不同模态应用不同的归一化方法
    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            # input_features 和 output_features 合并
            # 因为训练时输出（action）也需要归一化
            norm_map=config.normalization_mapping,
            # 归一化模式映射，如 {"STATE": MEAN_STD, "ACTION": MEAN_STD, ...}
            stats=dataset_stats,
            # 数据集统计信息，包含各特征的均值和标准差
            device=config.device,
        )
    )

    # ── 后处理器流水线 ──────────────────────────────────────────────

    # 步骤 1：反归一化
    # 将归一化的动作值转回原始物理尺度
    # 例如：归一化的 0.5 → 实际的关节角度 45°
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    ]

    # 步骤 2：移动到 CPU
    # 环境通常在 CPU 上运行
    # 将动作张量从 GPU/MPS 移回 CPU
    output_steps.append(DeviceProcessorStep(device="cpu"))

    # ── 构建并返回流水线 ─────────────────────────────────────────────

    # PolicyProcessorPipeline 封装了一个处理步骤序列
    # 泛型参数：[输入类型, 输出类型]

    # 预处理器：
    #   输入：原始环境观测 (dict[str, Any])
    #   输出：模型输入 (dict[str, Any])
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    # 后处理器：
    #   输入：PolicyAction（策略输出的动作对象）
    #   输出：PolicyAction（环境可执行的动作对象）
    # to_transition: 将 PolicyAction 转换为环境 transition 格式
    # to_output: 将 transition 转回 PolicyAction 格式
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    return (preprocessor, postprocessor)
