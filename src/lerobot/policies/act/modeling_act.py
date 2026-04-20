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

"""Action Chunking Transformer (ACT) 策略

本文档对应论文 Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
(https://huggingface.co/papers/2304.13705)。

核心设计思想：
  1. 将多步动作序列作为整体预测（action chunking），而非逐帧预测
  2. 使用 Transformer 编码器融合视觉-本体感知-环境状态
  3. 使用 Transformer 解码器（DETR 风格 object queries）生成未来动作序列
  4. 可选使用 VAE 变分目标来学习隐空间表示（训练时）

数据流概述（推理阶段）：
  observation.state (B, state_dim)
         │
         ▼
  ┌─────────────────────────┐
  │ encoder_latent_input_proj│ ← 全零向量（无 VAE 编码器时）
  └────────┬────────────────┘
           ▼
  ┌─────────────────────────┐
  │encoder_robot_state_input │ ← observation.state
  │_proj                    │     线性投影到 dim_model
  └────────┬────────────────┘
           ▼
  ┌─────────────────────────┐
  │ ResNet backbone (layer4) │ ← 各摄像头图像
  │ 输出: (B,512,H/32,W/32) │
  └────────┬────────────────┘
           ▼
  ┌─────────────────────────┐
  │encoder_img_feat_input   │ ← 1×1 卷积投影到 dim_model
  │_proj (Conv2d)          │
  └────────┬────────────────┘
           ▼
  ┌──────────────────────────────────────────┐
  │       Transformer Encoder                 │
  │  融合 latent + state + 图像 tokens       │
  │  输出: (enc_seq_len, B, dim_model)       │
  └────────┬─────────────────────────────────┘
           ▼
  ┌──────────────────────────────────────────┐
  │       Transformer Decoder                 │
  │  chunk_size 个零向量作为 query           │
  │  与 encoder_out 做交叉注意力              │
  │  输出: (chunk_size, B, dim_model)        │
  └────────┬─────────────────────────────────┘
           ▼
  ┌──────────────────────────────────────────┐
  │         action_head (Linear)             │
  │  映射到 action_dim 维动作空间            │
  │  输出: (B, chunk_size, action_dim)       │
  └──────────────────────────────────────────┘
"""

import math
# deque: 双端队列，用于实现 action 队列（FIFO）
# deque([], maxlen=n): 固定长度队列，填满后自动丢弃最旧的元素
from collections import deque
# Callable: 类型提示，用于激活函数类型注解
from collections.abc import Callable
# chain: 将多个迭代器扁平化连接，用于参数初始化时遍历所有层
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
# IntermediateLayerGetter: 提取 ResNet 中间层输出的工具
from torchvision.models._utils import IntermediateLayerGetter
# FrozenBatchNorm2d: 冻结的 BatchNorm2d（推理时不更新统计量，训练更快）
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer 策略实现。

    对应论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    论文链接：https://huggingface.co/papers/2304.13705
    原始代码：https://github.com/tonyzhaozh/act

    核心功能：
      1. 接收环境观测（图像 + 关节状态），输出未来 chunk_size 步的动作序列
      2. 支持 action chunking：一次预测多步，只执行其中 n_action_steps 步
      3. 可选时序集成（temporal ensembling）：对历史预测做指数加权平滑

    与环境的交互流程：
      reset() → select_action() → 环境执行 → select_action() → ...
    """

    # 配置类，用于从配置文件加载超参数
    config_class = ACTConfig
    # 策略名称，用于注册和识别
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """初始化 ACT 策略。

        Args:
            config: ACTConfig 实例，包含所有超参数：
                - chunk_size: 每次预测的动作步数（默认 100）
                - n_action_steps: 实际执行的步数（<= chunk_size，默认 100）
                - image_features: 要使用的摄像头 key 列表
                - use_vae: 是否使用 VAE 变分目标（默认 True）
                - temporal_ensemble_coeff: 时序集成系数（None=不启用）
                - vision_backbone: ResNet 骨架网络名称
                - dim_model: Transformer 隐藏维度（默认 512）
                - n_heads: 多头注意力头数（默认 8）
                - 等等...

        初始化流程：
            1. 调用父类初始化
            2. 验证 config 中的特征配置（至少需要图像或环境状态之一）
            3. 创建 ACT 神经网络模型
            4. 可选：创建时序集成器（当 temporal_ensemble_coeff 非 None 时）
            5. 重置内部状态（队列等）
        """
        super().__init__(config)
        config.validate_features()  # 验证至少有一个图像或环境状态输入
        self.config = config

        # 创建 ACT 神经网络（编码器-解码器 Transformer）
        self.model = ACT(config)

        # 可选：创建时序集成器（用于平滑动作预测）
        # temporal_ensemble_coeff 非 None 时启用
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        # 初始化内部状态（队列等）
        self.reset()

    def get_optim_params(self) -> dict:
        """返回优化器参数分组。

        为什么要分组？
            - backbone（ResNet 图像编码器）使用较小的学习率（lr_backbone）
            - 其他参数（Transformer 编码器/解码器等）使用默认学习率
        这样设计是因为预训练的 ResNet 已经提取了好的图像特征，不宜学习率过大。

        返回结构：
            [
                {
                    "params": [p for n,p in ... if not n.startswith("model.backbone")],
                    # 非 backbone 参数使用默认学习率（从 config.optimizer_lr 获取）
                },
                {
                    "params": [p for n,p in ... if n.startswith("model.backbone")],
                    "lr": self.config.optimizer_lr_backbone,
                    # backbone 参数使用独立的学习率（通常较小）
                },
            ]

        Returns:
            dict: 优化器参数字典，可直接传给 torch.optim.AdamW
        """
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """每当环境重置时调用此方法。

        功能：
            1. 如果启用了时序集成，重置集成器状态
            2. 否则，清空 action 队列

        重要：环境 reset 后必须调用此方法，否则队列状态会残留，
              导致新 episode 的动作从旧队列中取出而非重新推理

        内部状态说明：
            - _action_queue: deque 类型，最大长度 n_action_steps
              用于缓存已预测但尚未执行的动作
        """
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            # deque([], maxlen=n): 固定长度队列，填满后自动丢弃最旧的元素
            # maxlen=self.config.n_action_steps: 队列最多缓存 n_action_steps 个动作
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测选择单个动作（与环境交互的核心接口）。

        本方法是 select_actions 的包装，负责：
            1. 管理 action chunk 队列
            2. 控制推理频率（减少推理次数）

        Args:
            batch: 来自环境的一帧观测，字典结构：
                - "observation.state": (B, state_dim)  # 关节角度等本体感知
                  例如：14 维关节位置/角度
                - "observation.images.{camera_key}": (B, C, H, W)  # 摄像头图像
                  camera_key 由 config.image_features 指定，如 "observation.images.top"

        Returns:
            action: (B, action_dim)  # 当前时刻要执行的单个动作向量
                例如：14 维关节目标位置

        两种推理路径：

        路径 A - 时序集成（temporal_ensemble_coeff 非 None）：
            每步都调用 predict_action_chunk，将新预测与历史做指数加权平均
            优点：动作更平滑；缺点：每步都要推理（n_action_steps 必须为 1）

        路径 B - Action Chunk 队列（默认）：
            队列为空时调用 predict_action_chunk，填充队列
            每次只从队列取出 n_action_steps 步执行
            优点：减少推理频率；缺点：动作可能有"跳跃"

        示例（chunk_size=100, n_action_steps=50）：
            t=0: 推理 → 队列=[a0,a1,...,a49]，返回 a0
            t=1-49: 直接从队列取值，无需推理
            t=50: 队列空 → 再次推理 → 队列=[a50,a51,...,a99]，返回 a50
            t=51-99: 直接从队列取值
            t=100: 队列空 → 循环
        """
        # 强制进入 eval 模式，关闭 dropout/batchnorm 的训练行为
        # 必要：队列消费期间外部代码可能把模型切回 train 模式
        self.eval()

        # ── 路径 A：启用时序集成 ─────────────────────────────────────────────
        if self.config.temporal_ensemble_coeff is not None:
            # predict_action_chunk: (B, chunk_size, action_dim)
            # 完整预测未来 chunk_size 步
            actions = self.predict_action_chunk(batch)
            # temporal_ensembler.update: 对历史做指数加权，返回当前动作
            action = self.temporal_ensembler.update(actions)
            return action

        # ── 路径 B：action chunk 队列（默认） ────────────────────────────────
        # 当队列为空时，用当前观测查询策略，填充队列
        if len(self._action_queue) == 0:
            # predict_action_chunk 返回 (B, chunk_size, action_dim)
            # [:, :n_action_steps] 取前 n_action_steps 步，其余丢弃
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # actions: (B, n_action_steps, action_dim)
            # transpose(0,1): (n_action_steps, B, action_dim)
            # extend 后队列里有 n_action_steps 个 (B, action_dim) 张量
            self._action_queue.extend(actions.transpose(0, 1))

        # 从队列头部弹出当前动作：(B, action_dim)
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """预测完整的动作块（action chunk）。

        这是实际调用神经网络进行推理的方法。

        Args:
            batch: 环境观测字典，至少包含：
                - "observation.state": (B, state_dim)  # 关节状态
                - 各摄像头图像 key（如 "observation.images.top"）: (B, C, H, W)

        Returns:
            actions: (B, chunk_size, action_dim)  # 未来 chunk_size 步的预测动作序列

        处理流程：
            1. 如果使用多摄像头，将各路图像收集到 "observation.images" 列表
            2. 调用 self.model(batch) 进行前向传播
            3. 返回动作序列

        推理 vs 训练：
            推理时：latent = 全零向量，VAE 编码器被跳过
            训练时：latent = VAE 编码器从动作序列中采样
        """
        self.eval()

        # 多摄像头处理：收集到统一列表
        if self.config.image_features:
            # 浅拷贝，避免修改原始 batch 字典
            batch = dict(batch)
            # 之前：batch["observation.images.handeye"]=Tensor(1,3,360,640), batch["observation.images.fixed"]=Tensor(1,3,360,640)（两个独立键）
            # 之后：batch["observation.images"]=[Tensor(1,3,360,640), Tensor(1,3,360,640)]（按 config.image_features 顺序合并为一个列表）
            # 变成[tensor, tensor]格式的列表
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 调用 ACT 模型
        # model 返回 (actions, (mu, log_sigma_x2))
        #   actions: (B, chunk_size, action_dim)
        #   mu, log_sigma_x2: 推理时均为 None（只有训练+VAE 时才有意义）
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """训练/验证时的前向传播。

        Args:
            batch: 训练数据批次，包含：
                - "observation.state": (B, state_dim)  # 关节状态
                - "observation.images": [(B,C,H,W), ...]  # 图像列表
                - ACTION ("action"): (B, chunk_size, action_dim)  # 目标动作序列
                - "action_is_pad": (B, chunk_size)  # padding 掩码（True=填充）

        Returns:
            loss: 标量损失张量（用于反向传播）
            loss_dict: 字典，包含各损失分量的值：
                - "l1_loss": L1 重构损失（必有）
                - "kld_loss": KL 散度（仅 use_vae=True 时）

        损失计算：
            1. L1 损失：|action - pred_action|，排除 padding 位置
            2. KL 损失（可选）：KL(prior || posterior) = -0.5*(1+log(σ²)-μ²-σ²)
               用于正则化 latent 空间，使隐变量接近标准正态分布
            3. 总损失 = l1_loss + kl_weight * kld_loss
               kl_weight 默认 10.0，用于平衡两个目标的相对重要性
        """
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 前向传播
        # actions_hat: (B, chunk_size, action_dim) 预测动作
        # mu_hat, log_sigma_x2_hat: VAE 分布参数（训练时）
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # 计算 L1 损失，排除 padding 位置
        # batch[ACTION]: (B, chunk_size, action_dim) 目标动作
        # batch["action_is_pad"]: (B, chunk_size) True=填充位置
        # ~batch["action_is_pad"]: 取反，True 表示有效位置
        # unsqueeze(-1): (B, chunk_size, 1) 以便广播到 action_dim 维度
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        if self.config.use_vae:
            # 计算 KL 散度：D_KL(latent_pdf || standard_normal)
            # 公式：-0.5 * (1 + log(σ²) - μ² - σ²)
            # 步骤：
            #   1. 对每个 latent 维度计算上述公式
            #   2. sum(-1) 对 latent_dim 维度求和，得到每个 batch 元素的 KL
            #   3. mean() 对 batch 求平均
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp()))
                .sum(-1)  # 对 latent_dim 维度求和
                .mean()   # 对 batch 求平均
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    """时序集成器：在线计算动作序列的指数加权平均。

    论文 Algorithm 2 对应实现。
    核心思想：每一步都用新预测的 chunk 更新历史均值，权重随时间指数衰减。

    为什么需要时序集成？
        - ACT 一次预测 chunk_size 步，但环境是动态变化的
        - 随着时间推移，早期预测可能与实际环境状态不再匹配
        - 时序集成通过指数加权平滑，让近期预测权重更高，同时保留历史稳定性

    权重计算：
        w_i = exp(-temporal_ensemble_coeff * i)
        其中 i=0 对应最旧的预测（权重最大），i 越大越新

    衰减系数的影响（论文推荐 0.01）：
        - 0: 所有动作均匀权重
        - 正值(0.01): 越旧的动作权重越高（动作更平滑保守）
        - 负值: 越新的动作权重越高（动作更激进）

    在线更新公式：
        new_avg = (old_avg * Σw_old + new_val * w_new) / Σw_new
        高效实现：避免存储完整历史，每次只维护当前均值和计数

    Attributes:
        chunk_size: 动作块长度（每次预测的步数）
        ensemble_weights: (chunk_size,) 指数衰减权重 [1.0, exp(-c), exp(-2c), ...]
        ensemble_weights_cumsum: (chunk_size,) 权重的累积和 [w0, w0+w1, w0+w1+w2, ...]
        ensembled_actions: 维护的在线加权平均状态，形状 (B, chunk_size, action_dim)
        ensembled_actions_count: 每个时间步被更新过的次数，形状 (chunk_size, 1)
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """初始化时序集成器。

        Args:
            temporal_ensemble_coeff: 指数衰减系数，默认值 0.01（论文推荐）
                - 控制新旧动作的权重分配
                - 越大越保守（更重视历史），越小越激进（更重视新预测）
            chunk_size: 每次预测的动作块长度

        权重初始化示例（coeff=0.01, chunk_size=10）：
            weights = [1.0, 0.990, 0.980, 0.970, 0.960, ...]
            cumsum  = [1.0, 1.990, 2.970, 3.940, 4.900, ...]
        """
        self.chunk_size = chunk_size
        # torch.arange(chunk_size): [0, 1, 2, ..., chunk_size-1]
        # exp(-coeff * i): 指数衰减权重，i=0 时权重最大为 1.0
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size)
        )
        # 累积和，用于在线更新时快速计算 Σw
        # 例如: cumsum[3] = weights[0] + weights[1] + weights[2] + weights[3]
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """重置在线计算状态。

        在 episode 开始时调用，因为时序集成是跨步累积的，
        不同 episode 需要分开计算（不能把新 episode 的动作
        和旧 episode 的历史混在一起）。

        重置后：
            - ensembled_actions = None
            - ensembled_actions_count = None
        """
        self.ensembled_actions = None  # 历史动作的在线均值
        # 每个时间步被更新过的次数（用于计算累积权重）
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """更新时序集成状态并返回当前时刻的动作。

        这是核心方法，每一帧推理后调用一次。

        Args:
            actions: (B, chunk_size, action_dim)
                本次推理预测的完整动作序列
                B=batch_size, chunk_size=动作块长度, action_dim=关节维度

        Returns:
            action: (B, action_dim)
                当前时刻要执行的加权融合动作

        工作流程：

        第一步（ensembled_actions is None）：
            1. 用当前预测 actions 初始化 ensembled_actions
            2. ensembled_actions_count 初始化为全 1（每步被覆盖 1 次）
            3. 返回 actions[:, 0]（即 a0）

        后续步（ensembled_actions 已存在）：
            1. 在线更新公式：
               先把旧的均值乘以累积权重（还原到加权和状态）
               加上新预测 × 对应权重
               再除以新的累积权重（恢复到均值状态）
            2. 将新 chunk 的最后一步（无历史）拼接
            3. 返回 ensembled_actions[:, 0]（t=0 的融合动作）

        在线更新公式详解：
            假设 t=0 时刻的动作为 a0，已知其在线均值为 μ0
            t=1 时刻新预测 b1，要更新 μ1：
                历史加权和 = μ0 * w0（w0=1，因为 count=1 时 cumsum[0]=w0）
                新加权和 = b1 * w1
                总权重 = w0 + w1
                μ1 = (μ0*w0 + b1*w1) / (w0 + w1)
            代码实现分三步：
                1. ensembled_actions *= cumsum[count-1]  # 还原加权和
                2. ensembled_actions += new_val * weights[count]  # 加上新加权
                3. ensembled_actions /= cumsum[count]  # 恢复到均值
        """
        # 确保权重张量在正确的设备上（CPU/GPU/MPS）
        # 注意：actions 可能在 GPU 上，权重初始化时在 CPU
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            # ── Episode 第一步：初始化 ─────────────────────────────────────
            # 直接用当前预测作为初始在线均值
            # ensembled_actions: (B, chunk_size, action_dim)
            self.ensembled_actions = actions.clone()
            # 计数初始化：每个时间步都被覆盖了 1 次
            # shape: (chunk_size, 1)，最后一维为 1 以便广播到 action_dim
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # ── 后续步：在线加权更新 ───────────────────────────────────────
            # 此时 ensembled_actions 形状为 (B, chunk_size-1, action_dim)
            #（因为上一步已经 pop 掉了 t=0）
            # ensembled_actions 对应 t=1, 2, ..., chunk_size-1 的在线均值

            # 第一步：还原到加权求和状态
            # self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            # 取出每个时间步对应 count 的累积权重
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1  # -1 因为 count 从 1 开始
            ]

            # 第二步：加上新预测的加权值
            # actions[:, :-1]: (B, chunk_size-1, action_dim) 跳过最后一步
            # actions[:, -1]: (B, action_dim) 最后一步单独处理
            # self.ensemble_weights[self.ensembled_actions_count]: 对应权重
            self.ensembled_actions += (
                actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            )

            # 第三步：除以新的累积权重，得到新的在线均值
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]

            # 计数加 1，上限为 chunk_size（防止越界）
            # clamp 后最大为 chunk_size，此时索引 cumsum[chunk_size-1] 合法
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )

            # 将新 chunk 的最后一步（t=chunk_size-1）拼接到末尾
            # 这是该时间步首次出现，没有历史均值可更新，直接赋值
            # 计数设为 1
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )

        # ── 弹出 t=0 时刻的动作作为输出 ─────────────────────────────────
        # action: (B, action_dim)
        # ensembled_actions: (B, chunk_size-1, action_dim) 保留给下一步
        # ensembled_actions_count: (chunk_size-1, 1) 同步更新
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],      # 取出 t=0 的动作
            self.ensembled_actions[:, 1:],      # 丢弃 t=0，保留 t=1..end
            self.ensembled_actions_count[1:],   # 同步更新计数
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer 神经网络主体。

    架构图示：
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   训练模式（use_vae=True）：                                  │
    │   ┌─────────────────────────────────────────────────────┐  │
    │   │ VAE Encoder（Transformer Encoder）                    │  │
    │   │ 输入: [CLS, robot_state, action_sequence]           │  │
    │   │ 输出: latent 分布参数 (mu, log_sigma²)                 │  │
    │   └──────────────────────┬──────────────────────────────┘  │
    │                            │                                │
    │                            │ 重参数化采样 z = μ + σ*ε        │
    │                            ▼                                │
    │   ┌─────────────────────────────────────────────────────┐  │
    │   │ Transformer Encoder                                 │  │
    │   │ 输入: [latent, robot_state, env_state, image_tokens]│  │
    │   │ 输出: encoder_out（融合后的上下文特征）                │  │
    │   └──────────────────────┬──────────────────────────────┘  │
    │                            │                                │
    │                            ▼                                │
    │   ┌─────────────────────────────────────────────────────┐  │
    │   │ Transformer Decoder（DETR 风格 object queries）     │  │
    │   │ 输入: 零向量 + 可学习位置编码                         │  │
    │   │ 上下文: encoder_out                                 │  │
    │   │ 输出: decoder_out（每步对应一个动作隐向量）            │  │
    │   └──────────────────────┬──────────────────────────────┘  │
    │                            │                                │
    │                            ▼                                │
    │                      action_head                            │
    │                   → (B, chunk_size, action_dim)             │
    │                                                             │
    │   推理模式（use_vae=False）：                                │
    │   VAE Encoder 被跳过，latent = 全零向量                     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    三种"编码器"术语澄清：
        1. VAE encoder: 将动作序列编码为隐变量分布参数（仅训练+VAE）
        2. Transformer encoder: 融合观测信息的自注意力编码器
        3. Transformer decoder: 基于观测上下文解码动作序列

    Attributes:
        config: ACTConfig 实例
        vae_encoder: 可选，VAE 编码器（use_vae=True 时使用）
        backbone: ResNet 图像特征提取器（layer4 输出）
        encoder: Transformer 编码器
        decoder: Transformer 解码器
        encoder_*_input_proj: 各输入模态的投影层
        decoder_pos_embed: 解码器可学习位置编码（DETR 风格）
        action_head: 动作输出头
    """

    def __init__(self, config: ACTConfig):
        """
        Args:
            config: ACTConfig 实例，包含所有网络结构超参数

        初始化步骤：
            1. 可选：创建 VAE 编码器及相关投影层
            2. 创建 ResNet 图像 backbone（可选）
            3. 创建 Transformer 编码器和解码器
            4. 创建各输入模态的投影层
            5. 创建位置编码（1D 可学习 + 2D 正弦）
            6. 创建动作输出头
            7. Xavier-uniform 初始化
        """
        # BERT 风格的 VAE 编码器，输入 token 为 [CLS, robot_state, action_sequence]
        # The cls token forms parameters of the latent's distribution
        super().__init__()
        self.config = config

        # ── VAE 编码器（可选，仅训练时使用）────────────────────────────────
        if self.config.use_vae:
            # VAE 编码器使用独立的 Transformer 层数
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            # 可学习的 CLS token embedding，输出代表整个序列的语义向量
            # 1 个 token，dim_model 维
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)

            # robot_state 投影层: (state_dim → dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )

            # action 投影层: (action_dim → dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )

            # latent 输出投影: (dim_model → latent_dim * 2)
            # 输出前半为 mu，后半为 log(σ²)
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2
            )

            # 固定正弦位置编码（供 VAE 编码器使用）
            # token 数量 = 1(CLS) + chunk_size + [1(robot_state)]
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            # register_buffer: 将张量注册为模型的缓冲区（不参与梯度计算，但会随模型保存/加载）
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(
                    num_input_token_encoder, config.dim_model
                ).unsqueeze(0),  # (1, num_tokens, dim_model)
            )

        # ── ResNet 图像 backbone ─────────────────────────────────────────
        if self.config.image_features:
            # 获取 torchvision 中的 ResNet 模型（如 resnet18, resnet50）
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                # replace_stride_with_dilation: 是否用膨胀卷积替代最后 stride=2
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                # FrozenBatchNorm2d: 冻结的 BatchNorm，训练时不更新均值方差（加速且稳定）
                norm_layer=FrozenBatchNorm2d,
            )
            # IntermediateLayerGetter: 提取特定层的输出
            # return_layers = {"layer4": "feature_map"}: 返回 layer4 的输出作为特征图
            # layer4 输出尺寸: (B, 512, H/32, W/32)
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )

        # ── Transformer 编码器和解码器 ───────────────────────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # ── Transformer Encoder 输入投影层 ───────────────────────────────
        # token 顺序: [latent, (robot_state), (env_state), cam0_px, cam1_px, ...]
        # latent 投影: (latent_dim → dim_model)
        self.encoder_latent_input_proj = nn.Linear(
            config.latent_dim, config.dim_model
        )

        # robot_state 投影层: (state_dim → dim_model)
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )

        # env_state 投影层: (env_dim → dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )

        # 图像 token 投影: 1×1 卷积将 ResNet 输出的 512 通道映射到 dim_model
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features,  # ResNet 最后一层通道数（resnet18=512, resnet50=2048）
                config.dim_model,
                kernel_size=1,  # 1×1 卷积，不改变空间尺寸，只改变通道数
            )

        # ── Transformer Encoder 位置编码 ────────────────────────────────
        # 1D token（latent, robot_state, env_state）的可学习位置编码
        n_1d_tokens = 1  # latent token
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(
            n_1d_tokens, config.dim_model
        )

        # 2D 图像特征的正弦位置编码（编码空间位置信息）
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2  # 最终输出通道数为 2 * (dim_model // 2) = dim_model
            )

        # ── Transformer Decoder ─────────────────────────────────────────
        # DETR 风格的可学习位置编码（称为 object queries 或 action queries）
        # chunk_size 个可学习向量，每个对应一个动作时刻
        # 这些向量在交叉注意力中作为 query，与 encoder 输出交互
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ── 动作输出头 ───────────────────────────────────────────────────
        # 将 dim_model 映射到 action_dim（关节目标维度）
        self.action_head = nn.Linear(
            config.dim_model, self.config.action_feature.shape[0]
        )

        # ── 参数初始化 ─────────────────────────────────────────────────
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform 初始化，与原始 ACT 代码一致。

        目的：保持与原始实现相同的初始化策略，避免训练不稳定

        遍历编码器和解码器的所有参数，对二维以上的参数应用 Xavier 初始化。
        Xavier 初始化公式：U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        适用于 ReLU 激活函数
        """
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """ACT 完整前向传播。

        Args:
            batch: 输入数据字典，结构如下：
                - "observation.state" (可选): (B, state_dim) 关节状态
                - "observation.images": [(B, C, H, W), ...] 多路摄像头图像
                  （或 "observation.images.{key}": (B, C, H, W) 单路）
                - "observation.environment_state" (可选): (B, env_dim) 环境状态
                - "action" (仅训练+VAE): (B, chunk_size, action_dim) 目标动作
                - "action_is_pad" (仅训练+VAE): (B, chunk_size) padding 掩码

        Returns:
            actions: (B, chunk_size, action_dim) 预测的动作序列
            (mu, log_sigma_x2): VAE 分布参数元组
                - 训练+VAE: (mu, log_sigma_x2) 均为 (B, latent_dim)
                - 推理或无 VAE: (None, None)

        推理模式数据流（逐步）：
            1. latent = 全零向量（不依赖 VAE encoder）
            2. 准备 encoder token 序列（latent + state + 图像）
            3. 通过 Transformer Encoder 融合信息得到 encoder_out
            4. 通过 Transformer Decoder 生成动作查询响应 decoder_out
            5. 通过 action_head 输出动作

        训练模式数据流（逐步）：
            1. VAE encoder 从 action 序列提取 latent 分布参数
            2. 重参数化采样得到 latent_sample
            3-5. 同推理模式 2-5
        """
        # ── 检查训练模式：需要 VAE + action 提供 ───────────────────────────
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # ── 确定 batch_size ─────────────────────────────────────────────
        if "observation.images" in batch:
            # 多摄像头时取第一路的 batch 维度
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # ── 步骤 1：准备 latent（潜变量）────────────────────────────────
        if self.config.use_vae and "action" in batch and self.training:
            # ▼ 训练分支（use_vae=True）：用 VAE Encoder 从动作序列中编码出 latent 分布参数

            # 准备 VAE 编码器输入: [CLS, robot_state, action_sequence]
            # cls_embed: (B, 1, D) 可学习向量，代表整个序列的语义
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight,
                "1 d -> b 1 d",
                b=batch_size
            )  # (B, 1, D)

            if self.config.robot_state_feature:
                # robot_state_embed: (B, state_dim) → (B, 1, dim_model)
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch["observation.state"]
                ).unsqueeze(1)

            # action_embed: (B, chunk_size, action_dim) → (B, chunk_size, dim_model)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])

            # 拼接所有 token: [CLS, robot_state, action_sequence]
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)  # (B, S+2, D)

            # 克隆并detach正弦位置编码（不参与梯度计算）
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # 准备 padding mask: True = padding 位置（应被 attention 忽略）
            # CLS 和 robot_state token 不是 padding，填 False
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            # 拼接 action padding mask
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (B, S+2)

            # VAE 编码器前向传播
            # 输入: (seq, B, dim_model)，seq = 2 + chunk_size
            # [0] 取 CLS token 输出: (B, D)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),  # (seq, B, D)
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]

            # 投影到 latent 分布参数: (B, D) → (B, latent_dim * 2)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            # 前 latent_dim 维为均值
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 后 latent_dim 维为 2*log(σ)，即 log(σ²)
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]

            # 重参数化采样: z = μ + σ * ε, ε ~ N(0,1)
            # 关键：采样操作允许梯度通过 μ 和 logσ² 反向传播
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)

        else:
            # ▼ 推理分支：不使用 VAE Encoder，latent 直接设为全零向量
            mu = log_sigma_x2 = None  # 推理时无需返回分布参数
            # latent_sample: (B, latent_dim) 全零向量
            # 充当"无信息"的隐变量条件，让模型完全依赖观测信息预测动作
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim],
                dtype=torch.float32
            ).to(batch["observation.state"].device)

        # ── 步骤 2：准备 Transformer Encoder 输入 ─────────────────────────
        # token 顺序: [latent, (robot_state), (env_state), img_tokens...]
        # latent token: (B, latent_dim) → (1, B, dim_model)
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        # 位置编码列表
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )

        # Robot state token so101有关节状态，走着里
        if self.config.robot_state_feature:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_proj(batch["observation.state"])
            )

        # Environment state token外部传感器 我这个机器人没有（力传感器、物体位置等额外输入）
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # 图像 token（逐像素处理，每个像素作为一个 token）
        if self.config.image_features:
            for img in batch["observation.images"]:
                # img: (B, C, H, W) 单路摄像头图像

                # ① ResNet backbone 提取特征图: (B, 512, H/32, W/32)
                cam_features = self.backbone(img)["feature_map"]

                # ② 2D 正弦位置编码: (B, dim_model, h, w)
                # 编码特征图中每个像素的空间位置
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )

                # ③ 1×1 卷积投影: 512 → dim_model
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange: (B, dim_model, h, w) → (h*w, B, dim_model)
                # 每个像素变为一个 token，组成序列
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # 扩展到 token 列表
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 堆叠: (seq_len, B, dim_model)
        # seq_len = 1(latent) + [1(state)] + [1(env)] + n_cam*(h*w)
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # ── 步骤 3：Transformer Encoder ──────────────────────────────────
        # 融合 latent、state、图像信息
        # self-attention 让所有 token 相互交互
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # ── 步骤 4：Transformer Decoder ──────────────────────────────────
        # DETR 风格：零向量 + 可学习位置编码作为 query
        # decoder_in: (chunk_size, B, dim_model) 全零
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        # Decoder 以 decoder_in（零向量+位置编码）为 query
        # 以 encoder_out（观测上下文）为 key/value，通过交叉注意力解码
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # 转置: (chunk_size, B, dim_model) → (B, chunk_size, dim_model)
        decoder_out = decoder_out.transpose(0, 1)

        # ── 步骤 5：动作输出头 ───────────────────────────────────────────
        # (B, chunk_size, dim_model) → (B, chunk_size, action_dim)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Transformer 编码器模块。

    由多个 ACTEncoderLayer 组成，可选最后接 LayerNorm。

    与标准 Transformer Encoder 的区别：
        - 位置编码直接加到 Q 和 K 上（而非 V）
        - 支持 Pre-norm 或 Post-norm

    Args:
        config: ACTConfig 实例
        is_vae_encoder: 是否是 VAE 编码器（使用不同的层数 n_vae_encoder_layers）

    前向传播输入：
        x: (seq_len, B, dim_model) 输入 token 序列
        pos_embed: (seq_len, 1, dim_model) 或 (seq_len, B, dim_model) 位置编码
        key_padding_mask: (B, seq_len) True=padding 位置（用于 VAE 编码器）

    前向传播输出：
        x: (seq_len, B, dim_model) 编码后的 token 序列
    """

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        # VAE 编码器使用独立的层数配置
        num_layers = (
            config.n_vae_encoder_layers
            if self.is_vae_encoder
            else config.n_encoder_layers
        )
        self.layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(num_layers)]
        )
        # 可选的 LayerNorm（pre_norm=True 时生效）
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (seq_len, B, dim_model) 输入 token 序列
            pos_embed: (seq_len, 1, dim_model) 位置编码（加到 Q 和 K）
            key_padding_mask: (B, seq_len) True=padding 位置

        Returns:
            x: (seq_len, B, dim_model) 编码后的序列
        """
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """Transformer 编码器单层。

    结构（Pre-norm 模式）：
        1. LayerNorm → MultiHead Self-Attention → 残差
        2. LayerNorm → FeedForward → 残差

    或（Post-norm 模式，原始 Transformer）：
        1. MultiHead Self-Attention → LayerNorm → 残差
        2. FeedForward → LayerNorm → 残差

    Pre-norm vs Post-norm：
        - Pre-norm: 在注意力/FFN 之前做 LayerNorm，训练更稳定
        - Post-norm: 在注意力/FFN 之后做 LayerNorm，原始 Transformer 使用

    Args:
        config: ACTConfig 实例
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # FeedForward 网络：两个线性层，中间有激活函数
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self,
        x,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """单层编码器前向传播。

        Args:
            x: (seq_len, B, dim_model) 当前层输入
            pos_embed: (seq_len, 1, dim_model) 位置编码（加到 Q 和 K）
            key_padding_mask: (B, seq_len) padding 掩码

        Returns:
            x: (seq_len, B, dim_model) 编码后的特征
        """
        skip = x  # 残差连接

        if self.pre_norm:
            x = self.norm1(x)

        # （位置编码加到 Query 和 Key）Q = K = x + pos_embed
        # （不加位置编码，这是标准做法）V = x
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # FeedForward: Linear → Activation → Dropout → Linear
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)

        return x


class ACTDecoder(nn.Module):
    """Transformer 解码器模块。

    由多个 ACTDecoderLayer 组成，最后接 LayerNorm。

    与标准 Transformer Decoder 的区别：
        - 使用 DETR 风格的零向量 + 可学习位置编码作为输入 query
        - 真正的信息由位置编码（decoder_pos_embed）携带
        - 与 Encoder 输出做交叉注意力

    DETR 风格 object queries 的核心思想：
        - 不像标准解码器那样使用已偏移的输出作为输入
        - 而是使用固定的可学习向量（object queries）
        - 每个 query 通过交叉注意力从 encoder 输出中"查询"相关信息
        - 适合同时预测多个输出的场景（这里每个 query 对应一个动作时刻）

    Args:
        config: ACTConfig 实例
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,           # (chunk_size, B, dim_model) 初始为全零
        encoder_out: Tensor,  # (enc_seq_len, B, dim_model) Encoder 输出
        decoder_pos_embed: Tensor | None = None,  # (chunk_size, 1, dim_model)
        encoder_pos_embed: Tensor | None = None, # (enc_seq_len, 1, dim_model)
    ) -> Tensor:
        """
        Args:
            x: 解码器输入（初始为全零向量，由 decoder_pos_embed 携带信息）
            encoder_out: 编码器输出的上下文特征
            decoder_pos_embed: 解码器位置编码（DETR 风格 object queries）
            encoder_pos_embed: 编码器位置编码

        Returns:
            x: (chunk_size, B, dim_model) 解码后的特征
        """
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """Transformer 解码器单层。

    结构（Pre-norm 模式）：
        1. Self-Attention（action queries 相互注意）
        2. Cross-Attention（action queries 向 encoder 上下文注意）
        3. FeedForward

    子层详解：
        1. 自注意力：chunk_size 个 action query 相互协调
           - Q = K = x + decoder_pos_embed
           - V = x
           - 作用：考虑动作序列的时间一致性

        2. 交叉注意力：action queries 从观测中读取信息
           - Q = x + decoder_pos_embed
           - K = encoder_out + encoder_pos_embed
           - V = encoder_out
           - 作用：第 t 个 action query 查询与第 t 步相关的观测特征

        3. 前馈网络：标准的 FFN 层

    Args:
        config: ACTConfig 实例
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        # 自注意力和交叉注意力
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # FeedForward 网络
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(
        self, tensor: Tensor, pos_embed: Tensor | None
    ) -> Tensor:
        """可选地添加位置编码。"""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,              # (chunk_size, B, dim_model) 解码器输入
        encoder_out: Tensor,    # (enc_seq_len, B, dim_model) 编码器输出
        decoder_pos_embed: Tensor | None = None,  # 解码器位置编码
        encoder_pos_embed: Tensor | None = None,  # 编码器位置编码
    ) -> Tensor:
        """单层解码器前向传播。

        Args:
            x: 解码器当前输入
            encoder_out: 编码器上下文
            decoder_pos_embed: 解码器位置编码
            encoder_pos_embed: 编码器位置编码

        Returns:
            (chunk_size, B, dim_model) 解码后的特征
        """
        # ── 子层 1：自注意力 ───────────────────────────────────────────
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # ── 子层 2：交叉注意力 ─────────────────────────────────────────
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # ── 子层 3：前馈网络 ───────────────────────────────────────────
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)

        return x


def create_sinusoidal_pos_embedding(
    num_positions: int, dimension: int
) -> Tensor:
    """创建 1D 正弦位置编码。

    源自 Attention is All You Need 论文。
    用于 Transformer 的位置编码，让模型感知 token 的顺序。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    其中：
        - pos: 位置索引 [0, num_positions-1]
        - i: 维度索引 [0, dimension-1]
        - d: dimension（总维度）

    为什么用正弦/余弦？
        - 可以表示任意位置的相对偏移（sin(a-b), cos(a-b)）
        - 对于相对位置建模有帮助

    Args:
        num_positions: 位置数量（即 token 序列长度）
        dimension: 每个位置的编码维度（必须为偶数）

    Returns:
        Tensor: (num_positions, dimension) 位置编码矩阵
    """
    def get_position_angle_vec(position: int) -> list:
        # 计算每个维度的频率参数
        # 频率 = 1 / 10000^(2i/d)
        # 维度越高，频率越低（周期越长）
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    # 构建位置编码表
    # 每一行是一个位置的位置编码
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    # 偶数维度用 sin，奇数维度用 cos
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D 正弦位置编码模块。

    与标准 1D 正弦位置编码类似，但用于 2D 图像特征图。
    变体：位置索引归一化到 [0, 2π] 范围。

    输出形状：(1, C, H, W)，可广播到 (B, C, H, W)

    为什么需要 2D 位置编码？
        - 图像是 2D 结构，像素之间的空间关系很重要
        - 需要让模型知道哪个特征来自哪个空间位置

    Args:
        dimension: 编码维度（通常是 dim_model // 2）
                  最终输出通道数为 2 * dimension = dim_model
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: 每个空间位置的编码维度
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6  # 防止除零
        self._temperature = 10000  # 频率基数

    def forward(self, x: Tensor) -> Tensor:
        """生成 2D 正弦位置编码。

        Args:
            x: (B, C, H, W) 输入特征图（来自 ResNet backbone）

        Returns:
            pos_embed: (1, C, H, W) 位置编码（与输入同分辨率）
        """
        # not_mask: (1, H, W) 全 1 标记（表示有效位置）
        # cumsum(1) 对 H 维度累加，得到每行的索引（从 1 开始）
        # 例如：H=4 时，y_range 第一列为 [1,2,3,4]
        not_mask = torch.ones_like(x[0, :1])

        # y_range: (1, H, W) 垂直方向位置索引
        # x_range: (1, H, W) 水平方向位置索引
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # 归一化到 [0, 2π]
        # 除以最大值再乘 2π，使得最后一行/列的值正好为 2π
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        # inverse_frequency: (dimension,) 频率参数
        # 2 * (i // 2) / dimension 确保奇偶索引交替
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )

        # 广播并计算正弦/余弦编码
        # x_range: (1, H, W, 1) / (dimension,) → (1, H, W, dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        # 交错正弦余弦: stack → flatten → (1, H, W, dimension)
        # sin/cos 交替: [..., 0::2] = sin, [..., 1::2] = cos
        pos_embed_x = torch.stack(
            (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed_y = torch.stack(
            (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
        ).flatten(3)

        # 拼接 y 和 x 的编码，然后调整维度顺序
        # torch.cat((y, x), dim=3): (1, H, W, dimension)
        # permute(0, 3, 1, 2): (1, dimension, H, W)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """根据字符串返回对应的激活函数。

    Args:
        activation: 激活函数名称 ("relu", "gelu", "glu")

    Returns:
        对应的激活函数

    Raises:
        RuntimeError: 未知激活函数名称

    激活函数说明：
        - relu: max(0, x)，简单高效
        - gelu: x * sigmoid(1.702x)，Transformer 常用
        - glu: gated linear unit，x * sigmoid(W*x)，用于门控
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
