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

"""Action Chunking Transformer (ACT) 策略配置模块

本模块定义了 ACT 策略的所有超参数配置类。

主要配置项分类：
    1. 输入输出结构：n_obs_steps, chunk_size, n_action_steps
    2. 数据预处理：normalization_mapping, input/output_shapes
    3. 模型架构：vision_backbone, Transformer 参数
    4. VAE 设置：use_vae, latent_dim
    5. 推理选项：temporal_ensemble_coeff
    6. 训练设置：dropout, kl_weight, optimizer 参数
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """ACT 策略的配置类。

    默认配置针对在 Aloha 双臂机器人上执行精细操作任务（如插销、转移）调优。

    需要根据您的环境/传感器调整的参数：
        - input_shapes: 输入数据的形状（摄像头、关节状态等）
        - output_shapes: 输出数据的形状（动作维度）

    输入要求：
        - 至少需要一个以 "observation.image" 开头的 key 作为视觉输入
          或者 "observation.environment_state" 作为环境状态输入
        - 如果有多个以 "observation.images." 开头的 key，视为多摄像头视角
        - 可选包含 "observation.state" 作为机器人本体感知状态
        - 输出必须包含 "action" key

    超参数说明：

    Action Chunking 参数：
        - chunk_size: 每次预测的动作序列长度
          例如 chunk_size=100 表示一次预测未来 100 步的动作
        - n_action_steps: 每次实际执行的步数
          例如 chunk_size=100, n_action_steps=50 表示预测 100 步但只执行 50 步
          这样可以减少推理频率，提高实时性

    时序集成参数：
        - temporal_ensemble_coeff: 指数加权系数（论文推荐 0.01）
          非 None 时启用时序集成，每步都要推理（n_action_steps 必须为 1）
          启用后动作更平滑，但计算成本更高

    VAE 参数：
        - use_vae: 是否使用变分自编码器目标训练
          True 时需要 action 数据用于学习隐空间
          False 时推理更快（latent 全零跳过 encoder）
        - latent_dim: 隐变量维度，默认 32
        - kl_weight: KL 损失的权重，默认 10.0

    Transformer 参数：
        - dim_model: Transformer 隐藏层维度，默认 512
        - n_heads: 多头注意力头数，默认 8
        - n_encoder_layers: 编码器层数，默认 4
        - n_decoder_layers: 解码器层数，默认 1（原 ACT 实现有 bug）
        - dim_feedforward: FFN 隐藏层维度，默认 3200

    视觉 backbone 参数：
        - vision_backbone: ResNet 骨架网络，默认 resnet18
        - pretrained_backbone_weights: 预训练权重，默认 ImageNet1K

    训练参数：
        - dropout: Dropout 比例，默认 0.1
        - optimizer_lr: 默认学习率 1e-5
        - optimizer_lr_backbone: backbone 学习率，默认 1e-5
    """

    # ============================================================
    # 输入输出结构
    # ============================================================

    # 观测步数：输入给策略的观测帧数（从当前步往前数）
    # 目前仅支持 1（多步观测尚未实现）
    n_obs_steps: int = 1

    # 动作块大小：每次预测的动作序列长度（单位：环境步）
    # 这是 ACT 的核心参数，决定了动作预测的时间跨度
    chunk_size: int = 100

    # 每次执行的动作步数：调用策略一次实际执行多少步
    # 应该小于等于 chunk_size
    # 例如 chunk_size=100, n_action_steps=50 表示：
    #   - 推理一次预测 100 步动作
    #   - 执行其中的 50 步
    #   - 剩余 50 步缓存用于后续执行（如果环境执行<50步/调用）
    n_action_steps: int = 100

    # 数据归一化映射
    # 键为模态类型字符串，值为归一化模式
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,   # 图像：减均值除标准差
            "STATE": NormalizationMode.MEAN_STD,    # 状态：减均值除标准差
            "ACTION": NormalizationMode.MEAN_STD,   # 动作：减均值除标准差
        }
    )

    # ============================================================
    # 模型架构
    # ============================================================

    # 视觉骨架网络
    # 用于从摄像头图像提取特征
    vision_backbone: str = "resnet18"

    # 预训练的骨干网络权重
    # None 表示不使用预训练权重
    # "ResNet18_Weights.IMAGENET1K_V1" 使用 ImageNet 预训练
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # 是否将 ResNet 最后的 stride=2 替换为膨胀卷积
    # True：增加感受野但增加计算量
    # False：标准 ResNet
    replace_final_stride_with_dilation: int = False

    # Transformer 层归一化模式
    # True: Pre-norm（在注意力/FFN 前做 LayerNorm），训练更稳定
    # False: Post-norm（原始 Transformer 模式）
    pre_norm: bool = False

    # Transformer 主隐藏维度
    # 所有投影层、注意力层的维度基准
    dim_model: int = 512

    # 多头注意力头数
    # dim_model 必须能被 n_heads 整除
    n_heads: int = 8

    # Feed-Forward 网络的隐藏层维度
    # 通常是 dim_model 的 4-6 倍
    dim_feedforward: int = 3200

    # FFN 激活函数
    # 可选: "relu", "gelu", "glu"
    feedforward_activation: str = "relu"

    # Transformer 编码器层数
    # 用于融合观测信息（图像、状态、latent）
    n_encoder_layers: int = 4

    # Transformer 解码器层数
    # 注意：原始 ACT 实现有 bug，只使用了第一层
    # 这里与原始实现保持一致，默认为 1
    # See: https://github.com/tonyzhaozh/act/issues/25#issue-2258740521
    n_decoder_layers: int = 1

    # ============================================================
    # VAE（变分自编码器）设置
    # ============================================================

    # 是否使用 VAE 目标训练
    # True: 训练时使用 VAE encoder 学习动作序列的隐空间
    #       损失 = L1_loss + kl_weight * KL_loss
    # False: 训练时直接回归动作，不使用 VAE
    #        推理时 latent 全零
    use_vae: bool = True

    # VAE 隐变量维度
    # 论文使用 32 维
    latent_dim: int = 32

    # VAE 编码器层数（独立于 Transformer 编码器）
    n_vae_encoder_layers: int = 4

    # ============================================================
    # 推理选项
    # ============================================================

    # 时序集成系数
    # None: 不使用时序集成
    # 0.01: 论文推荐值，越大越保守（更重视历史预测）
    # 使用时 n_action_steps 必须为 1（每步都要推理）
    # See ACTTemporalEnsembler 了解工作原理
    temporal_ensemble_coeff: float | None = None

    # ============================================================
    # 训练和损失计算
    # ============================================================

    # Dropout 比例
    dropout: float = 0.1

    # KL 损失的权重（仅 use_vae=True 时使用）
    # 总损失 = L1_loss + kl_weight * KL_loss
    # 论文使用 10.0
    kl_weight: float = 10.0

    # ============================================================
    # 训练预设（传递给优化器）
    # ============================================================

    # 默认学习率
    optimizer_lr: float = 1e-5

    # 权重衰减
    optimizer_weight_decay: float = 1e-4

    # 骨干网络学习率（通常与主学习率相同或更小）
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        """配置验证和后处理。

        验证项：
            1. vision_backbone 必须是 ResNet 系列
            2. 时序集成时 n_action_steps 必须为 1
            3. n_action_steps 不能超过 chunk_size
            4. 目前仅支持 n_obs_steps=1
        """
        super().__post_init__()

        # 验证视觉骨架网络
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # 时序集成时 n_action_steps 必须为 1
        # 因为时序集成需要每步都推理来计算指数加权平均
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )

        # n_action_steps 不能超过 chunk_size
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        # 目前仅支持单步观测
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回优化器预设配置。

        Returns:
            AdamWConfig: 包含学习率、权重衰减等参数的优化器配置
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        """返回学习率调度器预设。

        ACT 策略目前不使用学习率调度器。

        Returns:
            None
        """
        return None

    def validate_features(self) -> None:
        """验证输入特征配置。

        至少需要提供以下之一：
            - 至少一个图像特征（observation.images.*）
            - 环境状态特征（observation.environment_state）

        Raises:
            ValueError: 当既没有图像也没有环境状态时抛出
        """
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        """观测增量索引。

        返回 None 表示直接使用原始观测，不计算增量。
        """
        return None

    @property
    def action_delta_indices(self) -> list:
        """动作增量索引。

        返回 list(range(chunk_size)) 表示预测的是绝对动作而非增量。
        """
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """奖励增量索引。

        返回 None 表示直接使用原始奖励。
        """
        return None
