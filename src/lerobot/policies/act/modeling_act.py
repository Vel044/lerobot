#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
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
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        Args:
            batch: 来自环境的一帧观测，字典结构如下：
                - "observation.state": (B, state_dim) 关节角度等本体感知状态
                - "observation.images" / 各摄像头 key: (B, C, H, W) 图像帧
                  （具体 key 由 config.image_features 决定）

        Returns:
            action: (B, action_dim) 当前时刻要执行的一个动作向量（关节目标位置）
        """
        # 强制进入 eval 模式，关闭 dropout/batchnorm 的训练行为。
        # 必要：队列消费期间外部代码可能把模型切回 train 模式。
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        # ── 路径 A：启用时序集成（temporal ensembling） ──────────────────────────────
        # temporal_ensemble_coeff 非 None 时，每一步都完整调用模型推理一次（n_action_steps 必须为 1），
        # 将新预测的 chunk 与历史预测做指数加权平均后输出当前动作，以平滑抖动。
        if self.config.temporal_ensemble_coeff is not None:
            # actions: (B, chunk_size, action_dim) —— 完整地预测未来 chunk_size 步的动作序列
            actions = self.predict_action_chunk(batch)
            # 将本次 chunk 与历史 chunk 做指数加权融合，返回当前时刻的融合动作
            # action: (B, action_dim)
            action = self.temporal_ensembler.update(actions)
            return action

        # ── 路径 B：action chunk 队列（默认，无时序集成） ────────────────────────────
        # 模型一次预测 chunk_size 步的动作，但只执行其中 n_action_steps 步。
        # 队列为空时才重新查询模型，减少推理次数，提升实时性。
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # predict_action_chunk 返回 (B, chunk_size, action_dim)；
            # 只取前 n_action_steps 步入队，其余丢弃
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            # transpose(0,1) → (n_action_steps, B, action_dim)；
            # extend 后队列里有 n_action_steps 个形如 (B, action_dim) 的张量
            self._action_queue.extend(actions.transpose(0, 1))

        # 从队列头部弹出当前时刻的动作，返回 (B, action_dim)
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: 环境观测字典，至少包含：
                - "observation.state": (B, state_dim) 本体感知关节状态
                - 各摄像头图像 key（如 "observation.images.top"）: (B, C, H, W)

        Returns:
            actions: (B, chunk_size, action_dim)  未来 chunk_size 步的预测动作序列
        """
        self.eval()

        if self.config.image_features:
            # 浅拷贝 batch，避免修改调用方的原始字典
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # 将各摄像头图像张量收集成列表，统一存入 "observation.images" key
            # config.image_features 是按顺序排好的摄像头 key 列表
            # 结果：batch["observation.images"] = [(B,C,H,W), (B,C,H,W), ...]
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 调用 ACT 模型前向推理（推理时无 VAE encoder，latent 全零）
        # model(batch) 返回 (actions, (mu, log_sigma_x2))
        #   actions: (B, chunk_size, action_dim)
        #   mu / log_sigma_x2: 推理时均为 None
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.

        Args:
            actions: (B, chunk_size, action_dim)
                本次推理预测的完整动作序列。
                B=batch_size，chunk_size 为动作块长度，action_dim 为关节维度。

        Returns:
            action: (B, action_dim)  当前时刻要执行的加权融合动作

        工作原理（在线指数加权平均）：
            每一步都用新预测的 chunk 更新历史均值。
            权重 w_i = exp(-coeff * i)，i=0 对应最旧的预测（权重最大）。
            这样旧预测的稳定性被保留，新预测只做小幅修正，避免动作抖动。
        """
        # 确保权重张量在与 actions 相同的设备上（CPU/GPU/MPS）
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            # ── episode 第一步：直接用当前预测初始化在线均值 ────────────────────────
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            # ensembled_actions: (B, chunk_size, action_dim)
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            # ensembled_actions_count[i] = 已有多少次预测覆盖了第 i 步（首步全为 1）
            # shape: (chunk_size, 1)，最后一维为 1 以便广播
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # ── 后续步骤：在线加权更新 ──────────────────────────────────────────────
            # 当前 ensembled_actions 已消费掉 t=0 这一步（见下方 popleft），
            # 此时形状为 (B, chunk_size-1, action_dim)，对应未来 t=1..chunk_size-1 的历史均值。
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.

            # 在线均值更新公式：
            #   new_avg = (old_avg * Σw_old + new_val * w_new) / Σw_new
            # 先乘回旧累积权重（还原加权和），加上新预测 × 新权重，再除以新累积权重
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            # actions[:, :-1]: (B, chunk_size-1, action_dim)，跳过最后一步（无历史可更新）
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            # 计数 +1，上限为 chunk_size（防止索引越界）
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)

            # The last action, which has no prior online average, needs to get concatenated onto the end.
            # 新 chunk 的最后一步（t=chunk_size-1）是首次预测，直接拼接到末尾，计数为 1
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )

        # "Consume" the first action.
        # 弹出 t=0 的融合动作作为本步执行动作，剩余部分留给下一步更新
        # action: (B, action_dim)
        # ensembled_actions 变为 (B, chunk_size-1, action_dim)
        # ensembled_actions_count 变为 (chunk_size-1, 1)
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.

        推理（非训练）时的整体数据流：
            1. latent 全零 → encoder_latent_input_proj → 1D token
            2. observation.state → encoder_robot_state_input_proj → 1D token
            3. 每路摄像头图像 → ResNet backbone → 1×1 Conv → (H*W) 个 2D token
            4. 所有 token 拼接 → Transformer Encoder → 上下文特征 encoder_out
            5. 零初始化查询 + 位置编码 → Transformer Decoder（与 encoder_out 交叉注意力）
            6. decoder_out → action_head（线性层）→ (B, chunk_size, action_dim)
        """
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # 从图像或环境状态中读取 batch_size
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # ── 步骤 1：准备 latent（潜变量）作为 Transformer Encoder 的输入 token ──────
        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch and self.training:
            # ▼ 训练分支（use_vae=True）：用 VAE Encoder 从动作序列中编码出 latent 分布参数
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            # cls_embed: (B, 1, D)  — 可学习的 CLS token，最终输出代表整个序列的语义
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                # robot_state_embed: (B, D) → (B, 1, D)
                # 将 state_dim 维关节状态线性映射到 dim_model 维隐空间
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            # action_embed: (B, chunk_size, D)  — 将动作序列（演示数据）线性映射到隐空间
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            # vae_encoder_input: (B, 1 or 2 + chunk_size, D)  拼接所有输入 token
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            # pos_embed: (1, S+1 or 2, D)  — 固定正弦位置编码，在 __init__ 中预计算并注册为 buffer
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            # cls_joint_is_pad: (B, 1 or 2)  — CLS 和 robot_state token 不是 padding，填 False
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            # key_padding_mask: (B, 1 or 2 + chunk_size)
            # True 表示该位置是 padding（应被 attention 忽略），来自数据集中短 episode 的末尾填充
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            # vae_encoder_input 需转置为 (seq, B, D) 以符合 PyTorch Transformer 的输入约定
            # [0] 取 CLS token 的输出，shape: (B, D)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            # 将 CLS 输出投影到 latent_dim*2 维：前半为均值 mu，后半为 2*log(sigma)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            # mu: (B, latent_dim)  — 潜变量分布的均值
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            # log_sigma_x2: (B, latent_dim)  — 2*log(σ)，即 log(σ²)
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 重参数化采样：z = mu + sigma * epsilon，epsilon ~ N(0,1)
            # 允许梯度通过 mu 和 log_sigma_x2 反向传播
            # latent_sample: (B, latent_dim)
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # ▼ 推理分支：不使用 VAE Encoder，latent 直接设为全零向量
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None  # 推理时无需返回分布参数
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            # latent_sample: (B, latent_dim)  — 全零向量，推理时充当"无信息"的 latent 条件
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # ── 步骤 2：组装 Transformer Encoder 的输入 token 序列 ──────────────────────
        # Prepare transformer encoder inputs.
        # token 顺序：[latent, (robot_state), (env_state), cam0_px_0, ..., cam0_px_N, cam1_px_0, ...]

        # latent token: (B, latent_dim) → 线性映射 → (B, dim_model)
        # unsqueeze(0) 后变 (1, B, dim_model)，作为序列的第一个 token
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        # encoder_1d_feature_pos_embed: Embedding(n_1d_tokens, dim_model)
        # .weight: (n_1d_tokens, dim_model)，unsqueeze(1) → (n_1d_tokens, 1, dim_model)
        # 转为 list 后每个元素 shape: (1, dim_model)，代表一个 1D token 的位置编码
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Robot state token.
        if self.config.robot_state_feature:
            # observation.state: (B, state_dim) → 线性映射 → (B, dim_model)
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))

        # Environment state token.
        if self.config.env_state_feature:
            # observation.environment_state: (B, env_dim) → 线性映射 → (B, dim_model)
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch["observation.images"]:
                # img: (B, C, H, W)  — 一路摄像头的 RGB 图像帧
                # ① ResNet backbone 提取特征图
                #    cam_features: (B, 512, H/32, W/32) — ResNet layer4 的输出特征图
                cam_features = self.backbone(img)["feature_map"]
                # ② 2D 正弦位置编码，编码特征图中每个像素的空间位置
                #    cam_pos_embed: (B, dim_model, h, w)
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                # ③ 1×1 卷积将通道数从 512 压缩到 dim_model（=512，但解耦与 backbone 输出维度）
                #    cam_features: (B, dim_model, h, w)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                # (B, dim_model, h, w) → (h*w, B, dim_model)  符合 PyTorch Transformer 序列优先的约定
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                # 将 h*w 个像素级 token 逐一加入序列，每个 shape: (B, dim_model)
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # ── 步骤 3：堆叠并送入 Transformer Encoder ──────────────────────────────────
        # Stack all tokens along the sequence dimension.
        # encoder_in_tokens: (seq_len, B, dim_model)
        #   seq_len = 1(latent) + [1(state)] + [1(env)] + n_cam*(h*w)
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        # Encoder 对所有输入 token 做自注意力，融合 latent、state、图像信息
        # encoder_out: (seq_len, B, dim_model)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # ── 步骤 4：Transformer Decoder 解码出 chunk_size 步的动作 ─────────────────
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        # decoder_in: (chunk_size, B, dim_model)  全零张量作为 Decoder 的初始查询（query）
        # 真正的位置信息完全由 decoder_pos_embed 携带（DETR 风格 object queries）
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        # Decoder 以 decoder_in（零向量+位置编码）为 query，
        # 以 encoder_out（观测上下文）为 key/value，通过交叉注意力解码动作
        # decoder_out: (chunk_size, B, dim_model)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            # decoder_pos_embed.weight: (chunk_size, dim_model)
            # unsqueeze(1) → (chunk_size, 1, dim_model)，广播到每个 batch 样本
        )

        # Move back to (B, S, C).
        # (chunk_size, B, dim_model) → (B, chunk_size, dim_model)
        decoder_out = decoder_out.transpose(0, 1)

        # ── 步骤 5：线性映射到动作空间 ────────────────────────────────────────────
        # action_head: Linear(dim_model, action_dim)
        # actions: (B, chunk_size, action_dim)  — 预测的未来 chunk_size 步关节目标角度
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            x: (seq_len, B, dim_model)  输入 token 序列（latent + state + 图像像素 token）
            pos_embed: (seq_len, 1, dim_model) or (seq_len, B, dim_model)  每个 token 的位置编码，
                       直接与 token 特征相加后作为 Q/K（不修改 V）
            key_padding_mask: (B, seq_len)  True 表示对应位置是 padding，attention 时被屏蔽（仅 VAE 训练时使用）

        Returns:
            x: (seq_len, B, dim_model)  经过 n_encoder_layers 层自注意力后的上下文特征
        """
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        # LayerNorm（pre_norm=True 时）或 Identity（pre_norm=False 时）
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        """单层 Transformer Encoder（Pre-norm 或 Post-norm 均支持）

        Args:
            x:               (seq_len, B, dim_model)  当前层输入 token 特征
            pos_embed:       (seq_len, 1, dim_model)  位置编码（与 token 特征相加后做 Q/K，V 不加）
            key_padding_mask:(B, seq_len)  True=padding token，attention 中忽略（推理时为 None）

        Returns:
            x: (seq_len, B, dim_model)  经过自注意力 + FFN 后的 token 特征
        """
        skip = x  # 残差连接暂存
        if self.pre_norm:
            x = self.norm1(x)
        # 位置编码只加到 Q 和 K，不加到 V（标准做法：让 attention 感知位置，但值不受影响）
        q = k = x if pos_embed is None else x + pos_embed
        # 多头自注意力：Q=K=x+pos_embed，V=x（不带位置）
        # 返回 (output, attn_weights)，[0] 取输出
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        # 残差 + dropout
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)  # Post-norm：注意力输出后先 Norm
            skip = x
        # 前馈网络（FFN）：Linear → Activation → Dropout → Linear
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)  # 残差 + dropout
        if not self.pre_norm:
            x = self.norm2(x)  # Post-norm：FFN 输出后再 Norm
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:               (chunk_size, B, dim_model)  Decoder 输入查询（推理时全零，靠位置编码携带信息）
            encoder_out:     (enc_seq_len, B, dim_model) Encoder 输出的上下文特征，作为交叉注意力的 K/V
            decoder_pos_embed:(chunk_size, 1, dim_model) Decoder 的可学习位置编码（DETR 风格 object queries）
            encoder_pos_embed:(enc_seq_len, 1, dim_model) Encoder 位置编码，交叉注意力时加到 K

        Returns:
            x: (chunk_size, B, dim_model)  经过 n_decoder_layers 层解码后的特征，
               每个位置对应一个未来时刻的动作隐向量
        """
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)  # LayerNorm 对 decoder 输出做归一化
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """单层 Transformer Decoder（自注意力 + 交叉注意力 + FFN）

        Args:
            x:               (chunk_size, B, dim_model)  Decoder 当前层输入（初始为全零）
            encoder_out:     (enc_seq_len, B, dim_model) Encoder 输出的观测上下文特征
            decoder_pos_embed:(chunk_size, 1, dim_model) Decoder object query 位置编码
                              — 推理时 x 全零，所有"动作意图"信息完全来自这里
            encoder_pos_embed:(enc_seq_len, 1, dim_model) Encoder 侧位置编码，
                              加到交叉注意力的 K，使 Decoder 知道各 Encoder token 的位置

        Returns:
            (chunk_size, B, dim_model) decoder 输出特征
        """
        # ── 子层 1：Decoder 自注意力（action queries 互相关注）────────────────────
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        # Q=K=x+decoder_pos_embed，V=x（不带位置）
        # chunk_size 个 action query 相互做 self-attention，协调不同时刻动作的一致性
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)  # 残差 + dropout
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # ── 子层 2：交叉注意力（action queries 向观测上下文提问）──────────────────
        # Q = decoder query + decoder_pos_embed
        # K = encoder_out   + encoder_pos_embed （观测 token 加位置）
        # V = encoder_out                        （观测 token 不加位置）
        # 每个 action query（时刻 t）从全局观测特征中"读取"与该时刻动作相关的信息
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)  # 残差 + dropout
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # ── 子层 3：前馈网络（FFN）──────────────────────────────────────────────
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
