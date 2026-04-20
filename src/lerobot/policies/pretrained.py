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
import abc
import builtins
import logging
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import packaging
import safetensors
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.hub import HubMixin

# T 是绑定到 PreTrainedPolicy 的泛型变量，用于 from_pretrained 等类方法返回正确的子类实例
T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    所有策略模型的基类。
    继承 nn.Module（PyTorch 模型）、HubMixin（HuggingFace Hub 上传/下载）、abc.ABC（抽象基类）。
    """

    # 子类必须覆盖：config_class 指向对应的配置类（如 ACTConfig），name 为策略字符串标识
    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        # config: PreTrainedConfig 实例，包含模型超参、设备、输入输出维度等所有配置
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        # 每当有子类继承本类时自动调用，强制要求子类必须定义 config_class 和 name，否则报错
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        # save_directory: Path，模型保存目录
        # 先保存配置文件（config.json / config.yaml），再以 safetensors 格式保存模型权重
        self.config._save_pretrained(save_directory)
        # 若模型被 DataParallel/DistributedDataParallel 包装，取内部的 .module；否则直接保存自身
        model_to_save = self.module if hasattr(self, "module") else self
        # SAFETENSORS_SINGLE_FILE = "model.safetensors"，安全、快速的张量存储格式
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,  # 若为 None，则从 Hub/本地路径自动加载配置
        force_download: bool = False,             # True：强制重新下载，忽略本地缓存
        resume_download: bool | None = None,      # True：断点续传
        proxies: dict | None = None,              # HTTP 代理设置，格式 {"http": "...", "https": "..."}
        token: str | bool | None = None,          # HuggingFace 访问令牌，用于私有模型
        cache_dir: str | Path | None = None,      # 本地缓存目录，None 则用默认 HF_HOME
        local_files_only: bool = False,           # True：仅使用本地缓存，不访问网络
        revision: str | None = None,              # 模型版本（branch/tag/commit hash）
        strict: bool = False,                     # False：允许 keys 不完全匹配（方便部分加载）
        **kwargs,
    ) -> T:
        """
        加载预训练策略模型，默认进入 eval 模式（dropout 关闭）。
        训练时需手动调用 policy.train() 切回训练模式。
        返回值 T：当前子类（如 ACTPolicy）的实例。
        """
        if config is None:
            # 未传入 config 时，从 pretrained_name_or_path 对应的 Hub 仓库或本地目录自动解析配置
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        # 用已解析的 config 构建空模型实例（此时权重随机初始化，后续会被覆盖）
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            # 本地目录加载：拼接 model.safetensors 的完整路径
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                # 从 HuggingFace Hub 下载 model.safetensors 到本地缓存后再加载
                model_file = hf_hub_download(
                    repo_id=model_id,           # Hub 仓库 ID，如 "lerobot/act_so101_push"
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # 将模型参数迁移到 config.device（如 "cpu"、"cuda:0"），并切换到推理模式
        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # model: 空权重的模型实例；model_file: safetensors 文件路径；map_location: 目标设备字符串
        # strict=False 允许模型与文件中的 key 不完全一致（额外 key 或缺失 key 不报错）
        kwargs = {"strict": strict}

        # safetensors >= 0.4.3 原生支持直接加载到指定设备，避免先加载到 CPU 再拷贝的开销
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # 返回值：(missing_keys: list[str], unexpected_keys: list[str])
        # missing_keys：模型中有但文件里没有的参数名；unexpected_keys：文件里有但模型中没有的参数名
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # 旧版 safetensors 不支持 device 参数，加载完成后手动将模型移到目标设备
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        返回用于构建优化器的参数组字典。
        子类可在此区分不同层的学习率/权重衰减，如 {"params": [...], "lr": 1e-4}。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        每次环境重置时调用，清空策略内部的状态缓存（如动作队列、历史观测帧缓冲区）。
        对于 action chunking 策略（ACT），需清空已预测但未执行的动作队列。
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """
        训练前向传播，计算损失。

        参数：
            batch: dict[str, Tensor]，一批训练样本，键包括：
                - "observation.*"：各传感器观测（图像 [B,C,H,W]、关节状态 [B,D] 等）
                - "action"：目标动作序列 [B, T_a, D_a]

        返回：
            (loss, info)
            loss: Tensor，标量，反向传播用的总损失
            info: dict | None，日志友好的原生 Python 类型（如 {"mse_loss": 0.03}），可为 None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        给定当前观测，预测完整的动作序列块（action chunk）。

        参数：
            batch: dict[str, Tensor]，当前观测，键同 forward 中的 observation.*

        返回：
            Tensor，形状 [B, T_a, D_a]，B=batch size，T_a=预测时域长度，D_a=动作维度
        子类在 select_action 内部调用此方法，将结果缓存后逐步取出执行。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        推理时每步调用，返回当前步应执行的单个动作。

        参数：
            batch: dict[str, Tensor]，当前时刻观测

        返回：
            Tensor，形状 [B, D_a]，本步要发送给机器人的关节目标位置/速度
        负责管理 action chunk 缓存：chunk 耗尽时调用 predict_action_chunk 重新预测。
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,  # 训练流水线配置，包含 dataset.repo_id 等元数据
    ):
        api = HfApi()
        # 在 Hub 上创建（或复用已有的）模型仓库，返回规范化的 repo_id 字符串
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        # 将所有文件保存到临时目录，再一次性作为单个 commit 上传，避免多次网络请求
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id

            # 保存模型权重（model.safetensors）和模型配置（config.json）
            self.save_pretrained(saved_path)

            # 生成 README.md（ModelCard），包含模型类型、数据集、许可证等元数据
            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            # 保存训练配置（train_config.yaml），记录超参数供复现
            cfg.save_pretrained(saved_path)

            # 将整个目录上传到 Hub，仅上传指定后缀文件，排除临时/日志文件
            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload policy weights, train config and readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"Model pushed to {commit_info.repo_url.url}")

    def generate_model_card(
        self,
        dataset_repo_id: str,      # 训练使用的数据集 Hub ID，如 "lerobot/push_t"
        model_type: str,            # 策略类型字符串，如 "act"、"smolvla"、"diffusion"
        license: str | None,        # 模型许可证，如 "apache-2.0"；None 时默认使用 apache-2.0
        tags: list[str] | None,     # 额外标签列表；None 时仅使用默认标签
    ) -> ModelCard:
        # smolvla 基于预训练视觉语言模型微调，需要声明 base_model；其他策略从头训练
        base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None

        # ModelCardData 存储 Hub 元数据，会被渲染为 YAML front matter 写入 README.md
        card_data = ModelCardData(
            license=license or "apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            # 合并用户自定义标签与默认标签集合，去重后转为列表
            tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
            model_name=model_type,
            datasets=dataset_repo_id,
            base_model=base_model,
        )

        # 从包内资源读取 Markdown 模板（lerobot/templates/lerobot_modelcard_template.md）
        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
        )
        # 用 card_data 填充模板，生成最终的 ModelCard 对象，并校验格式合法性
        card = ModelCard.from_template(card_data, template_str=template_card)
        card.validate()
        return card
