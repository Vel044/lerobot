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

"""
数据采集模块 - 记录机器人数据集。

功能说明：
    本模块是LeRobot的核心推理/采集模块，负责：
    1. 连接机器人和遥操作设备
    2. 控制机器人和采集数据（支持遥操作和策略两种模式）
    3. 将采集的数据打包成HuggingFace格式数据集

控制模式：
    1. 遥操作模式(Teleoperation): 人实时控制机器人，动作直接来自遥操作设备
    2. 策略模式(Policy): 预训练策略自动生成动作
    3. 混合模式: 遥操作和策略可同时使用

数据流：
    机器人观测 → 策略/遥操作推理 → 动作下发 → 数据存储

使用示例：
    # 命令行采集数据
    lerobot-record \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58760431541 \
        --robot.cameras="{laptop: {type: opencv, index_or_path: 0}}" \
        --dataset.repo_id=username/my_dataset \
        --dataset.num_episodes=10 \
        --dataset.single_task="抓取方块"

依赖模块：
    - robots: 机器人硬件接口
    - teleoperators: 遥操作设备接口
    - policies: 预训练策略
    - processor: 数据处理器流水线
    - datasets: 数据集存储

作者: HuggingFace Inc.
"""

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

# ============================================================
# 相机配置导入（支持多种相机类型）
# ============================================================
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

# ============================================================
# 配置解析
# ============================================================
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# ============================================================
# 数据集相关
# ============================================================
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager

# ============================================================
# 策略相关
# ============================================================
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy

# ============================================================
# 处理器流水线
# ============================================================
from lerobot.processor import (
    PolicyAction,  # 策略输出的动作类型
    PolicyProcessorPipeline,  # 策略预/后处理流水线类型
    RobotAction,  # 机器人动作类型
    RobotObservation,  # 机器人观测类型
    RobotProcessorPipeline,  # 机器人处理器流水线类型
    make_default_processors,  # 创建默认处理器流水线
)
from lerobot.processor.rename_processor import rename_stats

# ============================================================
# 机器人相关
# ============================================================
from lerobot.robots import (  # noqa: F401
    Robot,  # 机器人基类
    RobotConfig,  # 机器人配置基类
    # 具体机器人类型（用于类型检查）
    bi_so100_follower,
    bi_so101_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,  # 从配置创建机器人实例
    so100_follower,
    so101_follower,
    xlerobot,
)

# ============================================================
# 遥操作相关
# ============================================================
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,  # 遥操作设备基类
    TeleoperatorConfig,  # 遥操作配置基类
    # 具体遥操作类型
    bi_so100_leader,
    bi_so101_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,  # 从配置创建遥操作实例
    so100_leader,
    so101_leader,
    xlebi_so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

# ============================================================
# 控制和工具
# ============================================================
from lerobot.utils.control_utils import (
    init_keyboard_listener,  # 初始化键盘监听（用于中断控制）
    is_headless,  # 检查是否无头模式
    predict_action,  # 策略推理
    sanity_check_dataset_name,  # 检查数据集名称合法性
    sanity_check_dataset_robot_compatibility,  # 检查数据集与机器人兼容性
)
from lerobot.utils.robot_utils import busy_wait  # 忙等待（精确延时）
from lerobot.utils.utils import (
    get_safe_torch_device,  # 获取安全的torch设备
    init_logging,  # 初始化日志
    log_say,  # 语音播报（可选）
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class DatasetRecordConfig:
    """
    数据集采集配置 - 定义单个Episode的采集参数。

    Attributes (属性说明):

    repo_id (str):
        数据集标识符，格式为 "{hf用户名}/{数据集名}"
        示例: "lerobot/test" 或 "myusername/pick_place_dataset"
        用于HuggingFace Hub上的数据集发布

    single_task (str):
        任务描述文字，简短准确说明本episode执行的任务
        示例: "Pick the Lego block and drop it in the box on the right."
        将作为数据集的task字段存储

    root (str | Path | None):
        数据集本地存储根目录
        默认None时使用数据集缓存目录
        示例: "dataset/path" 或 Path("/home/user/data")

    fps (int):
        目标采集帧率，默认30fps
        控制数据采集和控制循环的频率
        注意：实际帧率受机器人硬件性能限制

    episode_time_s (int | float):
        每个Episode的采集时长（秒），默认60秒
        到时间后自动结束当前episode

    reset_time_s (int | float):
        每次Episode结束后的重置环境时长（秒），默认60秒
        用于手动将机器人或物体恢复到初始位置
        注意：最后一个episode不执行重置

    num_episodes (int):
        计划采集的Episode总数，默认50
        采集完成后自动停止

    video (bool):
        是否将帧编码为视频存储，默认True
        False时保存原始PNG序列，会占用更多磁盘空间

    push_to_hub (bool):
        是否上传数据集到HuggingFace Hub，默认True
        需要登录huggingface-cli

    private (bool):
        上传时是否创建私有仓库，默认False

    tags (list[str] | None):
        数据集的HuggingFace标签列表
        示例: ["robotics", "manipulation"]

    num_image_writer_processes (int):
        图像写入子进程数量，默认0（仅用线程）
        - 0: 只使用线程（推荐，避免进程开销）
        - >=1: 使用指定数量的子进程，每个子进程内再用线程
        如果帧率不稳定，先尝试增加线程数；仍不稳定则增加进程数

    num_image_writer_threads_per_camera (int):
        每个相机的图像写入线程数，默认4
        线程过多可能阻塞主线程导致遥操作帧率不稳
        线程过少可能导致相机帧率低

    video_encoding_batch_size (int):
        视频批量编码的Episode数量，默认1（立即编码）
        >1时可延迟编码，节省实时采集时的CPU

    rename_map (dict[str, str]):
        观测数据的键名重映射字典
        用于覆盖默认的image和state键名
        示例: {"observation.image": "custom_image"}
    """

    # 必需参数
    repo_id: str  # 数据集标识符
    single_task: str  # 任务描述

    # 可选参数（带默认值）
    root: str | Path | None = None  # 本地存储路径
    fps: int = 30  # 采集帧率
    episode_time_s: int | float = 60  # 单个Episode时长
    reset_time_s: int | float = 60  # 重置环境时长
    num_episodes: int = 50  # Episode总数
    video: bool = True  # 是否编码为视频
    push_to_hub: bool = True  # 是否上传Hub
    private: bool = False  # 是否私有
    tags: list[str] | None = None  # Hub标签
    num_image_writer_processes: int = 0  # 写入进程数
    num_image_writer_threads_per_camera: int = 4  # 每相机线程数
    video_encoding_batch_size: int = 1  # 批量编码大小
    rename_map: dict[str, str] = field(default_factory=dict)  # 键重映射

    def __post_init__(self):
        """
        配置验证钩子 - 在对象创建后自动调用。

        验证逻辑：
            检查single_task必须提供（不能为None）
            因为任务描述对于数据集的可重复性至关重要
        """
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    """
    采集主配置 - 整合机器人和数据集配置。

    Attributes (属性说明):

    robot (RobotConfig):
        机器人配置对象
        包含机器人类型、串口路径、相机配置等
        由--robot.*命令行参数构造

    dataset (DatasetRecordConfig):
        数据集采集配置
        包含数据集标识、任务描述、存储参数等
        由--dataset.*命令行参数构造

    teleop (TeleoperatorConfig | None):
        遥操作设备配置
        None时表示不使用遥操作（纯策略模式）
        由--teleop.*命令行参数构造

    policy (PreTrainedConfig | None):
        预训练策略配置
        None时表示不使用策略（纯遥操作模式）
        由--policy.path=xxx命令行参数构造

    display_data (bool):
        是否在屏幕上显示所有相机画面，默认False
        开启时会增加CPU/GPU负载

    play_sounds (bool):
        是否使用语音合成播报事件，默认True
        会在开始采集、结束等时刻语音提示

    resume (bool):
        是否恢复已存在数据集的采集，默认False
        True时会加载已有episode并追加新数据
    """

    robot: RobotConfig  # 机器人配置（必需）
    dataset: DatasetRecordConfig  # 数据集配置（必需）
    teleop: TeleoperatorConfig | None = None  # 遥操作配置
    policy: PreTrainedConfig | None = None  # 策略配置
    display_data: bool = False  # 是否显示画面
    play_sounds: bool = True  # 是否语音播报
    resume: bool = False  # 是否恢复采集

    def __post_init__(self):
        """
        配置后处理钩子 - 处理策略路径等特殊逻辑。

        HACK说明：
            此处重新解析CLI参数以获取policy路径
            因为PolicyConfig.from_pretrained需要路径参数
            这是为了支持--policy.path=local/dir这样的命令行用法

        验证逻辑：
            teleop和policy至少有一个不为None
            否则无法生成动作，采集无意义
        """
        # 获取CLI传入的policy路径参数
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # 如果指定了路径，从预训练模型加载配置
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        # 验证：必须至少有一种动作来源
        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """
        启用从路径加载配置的字段列表。

        用途：
            告诉配置解析器可以通过--policy.path=local/dir方式
            从本地目录加载预训练策略配置
        """
        return ["policy"]


""" --------------- record_loop() 数据流图 --------------------------

    ┌─────────────────────────────────────────────────────────────┐
    │                      观测阶段 (Observation)                  │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  robot.get_observation()                                     │
    │  - 读取机器人关节位置/速度                                     │
    │  - 读取相机图像帧                                             │
    │  - 返回: Dict[设备名, 数据]                                    │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  robot_observation_processor(obs)                            │
    │  - 统一格式整理                                              │
    │  - 返回: RobotObservation (标准化观测)                         │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  build_dataset_frame(features, obs_processed, prefix="observation")  │
    │  - 按数据集特征表打包                                        │
    │  - 返回: observation_frame (可直接存入数据集)                 │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────┬──────────────────────────┐
    │   推理阶段        │    (Inference)           │
    └──────────────────┴──────────────────────────┘
                      │                        │
          ┌────────────┘                        └────────────┐
          ▼                                              ▼
    ┌─────────────────┐                    ┌─────────────────────┐
    │  Teleoperator路径 │                    │     Policy路径      │
    │                  │                    │                     │
    │ teleop.get_action│                    │  predict_action(    │
    │   → raw_action   │                    │    observation,     │
    │         │        │                    │    policy, ...)     │
    │         ▼        │                    │   → action_values   │
    │ teleop_action_   │                    │         │           │
    │ processor        │                    │         ▼           │
    │   → processed_   │                    │  (张量→字典转换)     │
    │   teleop_action  │                    │                     │
    └─────────────────┘                    └─────────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  robot_action_processor((action, obs))                      │
    │  - 动作格式转换为机器人接口格式                               │
    │  - 返回: RobotAction (机器人可执行的动作)                     │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  robot.send_action(robot_action_to_send)                   │
    │  - 动作下发到机器人硬件                                       │
    │  - 机器人开始执行本帧动作                                     │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  dataset.add_frame(frame)                                    │
    │  - 打包action_frame                                          │
    │  - 合并observation_frame + action_frame + task               │
    │  - 写入episode缓冲区                                          │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  log_rerun_data(observation, action)                         │
    │  - 发送数据到可视化服务器（如果启用display_data）               │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  busy_wait(1/fps - elapsed_time)                             │
    │  - 等待以维持目标帧率                                          │
    │  - 使用忙等待实现精确 timing                                  │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                          ( 循环下一帧 )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    核心采集循环 - 执行单次Episode的数据采集或环境重置。

    功能说明：
        1. 从机器人获取当前观测（关节状态、相机图像）
        2. 根据模式（策略/遥操作/混合）生成动作
        3. 动作下发到机器人执行
        4. 将观测-动作对存入数据集
        5. 循环执行直到达到指定时长

    分阶段计时（用于性能分析）：
        - obs: 观测获取总时间 = 硬件读取 + 后处理 + 帧打包
        - inference: 推理总时间 = 策略推理或遥操作获取动作
        - action: 动作总时间 = 动作处理 + send_action + 数据写入
        - wait: 等待总时间 = busy_wait空转等待

    传入参数详解：

    robot (Robot):
        机器人实例，用于获取观测和下发动作
        重要方法: get_observation(), send_action()
        来自 make_robot_from_config() 创建

    events (dict):
        键盘事件字典，用于控制采集流程
        键值:
            - "stop_recording": bool, True时停止全部采集
            - "exit_early": bool, True时提前结束当前episode
            - "rerecord_episode": bool, True时重新录制当前episode
        由 init_keyboard_listener() 创建

    fps (int):
        目标控制/采集帧率
        用于计算每帧的时间间隔（1/fps秒）
        同时控制busy_wait的等待时间

    teleop_action_processor (RobotProcessorPipeline):
        遥操作动作处理器流水线
        输入: tuple[RobotAction, RobotObservation] 即 (原始动作, 观测)
        输出: RobotAction (处理后的动作)
        用于统一遥操作动作格式
        来自 make_default_processors() 创建

    robot_action_processor (RobotProcessorPipeline):
        机器人动作处理器流水线
        输入: tuple[RobotAction, RobotObservation]
        输出: RobotAction (可直接发给机器人的动作)
        用于动作的最终格式化（限幅、偏移等）
        来自 make_default_processors() 创建

    robot_observation_processor (RobotProcessorPipeline):
        机器人观测处理器流水线
        输入: RobotObservation (原始观测)
        输出: RobotObservation (标准化观测)
        用于统一不同机器人的观测格式
        来自 make_default_processors() 创建

    dataset (LeRobotDataset | None):
        数据集实例，用于存储采集的帧
        None时只执行控制循环，不存储数据
        用于重置阶段（不记录数据）

    teleop (Teleoperator | list[Teleoperator] | None):
        遥操作设备实例或列表
        - 单个Teleoperator: 标准遥操作
        - list[Teleoperator]: 混合输入（如机械臂+键盘底盘）
        - None: 不使用遥操作

    policy (PreTrainedPolicy | None):
        预训练策略实例
        None时表示纯遥操作模式

    preprocessor (PolicyProcessorPipeline | None):
        策略输入预处理器流水线
        输入: dict[str, Any] (observation字典)
        输出: dict[str, Any] (预处理后的观测)
        仅在policy模式使用

    postprocessor (PolicyProcessorPipeline | None):
        策略输出后处理器流水线
        输入: PolicyAction (原始策略输出)
        输出: PolicyAction (后处理后的动作)
        仅在policy模式使用

    control_time_s (int | None):
        控制循环持续时长（秒）
        None时表示无限循环（直到events触发退出）
        通常一个episode为60秒

    single_task (str | None):
        当前任务的文字描述
        作为元数据存储在每一帧中
        示例: "Pick and place the red cube"

    display_data (bool):
        是否启用实时可视化
        True时调用log_rerun_data()发送数据到可视化服务器

    返回值: 无（数据通过dataset.add_frame()存储）

    异常说明：
        - ValueError: dataset.fps与传入fps不匹配
        - 遥操作相关异常由具体设备实现抛出

    使用示例：
        # 采集模式
        record_loop(
            robot=robot, events=events, fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=dataset, teleop=teleop, policy=None,
            control_time_s=60, single_task="抓取方块"
        )

        # 重置模式（不存数据）
        record_loop(
            robot=robot, events=events, fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=None, teleop=teleop, policy=None,
            control_time_s=60, single_task=None
        )
    """
    # ============================================================
    # 前置校验
    # ============================================================
    # 验证数据集帧率与指定帧率一致
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # ============================================================
    # 多遥操作设备解析（特殊场景：LeKiwi机器人）
    # ============================================================
    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        # 从遥操作列表中提取键盘和机械臂设备
        # KeyboardTeleop: 键盘控制（用于底盘）
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        # 机械臂遥操作设备
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,
                        so101_leader.SO101Leader,
                        koch_leader.KochLeader,
                    ),
                )
            ),
            None,
        )

        # LeKiwi特殊校验：必须是1个键盘 + 1个机械臂
        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # ============================================================
    # 策略重置（如果使用策略模式）
    # ============================================================
    # 每个episode开始时重置策略状态，确保策略从干净状态开始
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()  # 重置策略内部状态（如RNN隐藏态）
        preprocessor.reset()  # 重置预处理器状态
        postprocessor.reset()  # 重置后处理器状态

    # ============================================================
    # 主循环：控制时间戳
    # ============================================================
    timestamp = 0  # 当前Episode已执行时间
    start_episode_t = time.perf_counter()  # Episode开始时刻
    last_log_t = 0.0  # 上次日志输出时刻

    # ============================================================
    # 计时累加器（跨帧统计）
    # ============================================================
    _acc_obs = 0.0
    _acc_inference = 0.0
    _acc_action = 0.0
    _acc_wait = 0.0
    _frame_count = 0

    # 主控制循环：执行直到达到指定时长
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()  # 本次循环开始时刻

        # ============================================================
        # 事件检查：提前退出
        # ============================================================
        if events["exit_early"]:
            events["exit_early"] = False
            break  # 退出循环但不退出采集

        # ============================================================
        # 观测阶段 (Observation)
        #
        # 目的：从机器人硬件获取当前状态
        #
        # 计时子阶段：
        #   - obs_hw_time: 读取机器人硬件的时间
        #   - obs_process_time: 观测后处理的时间
        #   - obs_frame_time: 打包成数据集帧的时间
        #
        # 数据流：
        #   robot.get_observation() → obs (原始)
        #   robot_observation_processor(obs) → obs_processed (标准化)
        #   build_dataset_frame(features, obs_processed, "observation") → observation_frame (数据集格式)
        # ============================================================
        obs_start_t = time.perf_counter()

        # 读取机器人硬件观测（阻塞调用）
        # 返回原始数据：关节角度、关节速度、末端执行器状态、相机图像等
        obs_hw_start_t = time.perf_counter()
        obs = robot.get_observation()
        obs_hw_end_t = time.perf_counter()

        # 标准化观测格式（不同机器人可能有不同的数据格式）
        obs_process_start_t = time.perf_counter()
        obs_processed = robot_observation_processor(obs)
        obs_process_end_t = time.perf_counter()
        obs_end_t = time.perf_counter()

        # 打包成数据集特征格式（仅当需要存储或推理时）
        if policy is not None or dataset is not None:
            obs_frame_start_t = time.perf_counter()
            # 按数据集features定义打包观测数据
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
            obs_frame_end_t = time.perf_counter()
        else:
            # 无需打包（重置阶段跳过）
            obs_frame_start_t = obs_frame_end_t = obs_end_t

        # ============================================================
        # 推理阶段 (Inference)
        #
        # 目的：根据当前观测生成动作
        #
        # 三条路径：
        #   1. policy路径：策略推理生成动作
        #   2. teleop路径（单个）：遥操作直接获取动作
        #   3. teleop路径（多个）：多设备组合获取动作
        #
        # 计时：包含完整的动作生成，不只是模型前向
        # ============================================================
        inference_start_t = time.perf_counter()

        if policy is not None and preprocessor is not None and postprocessor is not None:
            # ========== 策略推理路径 ==========
            # 输入: observation_frame (数据集格式的观测)
            # 内部流程: tensor转换 → 预处理 → policy.select_action() → 后处理
            # 输出: action_values (numpy数组，按动作索引排列)
            action_values = predict_action(
                observation=observation_frame,  # 观测帧
                policy=policy,  # 策略实例
                device=get_safe_torch_device(policy.config.device),  # 计算设备
                preprocessor=preprocessor,  # 观测预处理器
                postprocessor=postprocessor,  # 动作后处理器
                use_amp=policy.config.use_amp,  # 是否用自动混合精度
                task=single_task,  # 任务描述（用于语言条件策略）
                robot_type=robot.robot_type,  # 机器人类型
            )

            # 将numpy数组动作值转换为{name: float}字典
            # action_values形状: [num_actions]，按索引对应到各动作维度
            action_names = dataset.features["action"]["names"]  # 动作名称列表
            act_processed_policy: RobotAction = {
                f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
            }
            inference_end_t = time.perf_counter()

        elif policy is None and isinstance(teleop, Teleoperator):
            # ========== 单遥操作路径 ==========
            # 直接从遥操作设备获取当前时刻的动作
            # 无模型推理，延迟最低
            act = teleop.get_action()

            # 遥操作动作标准化处理
            act_processed_teleop = teleop_action_processor((act, obs))
            inference_end_t = time.perf_counter()

        elif policy is None and isinstance(teleop, list):
            # ========== 多遥操作路径（LeKiwi特殊） ==========
            # 分别获取机械臂动作和键盘动作

            # 1. 获取机械臂动作
            arm_action = teleop_arm.get_action()
            # 键名加上前缀"arm_"避免与键盘动作冲突
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            # 2. 获取键盘动作
            keyboard_action = teleop_keyboard.get_action()

            # 3. 键盘动作转换为底盘动作
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            # 4. 合并动作（底盘动作可选）
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

            # 5. 遥操作动作标准化处理
            act_processed_teleop = teleop_action_processor((act, obs))
            inference_end_t = time.perf_counter()

        else:
            # ========== 无动作来源（重置阶段可能发生）==========
            # 仅打印日志并跳过本帧
            # 这在重置阶段没有遥操作设备时是正常的
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue  # 跳过本帧

        # ============================================================
        # 动作阶段 (Action)
        #
        # 目的：动作格式化 → 下发机器人 → 存储数据
        #
        # 计时子阶段：
        #   - action_process_time: 动作后处理
        #   - send_action_time: 串口下发到机器人
        #   - dataset_frame_time: 写入数据集
        #   - display_time: 发送到可视化服务器
        # ============================================================
        action_start_t = time.perf_counter()

        # 动作后处理：转换为机器人接口格式
        action_process_start_t = time.perf_counter()
        if policy is not None and act_processed_policy is not None:
            # 策略动作路径
            action_values = act_processed_policy  # 保存用于存储
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            # 遥操作动作路径
            action_values = act_processed_teleop  # 保存用于存储
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        action_process_end_t = time.perf_counter()

        # ========== 动作下发 ==========
        # 调用robot.send_action()将动作发送到机器人硬件
        # 注意：某些机器人实现内部会先读取当前位置做安全检查
        send_action_start_t = time.perf_counter()
        _sent_action = robot.send_action(robot_action_to_send)
        send_action_end_t = time.perf_counter()

        # ========== 数据存储 ==========
        if dataset is not None:
            dataset_frame_start_t = time.perf_counter()

            # 打包动作帧
            action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")

            # 合并观测帧 + 动作帧 + 任务描述
            frame = {**observation_frame, **action_frame, "task": single_task}

            # 添加到数据集episode缓冲区
            dataset.add_frame(frame)
            dataset_frame_end_t = time.perf_counter()
        else:
            # 无数据集时不打包
            dataset_frame_start_t = dataset_frame_end_t = send_action_end_t

        # ========== 可视化 ==========
        if display_data:
            display_start_t = time.perf_counter()
            log_rerun_data(observation=obs_processed, action=action_values)
            display_end_t = time.perf_counter()
        else:
            display_start_t = display_end_t = dataset_frame_end_t

        action_end_t = time.perf_counter()

        # ============================================================
        # 等待阶段 (Wait)
        #
        # 目的：维持目标帧率
        #
        # 计算本帧已消耗时间，然后用busy_wait等待剩余时间
        # busy_wait使用忙循环实现精确延时
        # ============================================================
        wait_start_t = time.perf_counter()
        dt_s = time.perf_counter() - start_loop_t  # 本帧已消耗时间
        busy_wait(1 / fps - dt_s)  # 等待直到达到1/fps秒
        wait_end_t = time.perf_counter()

        # ============================================================
        # 累加本帧各阶段时间
        # ============================================================
        _acc_obs += obs_end_t - obs_start_t
        _acc_inference += inference_end_t - inference_start_t
        _acc_action += action_end_t - action_start_t
        _acc_wait += wait_end_t - wait_start_t
        _frame_count += 1

        # ============================================================
        # 时间更新和日志输出
        # ============================================================
        timestamp = time.perf_counter() - start_episode_t  # 更新已执行时间

        # 每秒输出一次性能日志
        if timestamp - last_log_t >= 1.0:
            # 计算各阶段耗时
            total_loop_time = time.perf_counter() - start_loop_t
            obs_time = obs_end_t - obs_start_t
            obs_hw_time = obs_hw_end_t - obs_hw_start_t
            obs_process_time = obs_process_end_t - obs_process_start_t
            obs_frame_time = obs_frame_end_t - obs_frame_start_t
            inference_time = inference_end_t - inference_start_t
            action_time = action_end_t - action_start_t
            action_process_time = action_process_end_t - action_process_start_t
            send_action_time = send_action_end_t - send_action_start_t
            dataset_frame_time = dataset_frame_end_t - dataset_frame_start_t
            display_time = display_end_t - display_start_t
            wait_time = wait_end_t - wait_start_t

            # 输出性能日志
            logging.info(
                f"[Record Loop] timestamp={timestamp:.1f}s | "
                f"obs={obs_time*1000:.1f}ms | "
                # f"(hw={obs_hw_time*1000:.1f}, proc={obs_process_time*1000:.1f}, frame={obs_frame_time*1000:.1f}) | "
                f"inference={inference_time*1000:.1f}ms | "
                f"action={action_time*1000:.1f}ms | "
                # f"(proc={action_process_time*1000:.1f}, send={send_action_time*1000:.1f}, "
                # f"dataset={dataset_frame_time*1000:.1f}, display={display_time*1000:.1f}) | "
                    f"wait={wait_time*1000:.1f}ms | "
                f"total={total_loop_time*1000:.1f}ms |"
                f"fps={1/total_loop_time:.1f}"
            )
            last_log_t = timestamp

    # ============================================================
    # 返回本次 episode 的累计计时统计
    # ============================================================
    _total = _acc_obs + _acc_inference + _acc_action + _acc_wait
    if _total > 0 and _frame_count > 0:
        return {
            "frames": _frame_count,
            "episode_total_s": _total,
            "obs_s": _acc_obs,
            "inference_s": _acc_inference,
            "action_s": _acc_action,
            "wait_s": _acc_wait,
            "obs_pct": _acc_obs / _total * 100,
            "inference_pct": _acc_inference / _total * 100,
            "action_pct": _acc_action / _total * 100,
            "wait_pct": _acc_wait / _total * 100,
        }
    return None


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    """
    数据采集主入口 - 初始化组件并执行多Episode采集。

    功能说明：
        1. 初始化日志和可视化（可选）
        2. 创建机器人和遥操作设备实例
        3. 创建数据处理器流水线
        4. 创建/加载数据集
        5. 执行多次Episode采集
        6. 保存并可选上传数据集到HuggingFace Hub

    传入参数详解：

    cfg (RecordConfig):
        采集配置对象，包含：
        - cfg.robot: 机器人配置
        - cfg.dataset: 数据集配置
        - cfg.teleop: 遥操作配置（可选）
        - cfg.policy: 策略配置（可选）
        - cfg.display_data: 是否显示画面
        - cfg.play_sounds: 是否语音播报
        - cfg.resume: 是否恢复采集

    返回值：
        LeRobotDataset: 采集完成的数据集实例

    执行流程：
        1. init_logging() → 配置日志输出
        2. _init_rerun() → 初始化可视化服务器（如果启用）
        3. make_robot_from_config(cfg.robot) → 创建机器人实例
        4. make_teleoperator_from_config(cfg.teleop) → 创建遥操作实例
        5. make_default_processors() → 创建处理器流水线
        6. LeRobotDataset.create/load() → 创建或加载数据集
        7. make_policy() → 加载预训练策略（如果配置）
        8. make_pre_post_processors() → 创建策略预/后处理器
        9. robot.connect() → 连接机器人硬件
        10. teleop.connect() → 连接遥操作设备
        11. init_keyboard_listener() → 启动键盘监听线程
        12. 循环执行record_loop() → 采集多个Episode
        13. dataset.save_episode() → 保存每个Episode
        14. dataset.push_to_hub() → 上传到Hub（如果启用）

    关键数据结构：

    dataset_features (dict):
        数据集特征定义，包含observation和action的：
        - names: 各字段名称列表
        - shape: 各字段维度
        - dtype: 数据类型
        由aggregate_pipeline_dataset_features()聚合生成

    events (dict):
        键盘事件字典，通过init_keyboard_listener()创建
        包含stop_recording、exit_early、rerecord_episode等标志
        在主线程和键盘监听线程间共享

    Episode流程：
        1. record_loop(..., dataset=dataset, control_time_s=episode_time_s)
           → 执行采集，存储数据到episode缓冲区
        2. dataset.save_episode()
           → 将episode缓冲区写入磁盘
        3. record_loop(..., dataset=None, control_time_s=reset_time_s)
           → 重置阶段，不存储数据，仅控制机器人

    异常处理：
        - @safe_stop_image_writer装饰器确保即使异常退出也正确关闭图像写入器
        - 断开连接在finally块或with块外执行
    """
    # ============================================================
    # 日志和可视化初始化
    # ============================================================
    init_logging(console_level="INFO")  # 初始化日志级别
    logging.info(pformat(asdict(cfg)))  # 打印完整配置

    # 初始化rerun可视化服务器（可选）
    if cfg.display_data:
        _init_rerun(session_name="recording")

    # ============================================================
    # 创建机器人和遥操作设备
    # ============================================================
    # 从配置创建机器人实例（不连接）
    robot = make_robot_from_config(cfg.robot)

    # 从配置创建遥操作设备实例（不连接）
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # ============================================================
    # 创建数据处理器流水线
    # ============================================================
    # 创建三个默认处理器：
    # 1. teleop_action_processor: 处理遥操作动作
    # 2. robot_action_processor: 处理机器人动作
    # 3. robot_observation_processor: 处理机器人观测
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ============================================================
    # 构建数据集特征定义
    # ============================================================
    # aggregate_pipeline_dataset_features: 将处理器输出聚合为数据集特征
    # combine_feature_dicts: 合并action和observation特征
    dataset_features = combine_feature_dicts(
        # action特征：来自robot.action_features + teleop_action_processor处理
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): 未来应来自teleop或policy
            use_videos=cfg.dataset.video,
        ),
        # observation特征：来自robot.observation_features + robot_observation_processor处理
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # ============================================================
    # 创建或加载数据集
    # ============================================================
    if cfg.resume:
        # 恢复模式：加载已存在的数据集
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        # 启动图像写入器（多线程/多进程）
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # 校验数据集与机器人兼容性
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # 新建模式：创建空数据集或加载已保存的episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # ============================================================
    # 加载预训练策略（如果配置）
    # ============================================================
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None

    if cfg.policy is not None:
        # 创建策略的预处理器和后处理器
        # 用于在推理前后对观测和动作进行转换
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,  # 策略配置
            pretrained_path=cfg.policy.pretrained_path,  # 预训练模型路径
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),  # 数据集统计信息
            preprocessor_overrides={
                # 覆盖处理器配置
                "device_processor": {"device": cfg.policy.device},  # 计算设备
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},  # 观测键重映射
            },
        )

    # ============================================================
    # 连接硬件设备
    # ============================================================
    robot.connect()  # 连接机器人
    if teleop is not None:
        teleop.connect()  # 连接遥操作设备

    # 初始化键盘监听（用于控制采集流程）
    listener, events = init_keyboard_listener()

    # ============================================================
    # 视频编码管理上下文
    # ============================================================
    with VideoEncodingManager(dataset):
        recorded_episodes = 0  # 已完成的Episode计数

        # 系统调用追踪文件（用于性能分析）
        TRACE_TIMES_FILE = os.path.expanduser("~/lerobot/syscall-analyze/.lerobot_trace_times.txt")

        def write_trace_time(marker: str, timestamp: float):
            """
            写入追踪时间戳到文件。

            用于分析系统的系统调用性能。
            """
            try:
                with open(TRACE_TIMES_FILE, "a") as f:
                    f.write(f"{marker}|{timestamp}\n")
                    f.flush()
            except Exception:
                pass

        # ============================================================
        # 计时统计输出文件
        # ============================================================
        TIMING_CSV_FILE = os.path.expanduser("~/lerobot/analysis/timing_stats.csv")
        _CSV_HEADER = [
            "model", "episode_idx",
            "frames",
            "episode_total_s",
            "obs_s", "inference_s", "action_s", "wait_s",
            "obs_pct", "inference_pct", "action_pct", "wait_pct",
        ]
        # 提取模型名称（取 pretrained_path 的最后一段）
        _model_name = (
            Path(cfg.policy.pretrained_path).name
            if cfg.policy is not None and cfg.policy.pretrained_path
            else "teleop"
        )
        # 若文件不存在则写入表头；同时统计该模型已有的行数，作为 episode_idx 起始偏移
        _write_csv_header = not os.path.exists(TIMING_CSV_FILE)
        _episode_idx_offset = 0
        if not _write_csv_header:
            with open(TIMING_CSV_FILE, newline="") as _f:
                _episode_idx_offset = sum(
                    1 for row in csv.DictReader(_f) if row.get("model") == _model_name
                )
        _timing_csv_fh = open(TIMING_CSV_FILE, "a", newline="")  # noqa: SIM115
        _timing_csv_writer = csv.DictWriter(_timing_csv_fh, fieldnames=_CSV_HEADER)
        if _write_csv_header:
            _timing_csv_writer.writeheader()
            _timing_csv_fh.flush()

        # ============================================================
        # Episode采集循环
        # ============================================================
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            # 语音提示开始录制
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

            episode_start = time.perf_counter()
            write_trace_time(f"episode_start|{episode_start}", episode_start)

            # ========== 执行采集Episode ==========
            ep_timing = record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            episode_end = time.perf_counter()
            write_trace_time(f"episode_end|{episode_end}", episode_end)

            # 写入本 episode 计时统计
            if ep_timing is not None:
                _timing_csv_writer.writerow({
                    "model": _model_name,
                    "episode_idx": _episode_idx_offset + recorded_episodes,
                    **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in ep_timing.items()},
                })
                _timing_csv_fh.flush()

            # ============================================================
            # 重置环境阶段
            # ============================================================
            # 给人工操作员时间将机器人/物体恢复到初始位置
            # 最后一个episode不执行重置（因为采集结束）
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)

                # 重置阶段：不存储数据，只控制机器人
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            # ============================================================
            # Episode完成处理
            # ============================================================
            if events["rerecord_episode"]:
                # 请求重新录制：清除缓冲区，重新开始
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue  # 不增加计数，重新执行当前episode

            # 保存episode到磁盘
            dataset.save_episode()
            recorded_episodes += 1

        # 关闭计时 CSV 文件
        _timing_csv_fh.close()
        logging.info(f"[Timing] 计时统计已写入: {TIMING_CSV_FILE}")

    # ============================================================
    # 采集结束：清理资源
    # ============================================================
    log_say("Stop recording", cfg.play_sounds, blocking=True)

    # 断开设备连接
    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    # 停止键盘监听
    if not is_headless() and listener is not None:
        listener.stop()

    # ============================================================
    # 上传数据集到HuggingFace Hub
    # ============================================================
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    """
    程序入口点 - 启动数据采集。

    调用方式：
        1. 直接运行: python -m lerobot.record
        2. 通过CLI: lerobot-record
        3. 导入调用: record()
    """
    record()


if __name__ == "__main__":
    main()
