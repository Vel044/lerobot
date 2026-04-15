#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    """
    SO-101 Follower 机械臂配置类

    继承自 RobotConfig，通过 register_subclass 装饰器注册到机器人配置注册表。
    当命令行指定 --robot.type=so101_follower 时，会实例化此类。

    实例化示例:
        config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            disable_torque_on_disconnect=True,
            max_relative_target=0.1,
            cameras={"handeye": CameraConfig(type="opencv", index_or_path=0, width=640, height=360, fps=30)},
            use_degrees=False
        )
    """

    # =========================================================================
    # 串口通信配置
    # =========================================================================

    #: str: 机械臂控制板连接的串口路径
    #:   - Linux: 如 "/dev/ttyACM0", "/dev/ttyUSB0"
    #:   - macOS: 如 "/dev/cu.usbmodem14201"
    #:   - Windows: 如 "COM3"
    #: 示例值: "/dev/ttyACM0"
    port: str

    # =========================================================================
    # 安全保护配置
    # =========================================================================

    #: bool: 断开连接时是否自动禁用舵机扭矩
    #:   - True (默认): 断开连接时自动 disable_torque，保护机械臂防止意外移动
    #:   - False: 断开连接时保持扭矩，可能导致机械臂失控
    #: 建议: 日常使用设为 True，调试时可设为 False
    disable_torque_on_disconnect: bool = True

    #: float | dict[str, float] | None: 单次动作的相对位置目标限制（安全保护）
    #:   限制机械臂单次移动的最大距离，防止碰撞或损伤
    #:
    #:   - None: 不限制（危险！仅推荐在仿真环境使用）
    #:   - float: 所有电机使用相同的限制值（如 0.1 表示限制为目标-当前位置差 <= 0.1）
    #:   - dict[str, float]: 电机名称映射到各自的限制值
    #:     格式: {"shoulder_pan": 0.1, "shoulder_lift": 0.15, ...}
    #:
    #:   工作原理:
    #:     每次发送动作前，比较目标位置与当前位置的差值。
    #:     如果差值超过 max_relative_target，则将目标位置限制在 安全范围内。
    #:     例如: 当前肩部角度=0.5，目标=0.8，max_relative_target=0.1
    #:           → 实际发送的目标会被限制为 0.5+0.1=0.6 或 0.5-0.1=0.4（取决于方向）
    #:
    #: 注意: 设为非 None 值会导致 send_action() 额外调用一次 sync_read("Present_Position")
    #:       来获取当前位置，这会影响控制频率（约降低 10-20%）
    max_relative_target: float | dict[str, float] | None = None

    # =========================================================================
    # 相机配置
    # =========================================================================

    #: dict[str, CameraConfig]: 相机配置字典
    #:   键: 相机名称（字符串），用于在观测数据中标识相机
    #:   值: CameraConfig 对象，包含相机类型和参数
    #:
    #: 典型配置示例:
    #:   cameras = {
    #:       "handeye": CameraConfig(     # 手眼相机 - 安装在机械臂末端
    #:           type="opencv",
    #:           index_or_path=0,         # 第一个USB相机
    #:           width=640,
    #:           height=360,
    #:           fps=30
    #:       ),
    #:       "fixed": CameraConfig(       # 固定相机 - 安装在工作台旁
    #:           type="opencv",
    #:           index_or_path=2,
    #:           width=640,
    #:           height=360,
    #:           fps=30
    #:       )
    #:   }
    #:
    #: 相机图像在 get_observation() 中的格式:
    #:   obs_dict["handeye"] = np.ndarray (shape: [height, width, 3], BGR格式)
    #:   obs_dict["fixed"] = np.ndarray (shape: [height, width, 3], BGR格式)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # =========================================================================
    # 兼容性配置
    # =========================================================================

    #: bool: 是否使用角度制（而非默认的原始编码值）
    #:   - False (默认): 使用 Feetech 原始编码值（范围 -100 到 100）
    #:   - True: 使用角度制（度数），需配合 MotorNormMode.DEGREES 使用
    #:
    #: 用途: 主要用于与旧版策略/数据集的向后兼容
    #: 注意: 设为 True 时，所有动作/观测值的单位为度，而非原始编码值
    use_degrees: bool = False
