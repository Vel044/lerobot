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

"""
SO-101 Follower 机械臂驱动程序

该模块实现了 SO-101 Follower 机械臂的完整控制逻辑，包括：
- 电机通信总线管理 (FeetechMotorsBus)
- 位置标定流程 (calibrate)
- 电机参数配置 (configure)
- 状态观测读取 (get_observation)
- 动作执行发送 (send_action)

机械臂硬件架构:
    控制板 (USB转串口) ←→ Feetech STS3215 舵机 × 6
                          ├─ shoulder_pan   (ID:1)  肩部旋转
                          ├─ shoulder_lift  (ID:2)  肩部升降
                          ├─ elbow_flex     (ID:3)  肘部弯曲
                          ├─ wrist_flex     (ID:4)  腕部弯曲
                          ├─ wrist_roll     (ID:5)  腕部旋转
                          └─ gripper        (ID:6)  夹爪

通信协议: TTL 串口 (Feetech 协议, 1Mbps 默认波特率)
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_follower import SO101FollowerConfig

logger = logging.getLogger(__name__)


class SO101Follower(Robot):
    """
    SO-101 Follower 机械臂类

    继承自 Robot 基类，实现 SO-101 机械臂的具体控制逻辑。
    该机械臂由 TheRobotStudio 设计，Hugging Face 定制开发。

    核心功能:
        - connect():        连接机械臂和相机，执行标定
        - get_observation(): 读取电机位置和相机图像
        - send_action():    发送目标位置到电机
        - disconnect():     断开连接

    使用示例:
        config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            cameras={"handeye": CameraConfig(type="opencv", index_or_path=0)}
        )
        robot = SO101Follower(config)
        robot.connect()
        obs = robot.get_observation()
        robot.send_action({"shoulder_pan.pos": 0.5, ...})
        robot.disconnect()
    """

    config_class = SO101FollowerConfig
    name = "so101_follower"

    def __init__(self, config: SO101FollowerConfig):
        """
        初始化 SO-101 Follower 机械臂实例

        Args:
            config (SO101FollowerConfig): 机器人配置对象，包含串口、相机、安全限制等参数

        初始化流程:
            1. 调用父类 Robot.__init__(config) 进行基础初始化
            2. 保存配置副本到 self.config
            3. 创建 FeetechMotorsBus 通信总线实例，管理6个舵机
            4. 创建相机实例字典

        电机ID映射:
            shoulder_pan  → ID 1  (肩部水平旋转)
            shoulder_lift → ID 2  (肩部垂直升降)
            elbow_flex    → ID 3  (肘部弯曲)
            wrist_flex   → ID 4  (腕部弯曲)
            wrist_roll   → ID 5  (腕部旋转)
            gripper      → ID 6  (夹爪)

        注意:
            - 前5个关节使用相同的 norm_mode（取决于 use_degrees 配置）
            - 夹爪使用独立的 MotorNormMode.RANGE_0_100（0-100范围，对应夹爪开合）
        """
        super().__init__(config)
        self.config = config

        # 根据配置决定是否使用角度制
        # use_degrees=True  → MotorNormMode.DEGREES (角度值，如 0-360°)
        # use_degrees=False → MotorNormMode.RANGE_M100_100 (原始编码值，-100 到 100)
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 创建 Feetech 舵机总线
        # FeetechMotorsBus 负责:
        #   - 串口通信管理（连接、断开、读写）
        #   - 舵机协议封装（sync_read, sync_write）
        #   - 标定数据管理（读取/写入校准参数）
        self.bus = FeetechMotorsBus(
            port=self.config.port,      # 串口路径，如 "/dev/ttyACM0"
            motors={                    # 电机名称 → Motor(id, 型号, 归一化模式)
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # 夹爪专用范围 0-100
            },
            calibration=self.calibration,  # 从父类 Robot 继承的标定数据
        )

        # 创建相机实例
        # make_cameras_from_configs() 根据 config.cameras 字典创建对应的相机对象
        # 返回类型: dict[str, CameraBase]，键为相机名称（如 "handeye", "fixed"）
        self.cameras = make_cameras_from_configs(config.cameras)

    # =========================================================================
    # 属性: 特征定义 (用于数据集和策略)
    # =========================================================================

    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        电机特征定义（内部使用）

        Returns:
            dict[str, type]: 键为 "{motor_name}.pos"，值为 float 类型
            示例: {"shoulder_pan.pos": float, "shoulder_lift.pos": float, ...}
        """
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """
        相机特征定义（内部使用）

        Returns:
            dict[str, tuple]: 键为相机名称，值为 (height, width, channels) 元组
            示例: {"handeye": (360, 640, 3), "fixed": (360, 640, 3)}
        """
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        观测特征定义（数据集接口）

        缓存属性，首次访问时计算并缓存。
        定义 get_observation() 返回数据的结构和类型。

        Returns:
            dict[str, type | tuple]: 合并电机和相机的特征定义
            示例:
                {
                    "shoulder_pan.pos": float,
                    "shoulder_lift.pos": float,
                    "elbow_flex.pos": float,
                    "wrist_flex.pos": float,
                    "wrist_roll.pos": float,
                    "gripper.pos": float,
                    "handeye": (360, 640, 3),
                    "fixed": (360, 640, 3),
                }
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        动作特征定义（策略接口）

        缓存属性，首次访问时计算并缓存。
        定义 send_action() 期望的输入数据的结构。

        Returns:
            dict[str, type]: 键为 "{motor_name}.pos"，值为 float 类型
            注意: 动作特征只包含电机位置，不包含相机（策略输出的是电机动作）
        """
        return self._motors_ft

    # =========================================================================
    # 属性: 连接状态
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """
        检查机械臂是否已连接

        Returns:
            bool: True 表示总线和所有相机都连接成功，False 表示任一组件未连接
        """
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """
        检查机械臂是否已标定

        Returns:
            bool: True 表示标定数据已写入舵机，False 表示未标定或标定数据不匹配
        """
        return self.bus.is_calibrated

    # =========================================================================
    # 方法: 连接与断开
    # =========================================================================

    def connect(self, calibrate: bool = True) -> None:
        """
        连接机械臂和相机

        连接流程:
            1. 检查是否已连接（避免重复连接）
            2. 连接舵机总线 (FeetechMotorsBus.connect)
            3. 如果未标定且 calibrate=True，执行标定流程
            4. 连接所有相机
            5. 调用 configure() 配置电机参数
            6. 记录日志

        Args:
            calibrate (bool): 是否执行标定流程
                - True (默认): 如果没有标定数据或标定数据不匹配，执行标定
                - False: 跳过标定（适用于已有标定数据的场景）

        Raises:
            DeviceAlreadyConnectedError: 如果已经连接则抛出异常

        注意:
            标定过程中需要手动移动机械臂，请确保：
            - 连接时机械臂处于静止状态
            - 可以安全地禁用扭矩进行标定
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接舵机通信总线
        self.bus.connect()

        # 标定检查: 如果没有标定数据或标定数据与电机不匹配
        if not self.is_calibrated and calibrate:
            if not self.calibration:
                logger.info("No calibration file found, running calibration")
            else:
                logger.info("Mismatch between calibration values in the motor and the calibration file")
            self.calibrate()

        # 连接所有配置的相机
        for cam in self.cameras.values():
            cam.connect()

        # 配置电机参数（PID、扭矩限制等）
        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """
        断开机械臂连接

        断开流程:
            1. 检查连接状态
            2. 断开舵机总线（根据配置可能禁用扭矩）
            3. 断开所有相机
            4. 记录日志

        Args:
            无

        Raises:
            DeviceNotConnectedError: 如果未连接则抛出异常

        安全提示:
            - 断开前确保机械臂处于安全位置
            - disable_torque_on_disconnect=True 时，舵机会释放扭矩，
              机械臂可能会因重力下落
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开舵机总线
        # 参数 disable_torque_on_disconnect 来自 config（默认 True）
        # True  → 断开时禁用所有舵机扭矩（安全）
        # False → 保持扭矩（可能失控，仅调试用）
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开所有相机连接
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # =========================================================================
    # 方法: 标定
    # =========================================================================

    def calibrate(self) -> None:
        """
        执行机械臂标定流程

        标定目的: 确定每个电机Encoder的零位偏移（homing_offset）和运动范围（range）

        标定流程:
            1. 检查是否有预存标定数据
               - 有 → 询问用户是使用文件还是重新标定
               - 无 → 直接进入标定流程
            2. 禁用扭矩，允许手动移动
            3. 设置所有电机为位置模式
            4. 用户将机械臂移动到中间位置 → 记录半圈回零偏移
            5. 用户顺次移动每个关节通过整个行程 → 记录最小/最大位置
            6. 为夹爪添加额外的负偏移（使夹爪在关闭时更紧）
            7. 保存标定数据到文件

        注意:
            - 标定是交互式过程，需要用户手动操作机械臂
            - 标定过程中扭矩被禁用，可以手动移动机械臂
            - 标定失败或中断可能导致机械臂无法正常工作

        标定数据存储位置:
            self.calibration_fpath (继承自 Robot 基类)
            格式: YAML 文件，包含每个电机的 id, drive_mode, homing_offset, range_min, range_max
        """
        # 检查是否有预存标定数据
        if self.calibration:
            # 有标定数据，询问用户选择
            # input() 会阻塞等待用户输入
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                # 用户按回车，使用预存标定数据
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)  # 将标定数据写入电机 EEPROM
                return

        # 开始新的标定流程
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()  # 禁用扭矩，允许手动移动

        # 设置所有电机为位置控制模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # 步骤1: 半圈回零
        # 用户将机械臂移动到行程中间位置
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        # 自动检测Encoder的零位偏移
        homing_offsets = self.bus.set_half_turn_homings()

        # 步骤2: 记录每个关节的运动范围
        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        # 用户顺次移动每个关节通过整个行程，系统自动记录最小/最大值
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        # 步骤3: 夹爪特殊处理
        # 添加负偏移使夹爪在关闭时更紧（仅当检测到夹爪范围时）
        if 'gripper' in range_mins:
            # 计算偏移量: 3.5度对应的编码值
            gripper_adjust_offset_deg = 3.5
            # 获取编码表中的 Homing_Offset 位数
            encoding_table = self.bus.model_encoding_table.get(self.bus.motors['gripper'].model, {})
            homing_offset_bits = encoding_table.get("Homing_Offset")
            # 计算电机全范围（2^(bits+1)）
            full_range = 1 << (homing_offset_bits + 1)
            # 将角度转换为编码值（负号表示向更紧的方向调整）
            gripper_adjust_offset = -int(full_range * gripper_adjust_offset_deg / 360)
            # 调整最小范围
            original_min = range_mins['gripper']
            adjusted_min = original_min + gripper_adjust_offset
            print(f"Gripper range adjusted: original min={original_min} -> adjusted min={adjusted_min} (offset={gripper_adjust_offset})")
            range_mins['gripper'] = adjusted_min

        # 步骤4: 构建标定数据字典
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,                    # 电机ID (1-6)
                drive_mode=0,               # 驱动模式 (0=位置模式)
                homing_offset=homing_offsets[motor],  # 零位偏移
                range_min=range_mins[motor],  # 最小位置
                range_max=range_maxes[motor],  # 最大位置
            )

        # 步骤5: 写入电机并保存到文件
        self.bus.write_calibration(self.calibration)
        self._save_calibration()  # 继承自 Robot 基类，保存到 YAML 文件
        print("Calibration saved to", self.calibration_fpath)

    # =========================================================================
    # 方法: 电机配置
    # =========================================================================

    def configure(self) -> None:
        """
        配置电机运行参数

        配置内容:
            1. 配置电机基本参数 (configure_motors)
            2. 设置所有电机为位置模式 (Operating_Mode = POSITION)
            3. 调整 PID 参数降低抖动:
               - P_Coefficient: 32 → 16 (降低比例增益，减少抖动)
               - I_Coefficient: 0 (积分增益，保持为0)
               - D_Coefficient: 32 → 0 (微分增益，设为0避免振荡)
            4. 夹爪特殊配置:
               - Max_Torque_Limit: 500 (50%最大扭矩，保护夹爪电机)
               - Protection_Current: 250 (50%最大电流)
               - Overload_Torque: 25 (过载扭矩阈值25%)

        注意:
            - 所有配置在扭矩禁用状态下进行 (torque_disabled context manager)
            - 配置参数存储在电机 EEPROM，断电后保持
            - 不当的 PID 参数可能导致机械臂抖动或响应迟缓
        """
        # torque_disabled() 是一个上下文管理器，块内禁用所有电机扭矩
        with self.bus.torque_disabled():
            # 配置电机基本参数（从 EEPROM 读取校准值等）
            self.bus.configure_motors()

            # 设置每个电机的控制参数
            for motor in self.bus.motors:
                # 设置为位置控制模式
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                # 降低 P_Coefficient 减少抖动（默认32，过高会导致抖动）
                self.bus.write("P_Coefficient", motor, 16)

                # I 和 D 系数设为默认值
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 0)

                # 夹爪特殊配置（电流保护和扭矩限制）
                if motor == "gripper":
                    # 最大扭矩限制: 500 (约为最大扭矩的50%，防止烧毁电机)
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    # 保护电流: 250 (约为最大电流的50%)
                    self.bus.write("Protection_Current", motor, 250)
                    # 过载扭矩: 25 (过载时使用的扭矩百分比)
                    self.bus.write("Overload_Torque", motor, 25)

    # =========================================================================
    # 方法: 电机设置 (首次使用时)
    # =========================================================================

    def setup_motors(self) -> None:
        """
        设置电机ID（首次使用或更换电机后执行）

        交互式流程:
            1. 检查总线上是否有意外电机
            2. 依次为每个电机设置ID:
               - 断开其他电机，只连接当前要设置的电机
               - 自动扫描并设置ID
               - 验证设置成功
            3. 重复直到所有6个电机设置完成

        注意:
            - 这是工厂设置流程，正常使用时无需执行
            - 设置过程中需要断开电机，只保留要设置的电机
            - 如果电机已经有ID，会跳过或报错

        电机ID分配:
            shoulder_pan  → ID 1
            shoulder_lift → ID 2
            elbow_flex    → ID 3
            wrist_flex    → ID 4
            wrist_roll    → ID 5
            gripper       → ID 6
        """
        expected_ids = [1]  # 从ID 1开始

        # 检查是否有意外电机在总线上
        succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)
        if not succ:
            input(msg)
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)

        # 逆序设置电机（从最后一个开始，因为电机链条的连接顺序）
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor ONLY and press enter.")
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)
            if not succ:
                input(msg)
                succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)

            # 设置电机ID
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
            expected_ids.append(self.bus.motors[motor].id)

    def _check_unexpected_motors_on_bus(
        self, expected_ids: list[int], raise_on_error: bool = True
    ) -> tuple[bool, str]:
        """
        检查总线上的意外电机

        用于 setup_motors() 流程，检测是否有不在预期列表中的电机。

        Args:
            expected_ids (list[int]): 预期存在的电机ID列表
                示例: [1] 表示当前应该只有ID=1的电机
            raise_on_error (bool): 是否在发现意外电机时抛出异常
                - True: 抛出 RuntimeError
                - False: 返回 (False, error_message)

        Returns:
            tuple[bool, str]:
                - (True, "OK"): 未发现意外电机
                - (False, "Please unplug..."): 发现意外电机或未找到电机

        扫描逻辑:
            1. 确保总线已连接（handshake=False 跳过握手）
            2. 在当前波特率下扫描电机
            3. 如果扫描失败，尝试其他波特率
            4. 恢复原始波特率
            5. 对比发现的电机ID与预期ID

        波特率扫描:
            Feetech 电机支持多种波特率，扫描时会尝试:
            [9600, 19200, 57600, 115200, 500000, 1000000, ...]
            确保能找到正确配置的电机。
        """
        # 确保总线已连接（handshake=False 跳过固件版本检查）
        if not self.bus.is_connected:
            self.bus.connect(handshake=False)

        # 获取当前波特率并扫描
        current_baudrate = self.bus.get_baudrate()
        self.bus.set_baudrate(current_baudrate)  # 确保使用当前波特率

        # 在当前波特率下扫描所有电机
        found_motors = self.bus.broadcast_ping(raise_on_error=False)

        # 如果当前波特率扫描失败，尝试其他波特率
        if found_motors is None:
            for baudrate in self.bus.available_baudrates:
                if baudrate == current_baudrate:
                    continue  # 跳过已尝试的波特率
                self.bus.set_baudrate(baudrate)
                found_motors = self.bus.broadcast_ping(raise_on_error=False)
                if found_motors is not None:
                    break  # 找到电机，停止扫描

        # 恢复原始波特率
        self.bus.set_baudrate(current_baudrate)

        # 分析扫描结果
        if found_motors is not None:
            # 发现电机，检查是否有意外电机
            unexpected_motors = [mid for mid in found_motors.keys() if mid not in expected_ids]

            if unexpected_motors:
                unexpected_motors_str = ", ".join(map(str, sorted(unexpected_motors)))
                if raise_on_error:
                    raise RuntimeError(
                        f"There are unexpected motors on the bus: {unexpected_motors_str}. "
                        f"Seems this arm has been setup before, not necessary to setup again."
                    )
                else:
                    logger.warning(f"There are unexpected motors on the bus: {unexpected_motors_str}.")
                    return False, "Please unplug the last motor and press ENTER to try again."
            return True, "OK"

        return False, "No motors found on the bus, please connect the arm and press ENTER to try again."

    # =========================================================================
    # 方法: 观测与动作
    # =========================================================================

    def get_observation(self) -> dict[str, Any]:
        """
        获取机器人当前观测数据

        观测数据包含:
            1. 所有电机的当前位置（本体感受）
            2. 所有相机的最新图像（视觉感知）

        Returns:
            dict[str, Any]: 观测字典，包含电机位置和相机图像
                键值对:
                    "{motor_name}.pos": float,  # 电机位置值
                        示例: {"shoulder_pan.pos": 0.5, "shoulder_lift.pos": 1.2, ...}
                    "{camera_name}": np.ndarray,  # 相机图像 (BGR格式)
                        示例: {"handeye": array(...), "fixed": array(...)}
                图像格式: numpy.ndarray, shape=(height, width, 3), dtype=uint8

        Raises:
            DeviceNotConnectedError: 如果机器人未连接

        性能考量:
            - 电机位置读取: ~1-2ms (sync_read 是阻塞式串口读取)
            - 相机读取: 0.1-1ms (async_read 是非阻塞式，可能返回缓存图像)
            - 总耗时: 约 2-5ms（取决于相机数量和图像分辨率）

        时序图:
            start ──► sync_read("Present_Position") ──► 读取6个电机位置
                     │                                  (~1-2ms)
                     ▼
                     for cam in cameras:
                         cam.async_read() ──► 读取相机图像
                                            (~0.1-1ms each)
                     ▼
                   return obs_dict
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # --- 读取电机位置 ---
        start = time.perf_counter()
        # sync_read() 是阻塞式读取，等待舵机响应
        # 返回格式: dict[motor_name] = position_value
        obs_dict = self.bus.sync_read("Present_Position")
        # 转换为标准格式: "{motor_name}.pos" → value
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        # logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # --- 读取相机图像 ---
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            # async_read() 是非阻塞式读取，可能返回最近缓存的图像
            # 返回格式: np.ndarray (BGR格式, shape: [H, W, 3])
            obs_dict[cam_key] = cam.async_read(timeout_ms=3000)
            dt_ms = (time.perf_counter() - start) * 1e3
            # logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作到机械臂

        将目标关节位置发送给机械臂，控制其移动到指定位置。

        Args:
            action (dict[str, Any]): 目标动作字典
                键格式: "{motor_name}.pos"
                值格式: float (目标位置值)
                示例:
                    {
                        "shoulder_pan.pos": 0.5,
                        "shoulder_lift.pos": 1.2,
                        "elbow_flex.pos": -0.3,
                        "wrist_flex.pos": 0.8,
                        "wrist_roll.pos": 0.1,
                        "gripper.pos": 50.0
                    }

        Returns:
            dict[str, Any]: 实际发送的动作字典
                注意: 由于 max_relative_target 限制，返回值可能与输入不同
                格式同输入: {"{motor_name}.pos": float, ...}

        Raises:
            DeviceNotConnectedError: 如果机器人未连接

        处理流程:
            1. 解析动作字典，提取电机名称和目标位置
            2. 安全检查（如果配置了 max_relative_target）:
               - 读取当前位置
               - 计算目标与当前的差值
               - 如果差值超过限制，修正目标位置
            3. 发送目标位置到电机
            4. 返回实际发送的动作

        安全机制 (max_relative_target):
            目的: 防止机械臂碰撞或损伤

            原理:
                如果 |goal_pos - current_pos| > max_relative_target
                则将目标限制在: current_pos ± max_relative_target

            性能代价:
                需要额外调用 sync_read("Present_Position")
                导致控制频率下降约 10-20%

        时序图 (无安全限制):
            start ──► 解析action字典 ──► sync_write("Goal_Position") ──► return
                       (~0.1ms)              (~1-2ms)

        时序图 (有安全限制):
            start ──► 解析action ──► sync_read("Present_Position")
                       (~0.1ms)           (~1-2ms, 额外开销)
                                      ▼
                                   计算安全目标位置
                                      ▼
                                   sync_write("Goal_Position")
                                      (~1-2ms)
                                      ▼
                                   return
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        total_start = time.perf_counter()

        # --- 解析动作字典 ---
        # 输入: {"shoulder_pan.pos": 0.5, ...} → 输出: {"shoulder_pan": 0.5, ...}
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }
        preprocess_ms = (time.perf_counter() - total_start) * 1e3

        # --- 安全限制检查 ---
        read_ms = 0.0
        if self.config.max_relative_target is not None:
            # 需要读取当前位置来计算相对移动距离
            read_start = time.perf_counter()
            present_pos = self.bus.sync_read("Present_Position")  # 阻塞读取 (~1-2ms)
            read_ms = (time.perf_counter() - read_start) * 1e3

            # 构建 (目标位置, 当前位置) 元组字典
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}

            # 调用安全检查函数，限制过大的目标位置
            # ensure_safe_goal_position() 会将差值限制在 max_relative_target 内
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # --- 发送目标位置 ---
        write_start = time.perf_counter()
        self.bus.sync_write("Goal_Position", goal_pos)  # 阻塞写入 (~1-2ms)
        write_ms = (time.perf_counter() - write_start) * 1e3
        total_ms = (time.perf_counter() - total_start) * 1e3

        # 调试日志（已注释）
        # logger.debug(
        #     f"{self} send_action total={total_ms:.1f}ms "
        #     f"(pre={preprocess_ms:.3f}, safety_read={read_ms:.1f}, write={write_ms:.1f})"
        # )

        # 返回实际发送的动作（可能被安全限制修改）
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
