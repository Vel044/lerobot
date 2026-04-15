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

# TODO(aliberts): Should we implement FastSyncRead/Write?
# https://github.com/ROBOTIS-GIT/DynamixelSDK/pull/643
# https://github.com/ROBOTIS-GIT/DynamixelSDK/releases/tag/3.8.2
# https://emanual.robotis.com/docs/en/dxl/protocol2/#fast-sync-read-0x8a
# -> Need to check compatibility across models

import logging
from copy import deepcopy
from enum import Enum

from lerobot.utils.encoding_utils import decode_twos_complement, encode_twos_complement

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
from .tables import (
    AVAILABLE_BAUDRATES,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION,
)

# Dynamixel 协议版本号，当前使用 2.0 协议
PROTOCOL_VERSION = 2.0
# 默认波特率 1_000_000 bps（1 Mbps），这是高速通信的常用配置
DEFAULT_BAUDRATE = 1_000_000
# 默认通信超时时间 1000 毫秒（1秒）
DEFAULT_TIMEOUT_MS = 1000

# 需要进行归一化处理的寄存器列表。
# 归一化：将电机原始的步进值转换为用户友好的相对值（如-100到100或角度）
# 这些寄存器在读写时会自动在原始值和归一化值之间转换
NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    """
    电机运行模式枚举。

    控制模式决定了电机如何响应控制指令：
    - CURRENT (0): 电流/扭矩控制模式，适用于夹爪或只需要扭矩控制的系统
    - VELOCITY (1): 速度控制模式，类似于传统的轮子模式（无限转动）
    - POSITION (3): 位置控制模式，关节模式，位置范围受限于最小/最大位置限制
    - EXTENDED_POSITION (4): 扩展位置控制模式，支持多圈（±512圈），适用于手腕或多圈应用
    - CURRENT_POSITION (5): 电流-位置混合控制模式，同时控制位置和扭矩
    - PWM (16): PWM直接输出控制模式（电压控制模式）
    """
    # DYNAMIXEL only controls current(torque) regardless of speed and position. This mode is ideal for a
    # gripper or a system that only uses current(torque) control or a system that has additional
    # velocity/position controllers.
    CURRENT = 0

    # This mode controls velocity. This mode is identical to the Wheel Mode(endless) from existing DYNAMIXEL.
    # This mode is ideal for wheel-type robots.
    VELOCITY = 1

    # This mode controls position. This mode is identical to the Joint Mode from existing DYNAMIXEL. Operating
    # position range is limited by the Max Position Limit(48) and the Min Position Limit(52). This mode is
    # ideal for articulated robots that each joint rotates less than 360 degrees.
    POSITION = 3

    # This mode controls position. This mode is identical to the Multi-turn Position Control from existing
    # DYNAMIXEL. 512 turns are supported(-256[rev] ~ 256[rev]). This mode is ideal for multi-turn wrists or
    # conveyer systems or a system that requires an additional reduction gear. Note that Max Position
    # Limit(48), Min Position Limit(52) are not used on Extended Position Control Mode.
    EXTENDED_POSITION = 4

    # This mode controls both position and current(torque). Up to 512 turns are supported (-256[rev] ~
    # 256[rev]). This mode is ideal for a system that requires both position and current control such as
    # articulated robots or grippers.
    CURRENT_POSITION = 5

    # This mode directly controls PWM output. (Voltage Control Mode)
    PWM = 16


class DriveMode(Enum):
    """
    驱动方向模式枚举。

    定义电机的旋转方向：
    - NON_INVERTED (0): 正常方向，不反转
    - INVERTED (1): 反转方向，用于需要反向安装电机的场景
    """
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    """
    扭矩使能模式枚举。

    控制电机是否输出扭矩：
    - ENABLED (1): 使能扭矩，电机可以响应目标位置/速度等指令
    - DISABLED (0): 禁用扭矩，电机可以自由转动，用于手动示教或校准
    """
    ENABLED = 1
    DISABLED = 0


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    """
    将一个整数拆分为多个单字节的列表。

    这是底层的字节序处理函数，用于将多字节数据打包成通信协议需要的格式。

    参数:
        value (int): 要拆分的目标整数
        length (int): 目标字节数（1、2 或 4）

    返回:
        list[int]: 单字节整数列表，按小端序（little-endian）排列

    示例:
        >>> _split_into_byte_chunks(0x1234, 2)
        [0x34, 0x12]  # 小端序：低字节在前
        >>> _split_into_byte_chunks(0x12345678, 4)
        [0x78, 0x56, 0x34, 0x12]
    """
    import dynamixel_sdk as dxl

    if length == 1:
        data = [value]
    elif length == 2:
        # DXL_LOBYTE 提取低8位，DXL_HIBYTE 提取高8位
        data = [dxl.DXL_LOBYTE(value), dxl.DXL_HIBYTE(value)]
    elif length == 4:
        # DXL_LOWORD 提取低16位，DXL_HIWORD 提取高16位
        data = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
        ]
    return data


class DynamixelMotorsBus(MotorsBus):
    """
    Dynamixel 总线实现类。

    继承自 MotorsBus 抽象基类，通过 dynamixel_sdk 与 Dynamixel 系列电机通信。
    支持同步读写、批量操作、校准等功能。

    主要特性：
    - 使用 Protocol 2.0 通信协议
    - 支持广播 ping 发现所有电机
    - 支持同步读写多个电机
    - 内置校准和归一化处理
    - 支持扭矩使能/禁用控制

    属性:
        apply_drive_mode (bool): 是否应用驱动模式（方向反转）
        available_baudrates (list[int]): 支持的波特率列表
        default_baudrate (int): 默认波特率
        default_timeout (int): 默认超时时间（毫秒）
        model_ctrl_table (dict): 各型号的控制表（寄存器地址映射）
        model_encoding_table (dict): 各型号的编码表（符号位处理）
        model_number_table (dict): 型号名称到型号编号的映射
        model_resolution_table (dict): 各型号的编码器分辨率
    """

    # 是否应用驱动模式（方向反转）。Dynamixel 不需要此功能，设为 False
    apply_drive_mode = False
    # 支持的波特率列表（用于端口扫描）
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    # 默认波特率 1 Mbps
    default_baudrate = DEFAULT_BAUDRATE
    # 默认超时 1000ms
    default_timeout = DEFAULT_TIMEOUT_MS
    # 型号到波特率先表的映射
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    # 型号到控制表的映射
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    # 型号到编码表的映射
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    # 型号名称到编号的映射
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    # 型号到分辨率的映射
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    # 需要归一化的寄存器列表
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        """
        初始化 Dynamixel 总线。

        参数:
            port (str): 串口设备路径，例如 "/dev/ttyUSB0" 或 "COM3"
            motors (dict[str, Motor]): 电机字典，键为电机名称，值为 Motor 数据类
                示例: {"shoulder_pan": Motor(id=1, model="xl430-w250", norm_mode=MotorNormMode.RANGE_0_100)}
            calibration (dict[str, MotorCalibration] | None, optional): 校准数据字典。
                包含每个电机的校准参数（最小位置、最大位置、零位偏移等）。
                如不提供，则使用空字典，后续需要手动校准。
        """
        super().__init__(port, motors, calibration)
        import dynamixel_sdk as dxl

        # 创建端口处理器，负责串口通信
        self.port_handler = dxl.PortHandler(self.port)
        # 创建数据包处理器，负责协议编解码
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        # 创建同步读取器，用于批量读取多个电机的同一寄存器
        self.sync_reader = dxl.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        # 创建同步写入器，用于批量向多个电机写入同一寄存器
        self.sync_writer = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        # 通信成功的返回值常量
        self._comm_success = dxl.COMM_SUCCESS
        # 无错误的返回值常量
        self._no_error = 0x00

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        """
        校验协议兼容性。

        Dynamixel 使用 Protocol 2.0，不存在协议兼容性问题，此方法为空实现。

        参数:
            instruction_name (str): 指令名称（用于错误信息）
        """
        pass

    def _handshake(self) -> None:
        """
        执行握手/初始化检测。

        验证总线上的电机是否与预期配置匹配。
        调用 _assert_motors_exist() 检查所有注册的电机都能被找到。
        """
        self._assert_motors_exist()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """
        在指定波特率下查找单个电机的 ID 和型号。

        通过轮询不同波特率来定位电机，确保找到正确的电机。

        参数:
            motor (str): 电机名称（用于从 self.motors 获取预期型号）
            initial_baudrate (int | None, optional): 初始波特率。如果提供，则只在此波特率下搜索；
                否则按 model_baudrate_table 中定义的波特率列表依次搜索

        返回:
            tuple[int, int]: 找到的 (baudrate, motor_id)

        异常:
            RuntimeError: 找不到电机或型号不匹配
        """
        model = self.motors[motor].model
        # 确定要搜索的波特率列表
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                expected_model_nb = self.model_number_table[model]
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(self, return_delay_time=0) -> None:
        """
        配置电机参数。

        向所有电机写入推荐配置参数。

        参数:
            return_delay_time (int, optional): 响应延迟时间。默认 0（最小2微秒）。
                Dynamixel 默认是 500 微秒（对应值为 250），设为 0 可减少通信延迟。
        """
        # By default, Dynamixel motors have a 500µs delay response time (corresponding to a value of 250 on
        # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)

    @property
    def is_calibrated(self) -> bool:
        """
        校准状态属性。

        返回:
            bool: True 如果缓存的校准数据与电机当前读取的校准数据一致
        """
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """
        从电机读取校准参数。

        读取每个电机的以下校准数据：
        - Homing_Offset（零位偏移）
        - Min_Position_Limit（最小位置限制）
        - Max_Position_Limit（最大位置限制）
        - Drive_Mode（驱动模式）

        返回:
            dict[str, MotorCalibration]: 键为电机名称，值为 MotorCalibration 数据类
        """
        # 同步读取多个寄存器的数据
        offsets = self.sync_read("Homing_Offset", normalize=False)
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        drive_modes = self.sync_read("Drive_Mode", normalize=False)

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """
        向电机写入校准参数。

        参数:
            calibration_dict (dict[str, MotorCalibration]): 校准参数字典
            cache (bool, optional): 是否将校准数据缓存到 self.calibration。默认为 True
        """
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        禁用指定电机的扭矩输出。

        禁用扭矩后，电机可以自由转动（适用于手动示教或校准）。
        注意：禁用扭矩后才能写入 EEPROM/EPROM 区域（如 ID、波特率等）。

        参数:
            motors (str | list[str] | None, optional): 目标电机。
                - None: 所有电机
                - str: 单个电机名称
                - list[str]: 多个电机名称列表
            num_retry (int, optional): 通信失败时的重试次数。默认为 0
        """
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        """
        内部方法：禁用单个电机的扭矩（通过电机 ID 和型号）。

        参数:
            motor_id (int): 电机 ID
            model (str): 电机型号
            num_retry (int, optional): 重试次数
        """
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        使能指定电机的扭矩输出。

        参数:
            motors (str | list[str] | None, optional): 目标电机（同 disable_torque）
            num_retry (int, optional): 重试次数
        """
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        对寄存器值进行符号位编码。

        某些寄存器使用有符号数值表示（如位置、速度等）。
        此方法将对这些值进行符号位编码以适应电机协议。

        参数:
            data_name (str): 寄存器名称（如 "Goal_Velocity"）
            ids_values (dict[int, int]): 电机 ID 到原始值的映射

        返回:
            dict[int, int]: 编码后的值（符号位已处理）
        """
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                # 使用二进制补码编码
                ids_values[id_] = encode_twos_complement(ids_values[id_], n_bytes)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        对寄存器值进行符号位解码。

        将电机返回的有符号数值解码为正常的整数值。

        参数:
            data_name (str): 寄存器名称
            ids_values (dict[int, int]): 电机 ID 到原始值的映射（可能包含符号位）

        返回:
            dict[int, int]: 解码后的正常整数值
        """
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                # 使用二进制补码解码
                ids_values[id_] = decode_twos_complement(ids_values[id_], n_bytes)

        return ids_values

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        计算半圈偏移量。

        用于将电机的当前位置设置为"零点"（半圈位置）。
        Dynamixel 电机的公式：Present_Position = Actual_Position + Homing_Offset

        参数:
            positions (dict[NameOrID, Value]): 电机名称/ID 到当前位置的映射

        返回:
            dict[NameOrID, Value]: 电机名称/ID 到计算出的 Homing_Offset 的映射
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            # 半圈位置减去当前位置得到偏移量
            half_turn_homings[motor] = int(max_res / 2) - pos

        return half_turn_homings

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """
        将整数拆分为字节列表。

        参数:
            value (int): 要拆分的目标整数
            length (int): 目标字节数（1、2 或 4）

        返回:
            list[int]: 单字节整数列表
        """
        return _split_into_byte_chunks(value, length)

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """
        广播 ping 发现所有在线电机。

        使用广播地址（ID=254）ping 所有电机，获取每个电机的 ID 和型号编号。

        参数:
            num_retry (int, optional): 重试次数。默认为 0
            raise_on_error (bool, optional): True 时通信失败抛出异常。默认为 False

        返回:
            dict[int, int] | None: 键为电机 ID，值为型号编号。
                如通信失败且 raise_on_error=False 则返回 None
        """
        for n_try in range(1 + num_retry):
            # broadcastPing 是 SDK 的广播 ping 方法
            data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))

            return

        # 将结果从 {id_: [model_number]} 转换为 {id_: model_number}
        return {id_: data[0] for id_, data in data_list.items()}
