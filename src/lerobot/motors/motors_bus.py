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

"""
电机总线（MotorsBus）抽象基类模块。

本模块定义了与串口总线上多个电机进行高效通信的抽象接口。
支持两种主要电机类型：
  - Dynamixel（ROBOTIS 公司生产的伺服电机）
  - Feetech（富兴科技生产的伺服电机）

核心概念：
  1. 控制表（Control Table）：每个电机内部都有一组寄存器，通过读写这些寄存器可以控制电机。
     寄存器由地址（address）和长度（length，单位字节）标识。
  2. 同步读写（Sync Read/Write）：一次通信同时读写多个电机的同一寄存器，效率高于逐一读写。
  3. 归一化（Normalization）：将电机的原始编码器步进值转换为用户友好的相对值（如-100~100或角度）。
  4. 校准（Calibration）：记录每个电机的最小/最大位置限制和零位偏移，以便正确归一化。

通信流程：
  连接 → 握手验证 → 读取/写入寄存器 → 断开连接

典型使用示例：
  ```python
  bus = FeetechMotorsBus(
      port="/dev/ttyUSB0",
      motors={"shoulder_pan": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)},
  )
  bus.connect()

  # 读取当前位置（归一化值，范围 0-100）
  pos = bus.read("Present_Position", "shoulder_pan")

  # 写入目标位置
  bus.write("Goal_Position", "shoulder_pan", 50)

  # 断开连接
  bus.disconnect()
  ```
"""

import abc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import Protocol, TypeAlias

import serial
from deepdiff import DeepDiff
from tqdm import tqdm

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

# 类型别名定义
# NameOrID: 电机标识，可以是字符串名称（如 "shoulder_pan"）或整数 ID（如 1）
NameOrID: TypeAlias = str | int
# Value: 寄存器值类型，可以是整数（位置、速度等）或浮点数（归一化后的值）
Value: TypeAlias = int | float

logger = logging.getLogger(__name__)


def get_ctrl_table(model_ctrl_table: dict[str, dict], model: str) -> dict[str, tuple[int, int]]:
    """
    根据电机型号获取其控制表（寄存器地址映射）。

    参数:
        model_ctrl_table (dict[str, dict]): 型号到控制表的映射字典
            示例: {"xl430-w250": {"Goal_Position": (116, 4), "Present_Position": (132, 4), ...}}
        model (str): 电机型号名称

    返回:
        dict[str, tuple[int, int]]: 该型号的控制表

    异常:
        KeyError: 当型号不在 model_ctrl_table 中时抛出
    """
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for {model=} not found.")
    return ctrl_table


def get_address(model_ctrl_table: dict[str, dict], model: str, data_name: str) -> tuple[int, int]:
    """
    获取指定电机型号和寄存器名称的地址和字节长度。

    参数:
        model_ctrl_table (dict[str, dict]): 型号到控制表的映射
        model (str): 电机型号
        data_name (str): 寄存器名称（如 "Goal_Position"、"Present_Position"）

    返回:
        tuple[int, int]: (地址, 字节长度)
            - 地址：寄存器在电机控制表中的起始地址
            - 字节长度：寄存器的数据字节数（1、2 或 4）

    异常:
        KeyError: 当寄存器名称不在该型号的控制表中时抛出
    """
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(f"Address for '{data_name}' not found in {model} control table.")
    return addr_bytes


def assert_same_address(model_ctrl_table: dict[str, dict], motor_models: list[str], data_name: str) -> None:
    """
    断言多个电机型号对同一寄存器使用相同的地址和字节长度。

    用于同步读写前检查兼容性。

    参数:
        model_ctrl_table (dict[str, dict]): 型号到控制表的映射
        motor_models (list[str]): 电机型号列表
        data_name (str): 寄存器名称

    异常:
        NotImplementedError: 当不同型号使用不同地址或字节长度时抛出
    """
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = get_address(model_ctrl_table, model, data_name)
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_addr, strict=False))})."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_bytes, strict=False))})."
        )


class MotorNormMode(str, Enum):
    """
    电机值归一化模式枚举。

    决定如何将电机的原始步进值转换为用户友好的数值：

    RANGE_0_100:
        归一化到 0-100 范围，常用于夹爪等只需在一个方向上动作的执行器。
        公式: normalized = ((raw - min) / (max - min)) * 100

    RANGE_M100_100:
        归一化到 -100 到 +100 范围，常用于需要双向动作的关节。
        公式: normalized = (((raw - min) / (max - min)) * 200) - 100
        当 drive_mode 反转时，符号也会反转。

    DEGREES:
        归一化为角度值，基于编码器分辨率计算。
        公式: degrees = (raw - midpoint) * 360 / max_resolution
    """
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class MotorCalibration:
    """
    电机校准数据类。

    存储每个电机的校准参数，用于将原始编码器值转换为用户友好的归一化值。

    属性:
        id (int): 电机在总线上的 ID（不是名称）
        drive_mode (int): 驱动模式，0=正常，1=反转
        homing_offset (int): 零位偏移，用于将编码器零点与机械零点对齐
        range_min (int): 最小位置限制（原始编码器步进值）
        range_max (int): 最大位置限制（原始编码器步进值）

    归一化原理：
        用户看到的归一化值是一个相对百分比（0-100 或 -100 到 +100），
        表示当前值在 min 和 max 范围内所处的位置。
    """
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    """
    电机配置数据类。

    存储单个电机的基本配置信息。

    属性:
        id (int): 电机在总线上的唯一标识符（0-254）
        model (str): 电机型号名称，如 "sts3215"、"xl430-w250" 等
        norm_mode (MotorNormMode): 归一化模式，决定如何转换原始值
    """
    id: int
    model: str
    norm_mode: MotorNormMode


class JointOutOfRangeError(Exception):
    """
    关节超出范围异常。

    当用户尝试将关节移动到校准范围之外时抛出。
    """
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class PortHandler(Protocol):
    """
    串口处理器协议接口。

    描述了与串口通信所需的方法集合。
    实现类来自各 SDK（dynamixel_sdk 或 scservo_sdk）。
    """
    def __init__(self, port_name):
        self.is_open: bool
        self.baudrate: int
        self.packet_start_time: float
        self.packet_timeout: float
        self.tx_time_per_byte: float
        self.is_using: bool
        self.port_name: str
        self.ser: serial.Serial

    def openPort(self): ...
    def closePort(self): ...
    def clearPort(self): ...
    def setPortName(self, port_name): ...
    def getPortName(self): ...
    def setBaudRate(self, baudrate): ...
    def getBaudRate(self): ...
    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class PacketHandler(Protocol):
    """
    数据包处理器协议接口。

    描述了协议编解码所需的方法集合。
    """
    def getTxRxResult(self, result): ...
    def getRxPacketError(self, error): ...
    def txPacket(self, port, txpacket): ...
    def rxPacket(self, port): ...
    def txRxPacket(self, port, txpacket): ...
    def ping(self, port, id): ...
    def action(self, port, id): ...
    def readTx(self, port, id, address, length): ...
    def readRx(self, port, id, length): ...
    def readTxRx(self, port, id, address, length): ...
    def read1ByteTx(self, port, id, address): ...
    def read1ByteRx(self, port, id): ...
    def read1ByteTxRx(self, port, id, address): ...
    def read2ByteTx(self, port, id, address): ...
    def read2ByteRx(self, port): ...
    def read2ByteTxRx(self, port, id, address): ...
    def read4ByteTx(self, port, id, address): ...
    def read4ByteRx(self, port): ...
    def read4ByteTxRx(self, port, id, address): ...
    def writeTxOnly(self, port, id, address, length, data): ...
    def writeTxRx(self, port, id, address, length, data): ...
    def write1ByteTxOnly(self, port, id, address, data): ...
    def write1ByteTxRx(self, port, id, address, data): ...
    def write2ByteTxOnly(self, port, id, address, data): ...
    def write2ByteTxRx(self, port, id, address, data): ...
    def write4ByteTxOnly(self, port, id, address, data): ...
    def write4ByteTxRx(self, port, id, address, data): ...
    def regWriteTxOnly(self, port, id, address, length, data): ...
    def regWriteTxRx(self, port, id, address, length, data): ...
    def syncReadTx(self, port, start_address, data_length, param, param_length): ...
    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length): ...


class GroupSyncRead(Protocol):
    """
    同步读取器协议接口。

    用于一次通信读取多个电机的同一寄存器。
    通过预先设置起始地址和数据长度，然后添加要读取的电机 ID，
    最后调用 txRxPacket() 发起批量读取。
    """
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.last_result: bool
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id): ...
    def removeParam(self, id): ...
    def clearParam(self): ...
    def txPacket(self): ...
    def rxPacket(self): ...
    def txRxPacket(self): ...
    def isAvailable(self, id, address, data_length): ...
    def getData(self, id, address, data_length): ...


class GroupSyncWrite(Protocol):
    """
    同步写入器协议接口。

    用于一次通信向多个电机写入同一寄存器。
    先设置起始地址和数据长度，然后为每个电机添加 ID 和对应的值，
    最后调用 txPacket() 发起批量写入。
    """
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id, data): ...
    def removeParam(self, id): ...
    def changeParam(self, id, data): ...
    def clearParam(self): ...
    def txPacket(self): ...


class MotorsBus(abc.ABC):
    """
    电机总线抽象基类。

    该类抽象了与多个级联电机通信的所有通用操作。
    实际使用中，通过子类 DynamixelMotorsBus 或 FeetechMotorsBus 来具体实现。

    核心功能：
      1. 串口连接管理（连接、断开）
      2. 电机发现与验证（ping、握手）
      3. 寄存器读写（单读、单写、批量读写）
      4. 校准与归一化处理
      5. 扭矩控制（使能/禁用）

    重要概念：
      - NameOrID：可以用电机名称（str）或 ID（int）来引用电机
      - 归一化：用户值（0-100 或 -100 到 +100）与电机原始值之间的转换
      - 同步操作：通过一次总线通信读取/写入多个电机的同一寄存器

    使用示例：
      ```python
      bus = FeetechMotorsBus(
          port="/dev/tty.usbmodem575E0031751",
          motors={"my_motor": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)},
      )
      bus.connect()

      # 读取当前位置（归一化值，范围 0-100）
      position = bus.read("Present_Position", "my_motor", normalize=False)

      # 写入目标位置
      bus.write("Goal_Position", "my_motor", position + 30, normalize=False)

      bus.disconnect()
      ```
    """

    # 类属性：子类必须设置的具体实现相关属性
    apply_drive_mode: bool
    available_baudrates: list[int]
    default_baudrate: int
    default_timeout: int
    model_baudrate_table: dict[str, dict]
    model_ctrl_table: dict[str, dict]
    model_encoding_table: dict[str, dict]
    model_number_table: dict[str, int]
    model_resolution_table: dict[str, int]
    normalized_data: list[str]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        """
        初始化电机总线。

        参数:
            port (str): 串口设备路径。
                Linux 下通常是 "/dev/ttyUSB0" 或 "/dev/ttyACM0"
                Mac 下通常是 "/dev/tty.usbmodemXXX"
                Windows 下通常是 "COM3" 等
            motors (dict[str, Motor]): 电机字典。
                键：电机名称（字符串），用于代码中引用电机
                值：Motor 数据类，包含 id（总线ID）、model（型号）、norm_mode（归一化模式）
                示例: {"shoulder_pan": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)}
            calibration (dict[str, MotorCalibration] | None, optional): 预加载的校准数据。
                如不提供，将使用空字典，后续需要通过校准流程获取。
        """
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

        # 以下属性将在子类中由 SDK 初始化
        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.sync_reader: GroupSyncRead
        self.sync_writer: GroupSyncWrite
        self._comm_success: int
        self._no_error: int

        # 构建 ID 到型号、名称的映射字典，用于快速查找
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}
        # 型号编号到型号名称的反向映射（用于通过 ping 返回的编号查找型号）
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

        self._validate_motors()

    def __len__(self):
        """返回总线上注册的电机数量。"""
        return len(self.motors)

    def __repr__(self):
        """返回总线的可读表示。"""
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        """
        检查总线上是否有不同控制表的电机型号。

        返回:
            bool: True 如果有多个型号且它们使用不同的控制表
        """
        if len(self.models) < 2:
            return False

        first_table = self.model_ctrl_table[self.models[0]]
        return any(
            DeepDiff(first_table, get_ctrl_table(self.model_ctrl_table, model)) for model in self.models[1:]
        )

    @cached_property
    def models(self) -> list[str]:
        """
        获取总线上所有电机的型号列表（去重）。

        返回:
            list[str]: 型号名称列表
        """
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        """
        获取总线上所有电机的 ID 列表。

        返回:
            list[int]: 电机 ID 列表
        """
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        """
        通过型号编号查找型号名称。

        参数:
            motor_nb (int): 型号编号（ping 返回的值）

        返回:
            str: 型号名称
        """
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        """
        通过电机 ID 查找型号名称。

        参数:
            motor_id (int): 电机 ID

        返回:
            str: 型号名称
        """
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        """
        通过电机 ID 查找电机名称。

        参数:
            motor_id (int): 电机 ID

        返回:
            str: 电机名称
        """
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        """
        将电机标识符（名称或ID）转换为电机 ID。

        参数:
            motor (NameOrID): 电机名称（str）或 ID（int）

        返回:
            int: 电机 ID

        异常:
            TypeError: 当类型既不是 str 也不是 int 时抛出
        """
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motor_model(self, motor: NameOrID) -> int:
        """
        将电机标识符（名称或ID）转换为电机型号。

        参数:
            motor (NameOrID): 电机名称（str）或 ID（int）

        返回:
            str: 电机型号

        异常:
            TypeError: 当类型既不是 str 也不是 int 时抛出
        """
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """
        规范化电机列表输入。

        将各种输入格式统一为字符串列表。

        参数:
            motors (str | list[str] | None):
                - None: 返回所有电机名称
                - str: 返回包含单个名称的列表
                - list[str]: 原样返回

        返回:
            list[str]: 电机名称列表

        异常:
            TypeError: 当格式不正确时抛出
        """
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors.copy()
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> list[str]:
        """
        将输入值转换为 {motor_id: value} 格式。

        用于同步写入操作。

        参数:
            values (Value | dict[str, Value] | None):
                - 单一数值：将应用于所有电机
                - 字典：电机名称到值的映射

        返回:
            dict[int, Value]: 电机 ID 到值的映射

        异常:
            TypeError: 当格式不正确时抛出
        """
        if isinstance(values, (int, float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        else:
            raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def _validate_motors(self) -> None:
        """
        验证电机配置的合法性。

        检查项：
          1. 所有电机 ID 唯一
          2. 所有电机型号都有对应的控制表
        """
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Some motors have the same id!\n{self}")

        # Ensure ctrl table available for all models
        for model in self.models:
            get_ctrl_table(self.model_ctrl_table, model)

    def _is_comm_success(self, comm: int) -> bool:
        """
        检查通信结果是否成功。

        参数:
            comm (int): SDK 返回的通信结果码

        返回:
            bool: True 如果通信成功
        """
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        """
        检查电机返回的错误状态。

        参数:
            error (int): 电机返回的错误码

        返回:
            bool: True 如果有错误
        """
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        """
        断言所有注册的电机都能在总线上找到。

        通过逐一 ping 每个电机的 ID，检查：
          1. 电机是否存在（能响应 ping）
          2. 电机型号是否匹配预期

        异常:
            RuntimeError: 当有电机缺失或型号不匹配时抛出
        """
        expected_models = {m.id: self.model_number_table[m.model] for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            error_lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]

            if missing_ids:
                error_lines.append("\nMissing motor IDs:")
                error_lines.extend(
                    f"  - {id_} (expected model: {expected_models[id_]})" for id_ in missing_ids
                )

            if wrong_models:
                error_lines.append("\nMotors with incorrect model numbers:")
                error_lines.extend(
                    f"  - {id_} ({self._id_to_name(id_)}): expected {expected}, found {found}"
                    for id_, (expected, found) in wrong_models.items()
                )

            error_lines.append("\nFull expected motor list (id: model_number):")
            error_lines.append(pformat(expected_models, indent=4, sort_dicts=False))
            error_lines.append("\nFull found motor list (id: model_number):")
            error_lines.append(pformat(found_models, indent=4, sort_dicts=False))

            raise RuntimeError("\n".join(error_lines))

    @abc.abstractmethod
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        """
        断言当前协议支持特定指令。

        抽象方法，子类实现具体检查逻辑。

        参数:
            instruction_name (str): 指令名称（如 "sync_read"、"broadcast_ping"）
        """
        pass

    @property
    def is_connected(self) -> bool:
        """
        端口连接状态属性。

        返回:
            bool: True 如果串口已打开
        """
        return self.port_handler.is_open

    def connect(self, handshake: bool = True) -> None:
        """
        打开串口并初始化通信。

        参数:
            handshake (bool, optional): 是否执行握手验证。
                True（默认）：ping 每个预期的电机并执行完整性检查
                False：仅打开端口，不验证电机存在
                通常在首次连接或调试时设为 True，正常使用时保持 True

        异常:
            DeviceAlreadyConnectedError: 端口已经打开时抛出
            ConnectionError: 打开端口失败或握手失败时抛出
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected. "
                f"Do not call `{self.__class__.__name__}.connect()` twice."
            )

        self._connect(handshake)
        self.set_timeout()
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        """
        内部方法：执行实际的连接操作。

        参数:
            handshake (bool): 是否执行握手
        """
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            elif handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

    @abc.abstractmethod
    def _handshake(self) -> None:
        """
        抽象方法：执行握手验证。

        子类实现具体的握手/初始化逻辑。
        """
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        """
        关闭串口连接。

        参数:
            disable_torque (bool, optional): 关闭前是否禁用扭矩。
                True（默认）：在关闭前禁用所有电机的扭矩，
                    防止电机在断开后继续保持阻力姿态而损坏
                False：直接关闭端口（可能不安全）
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"Try running `{self.__class__.__name__}.connect()` first."
            )

        if disable_torque:
            self.port_handler.clearPort()
            self.port_handler.is_using = False
            self.disable_torque(num_retry=5)

        self.port_handler.closePort()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_port(cls, port: str, *args, **kwargs) -> dict[int, list[int]]:
        """
        扫描指定端口，查找所有响应的电机。

        尝试所有支持的波特率，找出在每个波特率上响应的电机 ID。

        参数:
            port (str): 要扫描的串口路径
            *args, **kwargs: 转发给子类构造函数的额外参数

        返回:
            dict[int, list[int]]: 键为波特率，值为在该波特率上响应的电机 ID 列表
                示例: {1000000: [1, 2, 3], 57600: [5]}

        使用示例:
            ```python
            found = FeetechMotorsBus.scan_port("/dev/ttyUSB0")
            for baudrate, ids in found.items():
                print(f"Baudrate {baudrate}: IDs {ids}")
            ```
        """
        bus = cls(port, {}, *args, **kwargs)
        bus._connect(handshake=False)
        baudrate_ids = {}
        for baudrate in tqdm(bus.available_baudrates, desc="Scanning port"):
            bus.set_baudrate(baudrate)
            ids_models = bus.broadcast_ping()
            if ids_models:
                tqdm.write(f"Motors found for {baudrate=}: {pformat(ids_models, indent=4)}")
                baudrate_ids[baudrate] = list(ids_models)

        bus.port_handler.closePort()
        return baudrate_ids

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """
        配置单个电机的 ID 和波特率。

        这是一个引导程序，用于首次设置新电机或更改电机 ID。
        过程：
          1. 搜索电机（如果未提供初始波特率和 ID）
          2. 禁用扭矩
          3. 设置新 ID
          4. 设置新波特率
          5. 切换到默认波特率

        参数:
            motor (str): 电机名称（在 self.motors 中定义）
            initial_baudrate (int | None, optional): 电机当前的波特率。
                如果不知道则设为 None，会自动搜索
            initial_id (int | None, optional): 电机当前的 ID。
                如果不知道则设为 None，会自动搜索

        异常:
            RuntimeError: 找不到电机或型号不匹配
            ConnectionError: 通信失败
        """
        if not self.is_connected:
            self._connect(handshake=False)

        if initial_baudrate is None:
            initial_baudrate, initial_id = self._find_single_motor(motor)

        if initial_id is None:
            _, initial_id = self._find_single_motor(motor, initial_baudrate)

        model = self.motors[motor].model
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        self._disable_torque(initial_id, model)

        # Set ID
        addr, length = get_address(self.model_ctrl_table, model, "ID")
        self._write(addr, length, initial_id, target_id)

        # Set Baudrate
        addr, length = get_address(self.model_ctrl_table, model, "Baud_Rate")
        baudrate_value = self.model_baudrate_table[model][self.default_baudrate]
        self._write(addr, length, target_id, baudrate_value)

        self.set_baudrate(self.default_baudrate)

    @abc.abstractmethod
    def _find_single_motor(self, motor: str, initial_baudrate: int | None) -> tuple[int, int]:
        """
        抽象方法：查找单个电机。

        参数:
            motor (str): 电机名称
            initial_baudrate (int | None): 初始波特率

        返回:
            tuple[int, int]: (波特率, 电机ID)
        """
        pass

    @abc.abstractmethod
    def configure_motors(self) -> None:
        """
        抽象方法：配置所有电机参数。

        子类实现具体的配置写入，如响应延迟、加速度限制等。
        """
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        抽象方法：禁用扭矩。

        禁用扭矩后电机可以自由转动，用于手动示教或校准。

        参数:
            motors (int | str | list[str] | None, optional): 目标电机
                - None: 所有电机
                - int: 单个电机 ID
                - str: 单个电机名称
                - list[str]: 多个电机名称
            num_retry (int, optional): 重试次数
        """
        pass

    @abc.abstractmethod
    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        """
        抽象方法：禁用单个电机的扭矩（内部方法）。

        参数:
            motor (int): 电机 ID
            model (str): 电机型号
            num_retry (int, optional): 重试次数
        """
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        抽象方法：使能扭矩。

        参数:
            motors (str | list[str] | None, optional): 目标电机
            num_retry (int, optional): 重试次数
        """
        pass

    @contextmanager
    def torque_disabled(self, motors: int | str | list[str] | None = None):
        """
        上下文管理器：临时禁用扭矩后自动恢复。

        用于在配置电机时（如写入校准数据）临时禁用扭矩，
        然后自动重新使能。

        使用示例:
            ```python
            with bus.torque_disabled():
                bus.write("Homing_Offset", "shoulder_pan", 100)
                bus.write("Min_Position_Limit", "shoulder_pan", 0)
            # 离开 with 块时扭矩自动恢复
            ```

        参数:
            motors: 同 disable_torque
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):
        """
        设置通信超时时间。

        参数:
            timeout_ms (int | None, optional): 超时时间（毫秒）。
                None 时使用 default_timeout（类属性）
        """
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:
        """
        获取当前波特率。

        返回:
            int: 当前波特率（bps）
        """
        return self.port_handler.getBaudRate()

    def set_baudrate(self, baudrate: int) -> None:
        """
        设置总线波特率。

        参数:
            baudrate (int): 目标波特率（bps）

        异常:
            RuntimeError: 设置失败时抛出
        """
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            logger.info(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise RuntimeError("Failed to write bus baud rate.")

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """
        抽象属性：校准状态。

        返回:
            bool: True 如果缓存的校准与电机当前值匹配
        """
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """
        抽象方法：从电机读取校准数据。

        返回:
            dict[str, MotorCalibration]: 键为电机名称
        """
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """
        抽象方法：向电机写入校准数据。

        参数:
            calibration_dict (dict[str, MotorCalibration]): 校准数据
            cache (bool, optional): 是否缓存到 self.calibration
        """
        pass

    def reset_calibration(self, motors: NameOrID | list[NameOrID] | None = None) -> None:
        """
        重置电机的校准为出厂默认值。

        将以下参数恢复为默认值：
          - Homing_Offset = 0
          - Min_Position_Limit = 0
          - Max_Position_Limit = 最大分辨率 - 1（全范围）

        同时清空内存中的校准缓存。

        参数:
            motors (NameOrID | list[NameOrID] | None, optional): 目标电机。
                None 表示所有电机
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        for motor in motors:
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None) -> dict[NameOrID, Value]:
        """
        将当前位置设置为电机的"半圈零点"。

        计算并写入 Homing_Offset，使得当前位置变成编码器的半程值
        （如 12 位编码器的 2047，即 4096/2）。

        用于将机械中点与编码器中点对齐。

        参数:
            motors (NameOrID | list[NameOrID] | None, optional): 目标电机

        返回:
            dict[NameOrID, Value]: 写入的 Homing_Offset 值
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        self.reset_calibration(motors)
        actual_positions = self.sync_read("Present_Position", motors, normalize=False)
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        抽象方法：计算半圈偏移量。

        子类实现，因为 Feetech 和 Dynamixel 的偏移计算公式不同。

        参数:
            positions (dict[NameOrID, Value]): 当前位置映射

        返回:
            dict[NameOrID, Value]: Homing_Offset 映射
        """
        pass

    def record_ranges_of_motion(
        self, motors: NameOrID | list[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
        """
        交互式记录电机的运动范围（最小值和最大值）。

        在扭矩禁用状态下手动移动关节，系统实时记录位置范围。
        按 Enter 键结束记录。

        参数:
            motors (NameOrID | list[NameOrID] | None, optional): 目标电机
            display_values (bool, optional): 是否在终端显示实时数值

        返回:
            tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
                - 第一个字典：各电机的最小位置
                - 第二个字典：各电机的最大位置

        异常:
            ValueError: 如果所有电机的 min==max（没有移动）
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        start_positions = self.sync_read("Present_Position", motors, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motors, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<23} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motors:
                    print(f"{motor:<23} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(motors) + 3)

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        """
        将电机的原始值转换为归一化值。

        根据电机的 norm_mode 和校准数据，将原始编码器步进值
        转换为用户友好的相对值。

        参数:
            ids_values (dict[int, int]): 电机 ID 到原始值的映射

        返回:
            dict[int, float]: 电机 ID 到归一化值的映射

        异常:
            RuntimeError: 当没有校准数据时抛出
        """
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                # 范围 -100 到 +100
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                # 范围 0 到 100
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                # 转换为角度
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        """
        将归一化值转换为电机的原始值。

        是 _normalize 的逆操作。

        参数:
            ids_values (dict[int, float]): 电机 ID 到归一化值的映射

        返回:
            dict[int, int]: 电机 ID 到原始值的映射

        异常:
            RuntimeError: 当没有校准数据时抛出
        """
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @abc.abstractmethod
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        抽象方法：对有符号寄存器值进行编码。

        某些寄存器使用符号-幅度编码，需要特殊处理。

        参数:
            data_name (str): 寄存器名称
            ids_values (dict[int, int]): 电机 ID 到值的映射

        返回:
            dict[int, int]: 编码后的值
        """
        pass

    @abc.abstractmethod
    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        抽象方法：对有符号寄存器值进行解码。

        参数:
            data_name (str): 寄存器名称
            ids_values (dict[int, int]): 电机 ID 到值的映射

        返回:
            dict[int, int]: 解码后的值
        """
        pass

    def _serialize_data(self, value: int, length: int) -> list[int]:
        """
        将整数序列化为字节列表。

        用于通信协议的数据打包。

        参数:
            value (int): 要序列化的整数值（必须是正数）
            length (int): 目标字节数（1、2 或 4）

        返回:
            list[int]: 单字节整数列表

        异常:
            ValueError: 当值为负数或超出指定字节能表示的范围时抛出
        """
        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {length} bytes ({max_value}).")

        return self._split_into_byte_chunks(value, length)

    @abc.abstractmethod
    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """
        抽象方法：将整数拆分为字节列表。

        参数:
            value (int): 要拆分的目标整数
            length (int): 目标字节数

        返回:
            list[int]: 单字节整数列表
        """
        pass

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """
        Ping 单个电机，获取其型号编号。

        参数:
            motor (NameOrID): 电机名称或 ID
            num_retry (int, optional): 额外重试次数
            raise_on_error (bool, optional): True 时通信错误抛出异常

        返回:
            int | None: 电机型号编号（成功时），失败返回 None
        """
        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
            if self._is_comm_success(comm):
                break
            logger.debug(f"ping failed for {id_=}: {n_try=} got {comm=} {error=}")

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            else:
                return
        if self._is_error(error):
            if raise_on_error:
                raise RuntimeError(self.packet_handler.getRxPacketError(error))
            else:
                return

        return model_number

    @abc.abstractmethod
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """
        抽象方法：广播 ping 发现所有电机。

        参数:
            num_retry (int, optional): 重试次数
            raise_on_error (bool, optional): True 时失败抛出异常

        返回:
            dict[int, int] | None: ID 到型号编号的映射
        """
        pass

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """
        读取单个电机的寄存器值。

        参数:
            data_name (str): 寄存器名称（如 "Present_Position"、"Goal_Velocity" 等）
            motor (str): 电机名称（在 self.motors 中定义的键）
            normalize (bool, optional): 是否归一化。
                True（默认）：将原始值转换为归一化值（0-100、-100到100 或角度）
                False：返回原始编码器步进值
            num_retry (int, optional): 重试次数

        返回:
            Value: 读取的值（归一化或原始值）

        异常:
            DeviceNotConnectedError: 未连接时抛出
            KeyError: 寄存器名称不存在时抛出
            ConnectionError: 通信失败时抛出
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        value, _, _ = self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

        # 对有符号值进行解码
        id_value = self._decode_sign(data_name, {id_: value})

        # 如需要，进行归一化处理
        if normalize and data_name in self.normalized_data:
            id_value = self._normalize(id_value)

        return id_value[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        """
        内部方法：执行底层单电机寄存器读取。

        参数:
            address (int): 寄存器地址
            length (int): 数据字节长度（1、2 或 4）
            motor_id (int): 电机 ID
            num_retry (int, optional): 重试次数
            raise_on_error (bool, optional): True 时错误抛出异常
            err_msg (str, optional): 错误信息前缀

        返回:
            tuple[int, int]: (读取的值, 通信结果码, 错误码)
        """
        # 根据数据长度选择读取函数
        if length == 1:
            read_fn = self.packet_handler.read1ByteTxRx
        elif length == 2:
            read_fn = self.packet_handler.read2ByteTxRx
        elif length == 4:
            read_fn = self.packet_handler.read4ByteTxRx
        else:
            raise ValueError(length)

        for n_try in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to read @{address=} ({length=}) on {motor_id=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return value, comm, error

    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """
        向单个电机写入寄存器值。

        与 sync_write 不同，此方法等待电机返回确认响应，
        因此更可靠但速度较慢。适用于配置操作。

        参数:
            data_name (str): 寄存器名称
            motor (str): 电机名称
            value (Value): 要写入的值
            normalize (bool, optional): 是否反归一化。
                True（默认）：将归一化值转换为原始值再写入
                False：直接写入原始值
            num_retry (int, optional): 重试次数

        异常:
            DeviceNotConnectedError: 未连接时抛出
            KeyError: 寄存器名称不存在时抛出
            ConnectionError: 通信失败时抛出
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        # 如需要，将归一化值转换为原始值
        if normalize and data_name in self.normalized_data:
            value = self._unnormalize({id_: value})[id_]

        # 对有符号值进行编码
        value = self._encode_sign(data_name, {id_: value})[id_]

        err_msg = f"Failed to write '{data_name}' on {id_=} with '{value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        """
        内部方法：执行底层单电机寄存器写入。

        参数:
            addr (int): 寄存器地址
            length (int): 数据字节长度
            motor_id (int): 电机 ID
            value (int): 要写入的值（必须是已序列化的原始值）
            num_retry (int, optional): 重试次数
            raise_on_error (bool, optional): True 时错误抛出异常
            err_msg (str, optional): 错误信息前缀

        返回:
            tuple[int, int]: (通信结果码, 错误码)
        """
        data = self._serialize_data(value, length)
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, length, data)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) on id={motor_id} with {value=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return comm, error

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """
        同步读取多个电机的同一寄存器。

        通过一次总线通信读取所有目标电机的同一寄存器，
        效率远高于循环调用 read()。

        参数:
            data_name (str): 寄存器名称
            motors (str | list[str] | None, optional): 目标电机。
                None 表示所有电机
            normalize (bool, optional): 是否归一化。默认 True
            num_retry (int, optional): 重试次数

        返回:
            dict[str, Value]: 键为电机名称，值为读取的值

        异常:
            DeviceNotConnectedError: 未连接时抛出
            NotImplementedError: 不同电机型号使用不同寄存器地址时抛出
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )

        # 1) 校验当前总线/协议是否支持 sync read。
        self._assert_protocol_is_compatible("sync_read")

        # 2) 将用户传入的电机选择解析为具体的电机 id / model。
        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        # 3) 为目标电机解析寄存器地址和字节长度。
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        # 4) 执行总线读取，并对寄存器值做符号位解码。
        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        ids_values, _ = self._sync_read(
            addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )
        ids_values = self._decode_sign(data_name, ids_values)

        # 5) 如有需要，将原始值转换回用户侧归一化范围。
        if normalize and data_name in self.normalized_data:
            ids_values = self._normalize(ids_values)

        # 6) 按"电机名 -> 数值"返回（对外接口约定）。
        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[dict[int, int], int]:
        """
        内部方法：执行底层同步读取。

        该方法使用底层的 sync_reader 执行实际的读取操作。

        参数:
            addr (int): 寄存器地址
            length (int): 读取数据的字节长度
            motor_ids (list[int]): 目标电机 ID 列表
            num_retry (int, optional): 重试次数。默认为 0
            raise_on_error (bool, optional): 出错时是否抛出异常。默认为 True
            err_msg (str, optional): 出错时抛出的异常信息前缀。默认为空字符串

        返回:
            tuple[dict[int, int], int]: 返回一个元组，包含:
                - dict[int, int]: 映射 *电机 ID -> 读取的值*
                - int: 通信结果状态码

        异常:
            ConnectionError: 通信失败且 raise_on_error 为 True 时抛出
        """
        # A) 组装 GroupSyncRead 参数（地址、长度、目标 id 列表）。
        t0 = time.perf_counter()
        self._setup_sync_reader(motor_ids, addr, length)
        setup_ms = (time.perf_counter() - t0) * 1e3

        # B) 执行总线收发（通常是主要耗时段）。
        txrx_start = time.perf_counter()
        for n_try in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync read @{addr=} ({length=}) on {motor_ids=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )
        txrx_ms = (time.perf_counter() - txrx_start) * 1e3

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        # C) 从回包中提取各电机的寄存器值。
        unpack_start = time.perf_counter()
        values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
        unpack_ms = (time.perf_counter() - unpack_start) * 1e3
        total_ms = (time.perf_counter() - t0) * 1e3
        logger.debug(
            "bus._sync_read addr=%s len=%s motors=%s total=%.3fms (setup=%.3f, txrx=%.3f, unpack=%.3f)",
            addr,
            length,
            len(motor_ids),
            total_ms,
            setup_ms,
            txrx_ms,
            unpack_ms,
        )
        return values, comm

    def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
        """
        配置同步读取器。

        参数:
            motor_ids (list[int]): 要读取的电机 ID 列表
            addr (int): 寄存器地址
            length (int): 数据字节长度
        """
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """
        同步向多个电机写入同一寄存器。

        通过一次总线通信向所有目标电机写入同一寄存器的值，
        效率高但不等待确认响应，可能丢包。适用于高频控制场景。

        参数:
            data_name (str): 寄存器名称
            values (Value | dict[str, Value]): 要写入的值。
                - 单一数值：所有电机写入相同值
                - 字典：电机名称到值的映射
            normalize (bool, optional): 是否反归一化。默认 True
            num_retry (int, optional): 重试次数

        异常:
            DeviceNotConnectedError: 未连接时抛出
            NotImplementedError: 不同电机型号使用不同寄存器地址时抛出
            ConnectionError: 通信失败时抛出
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )

        # 1) 将输入规整成 {motor_id: value}，并收集目标电机 model。
        ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in ids_values]
        # 2) 解析目标寄存器地址和字节长度。
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        # 3) 如有需要，将用户侧归一化数值反变换为寄存器原始单位。
        if normalize and data_name in self.normalized_data:
            ids_values = self._unnormalize(ids_values)
        # 4) 做寄存器符号位编码后，下发到总线。
        ids_values = self._encode_sign(data_name, ids_values)

        err_msg = f"Failed to sync write '{data_name}' with {ids_values=} after {num_retry + 1} tries."
        self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int:
        """
        内部方法：执行底层同步写入。

        参数:
            addr (int): 寄存器地址
            length (int): 数据字节长度
            ids_values (dict[int, int]): 电机 ID 到值的映射
            num_retry (int, optional): 重试次数
            raise_on_error (bool, optional): True 时错误抛出异常
            err_msg (str, optional): 错误信息前缀

        返回:
            int: 通信结果码
        """
        # A) 组装 GroupSyncWrite 载荷。
        t0 = time.perf_counter()
        self._setup_sync_writer(ids_values, addr, length)
        setup_ms = (time.perf_counter() - t0) * 1e3

        # B) 在总线上发送数据包（通常是主要耗时段）。
        tx_start = time.perf_counter()
        for n_try in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) with {ids_values=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )
        tx_ms = (time.perf_counter() - tx_start) * 1e3

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        total_ms = (time.perf_counter() - t0) * 1e3
        logger.debug(
            "bus._sync_write addr=%s len=%s motors=%s total=%.3fms (setup=%.3f, tx=%.3f)",
            addr,
            length,
            len(ids_values),
            total_ms,
            setup_ms,
            tx_ms,
        )

        return comm

    def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
        """
        配置同步写入器。

        参数:
            ids_values (dict[int, int]): 电机 ID 到值的映射
            addr (int): 寄存器地址
            length (int): 数据字节长度
        """
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)
