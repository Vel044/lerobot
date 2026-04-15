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
Feetech 电机总线驱动模块

本模块是Feetech品牌舵机的核心通信驱动,基于SCServoSDK(仿Dynamixel协议)实现。
支持Protocol 0(STS/SMS系列)和Protocol 1(SCS系列)两种通信协议。

核心功能:
1. 通过串口与电机通信 (RS485/RS232)
2. 读写电机寄存器 (控制表)
3. 同步读写多个电机 (GroupSyncRead/Write)
4. 电机校准与配置
5. 位置/速度/PWM等多种控制模式

通信流程:
  用户调用 → FeetechMotorsBus → scservo_sdk → 串口 → 电机

主要类:
- FeetechMotorsBus: 电机总线类,管理一组电机
- OperatingMode: 运行模式枚举
- DriveMode: 驱动方向枚举
- TorqueMode: 扭矩使能枚举

作者: HuggingFace LeRobot Team
"""

import logging
from copy import deepcopy
from enum import Enum
from pprint import pformat

# 从lerobot通用模块导入编码/解码工具
# sign-magnitude编码: 处理电机寄存器中符号位与数据位混合的情况
from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

# 从父目录导入电机总线基类和类型定义
# MotorsBus: 通用电机总线抽象类
# Motor: 单个电机配置(name, id, model)
# MotorCalibration: 电机校准数据(零位偏移, 位置限位等)
# NameOrID: 电机名称或ID的联合类型
# Value: 电机值的类型(通常是int)
# get_address: 从控制表获取寄存器地址的工具函数
from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address

# 从tables模块导入所有电机型号配置
from .tables import (
    FIRMWARE_MAJOR_VERSION,   # 固件主版本号寄存器 (0, 1)
    FIRMWARE_MINOR_VERSION,   # 固件次版本号寄存器 (1, 1)
    MODEL_BAUDRATE_TABLE,     # 电机型号→波特率表映射
    MODEL_CONTROL_TABLE,      # 电机型号→控制表映射
    MODEL_ENCODING_TABLE,     # 电机型号→编码表映射
    MODEL_NUMBER,             # 型号寄存器地址 (3, 2)
    MODEL_NUMBER_TABLE,       # 型号编号映射 (如 sts3215→777)
    MODEL_PROTOCOL,           # 电机型号→协议版本映射
    MODEL_RESOLUTION,         # 电机分辨率 (一圈脉冲数)
    SCAN_BAUDRATES,           # 电机扫描时尝试的波特率列表
)

# ============================================================
# 全局常量定义
# ============================================================

# 协议版本,0=Protocol 0(STS/SMS系列),1=Protocol 1(SCS系列)
# Protocol 0是Feetech自定义的半双工协议
# Protocol 1兼容标准Dynamixel协议
DEFAULT_PROTOCOL_VERSION = 0

# 默认通信波特率 (1Mbps),高速率减少通信延迟
DEFAULT_BAUDRATE = 1_000_000

# 默认通信超时时间 (毫秒)
# 超时用于检测电机响应丢失,避免程序阻塞
DEFAULT_TIMEOUT_MS = 1000

# 归一化数据字段名列表
# 归一化指将电机原始值(0-4095)转换为标准化的相对值(0-1)
# 主要用于不同电机型号之间的兼容性
NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

# 日志记录器
logger = logging.getLogger(__name__)


# ============================================================
# 枚举类定义 - 电机运行模式和配置选项
# ============================================================

class OperatingMode(Enum):
    """
    电机运行模式枚举

    决定电机的控制方式和行为特性

    成员:
    - POSITION (值=0): 位置伺服模式
        最常用的模式,通过Goal_Position设定目标位置
        电机自动以设定的速度加速/减速到达目标

    - VELOCITY (值=1): 恒速模式
        电机以恒定速度运转,速度由Goal_Velocity设定
        最高位(bit15)作为方向位: 0=顺时针,1=逆时针

    - PWM (值=2): PWM开环调速模式
        直接通过PWM占空比控制电机功率/速度
        运行时间由参数0x2c控制,bit11作为方向位

    - STEP (值=3): 步进伺服模式
        步进式转动,步数由参数0x2a指定
        最高位(bit15)作为方向位
    """
    POSITION = 0   # 位置伺服模式(默认)
    VELOCITY = 1   # 恒速模式
    PWM = 2        # PWM开环调速
    STEP = 3       # 步进模式


class DriveMode(Enum):
    """
    驱动方向模式枚举

    用于设置电机输出轴的旋转方向

    成员:
    - NON_INVERTED (值=0): 正常方向
        电机输出轴旋转方向与指令一致

    - INVERTED (值=1): 反转方向
        电机输出轴旋转方向与指令相反
        用于机械结构需要反转的场合
    """
    NON_INVERTED = 0  # 正常方向
    INVERTED = 1      # 方向反转


class TorqueMode(Enum):
    """
    扭矩使能模式枚举

    控制电机是否输出扭矩

    成员:
    - ENABLED (值=1): 使能扭矩输出
        电机保持当前位置,对抗外部力量

    - DISABLED (值=0): 禁用扭矩输出
        电机处于自由状态,可以手动转动输出轴
        常用于校准或手动操作时
    """
    ENABLED = 1    # 使能扭矩(锁住当前位置)
    DISABLED = 0   # 禁用扭矩(可自由转动)


# ============================================================
# 工具函数
# ============================================================

def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    """
    将整数拆分为字节数组

    用于将多字节寄存器值拆分为单字节列表,以便通过串口发送

    参数:
    -----------
    value : int
        要拆分的整数值
        例如: 0x1234 (十进制4660)
    length : int
        字节长度,必须是1、2或4
        1=1字节(8位), 2=2字节(16位), 4=4字节(32位)

    返回:
    --------
    list[int]
        字节列表,每个元素是0-255的字节值
        例如: value=0x1234, length=2 → [0x34, 0x12] (小端序)

    示例:
    --------
    _split_into_byte_chunks(0x1234, 2) → [0x34, 0x12]  # LO_BYTE=0x34, HI_BYTE=0x12
    _split_into_byte_chunks(0x12345678, 4) → [0x78, 0x56, 0x34, 0x12]

    注意:
    --------
    Feetech电机使用小端序(Little Endian),低字节在前
    这与标准Dynamixel协议一致
    """
    import scservo_sdk as scs

    if length == 1:
        # 1字节:直接返回
        data = [value]
    elif length == 2:
        # 2字节:分解为高低字节
        # SCS_LOBYTE: 提取低8位
        # SCS_HIBYTE: 提取高8位
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    elif length == 4:
        # 4字节:分解为4个字节
        # SCS_LOWORD: 提取低16位
        # SCS_HIWORD: 提取高16位
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),  # 字节0: 低16位的低字节
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),   # 字节1: 低16位的高字节
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),   # 字节2: 高16位的低字节
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),   # 字节3: 高16位的高字节
        ]
    return data


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    猴子补丁:修复SCServoSDK的packet timeout计算bug

    问题来源:
    - 官方scservo_sdk的PortHandler.setPacketTimeout实现有误
    - issue: https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6

    修复原理:
    - 原始bug: 超时时间计算少了3倍字节传输时间
    - 正确公式: packet_timeout = tx_time_per_byte × packet_length
    -            + tx_time_per_byte × 3.0 + 50

    参数:
    -----------
    self : PortHandler
        串口处理器实例(被patch后替代原方法)
    packet_length : int
        数据包长度(字节数)

    注意:
    --------
    这是临时补丁,因为PyPI上的scservo_sdk版本未修复此bug
    官方已修复但未发布到PyPI
    """
    # 获取当前时间作为包传输开始时间
    self.packet_start_time = self.getCurrentTime()
    # 计算超时: 传输时间 + 3倍冗余 + 50ms固定延迟
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


# ============================================================
# 主类: FeetechMotorsBus
# ============================================================

class FeetechMotorsBus(MotorsBus):
    """
    Feetech电机总线类 - 管理一组Feetech电机

    本类继承自MotorsBus基类,实现了Feetech特定通信协议。

    核心功能:
    1. 串口通信管理 (通过scservo_sdk)
    2. 电机寄存器读写 (同步/异步)
    3. 多电机同步控制 (GroupSyncRead/GroupSyncWrite)
    4. 电机发现与ID分配
    5. 校准数据管理

    使用示例:
    -----------
    # 创建电机总线
    motors = {
        "shoulder_pan": Motor(name="shoulder_pan", id=1, model="sts3215"),
        "shoulder_lift": Motor(name="shoulder_lift", id=2, model="sts3215"),
    }
    bus = FeetechMotorsBus("/dev/ttyUSB0", motors)

    # 读取当前位置
    positions = bus.read("Present_Position")

    # 设置目标位置
    bus.write("Goal_Position", {"shoulder_pan": 2048, "shoulder_lift": 1024})

    # 使能扭矩
    bus.enable_torque()

    继承属性:
    -----------
    - port: str 串口设备路径
    - motors: dict[str, Motor] 电机字典 {名称: Motor}
    - calibration: dict[str, MotorCalibration] 校准数据

    类属性:
    -----------
    - apply_drive_mode: bool 是否应用驱动方向
    - available_baudrates: list 支持的波特率列表
    - default_baudrate: int 默认波特率
    - default_timeout: int 默认超时时间(ms)
    - model_*: 各型号配置表
    - normalized_data: list 归一化数据字段
    """

    # 类属性: 从tables模块复制(深拷贝避免修改原始表)
    apply_drive_mode = True                      # 是否应用驱动方向反转
    available_baudrates = deepcopy(SCAN_BAUDRATES)  # 可用波特率列表
    default_baudrate = DEFAULT_BAUDRATE           # 默认1Mbps
    default_timeout = DEFAULT_TIMEOUT_MS         # 默认1000ms超时
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)   # 波特率配置表
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)        # 控制表
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)    # 编码表
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)         # 型号表
    model_resolution_table = deepcopy(MODEL_RESOLUTION)      # 分辨率表
    normalized_data = deepcopy(NORMALIZED_DATA)              # 归一化字段

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ):
        """
        初始化Feetech电机总线

        参数:
        -----------
        port : str
            串口设备路径
            Linux示例: "/dev/ttyUSB0", "/dev/ttyACM0"
            Windows示例: "COM3"

        motors : dict[str, Motor]
            电机字典,键为电机名称(字符串),值为Motor对象
            示例: {"shoulder_pan": Motor(name="shoulder_pan", id=1, model="sts3215")}

        calibration : dict[str, MotorCalibration] | None
            校准数据字典,键为电机名称,值为MotorCalibration对象
            包含: homing_offset(零点偏移), range_min/max(位置限位)
            可选,首次使用需要校准

        protocol_version : int
            通信协议版本
            0 = Protocol 0 (用于STS/SMS系列,半双工)
            1 = Protocol 1 (用于SCS系列,全双工)
            默认值: 0

        示例:
        -----------
        bus = FeetechMotorsBus(
            port="/dev/ttyUSB0",
            motors={
                "joint1": Motor(name="joint1", id=1, model="sts3215"),
                "joint2": Motor(name="joint2", id=2, model="sts3215"),
            },
            calibration={
                "joint1": MotorCalibration(id=1, drive_mode=0, homing_offset=100, range_min=0, range_max=4095),
                "joint2": MotorCalibration(id=2, drive_mode=0, homing_offset=-50, range_min=0, range_max=4095),
            },
            protocol_version=0
        )
        """
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version
        self._assert_same_protocol()
        import scservo_sdk as scs

        # 创建串口处理器 - 管理底层串口通信
        self.port_handler = scs.PortHandler(self.port)

        # 猴子补丁: 修复scservo_sdk的超时计算bug
        # 将patch_setPacketTimeout绑定到port_handler
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
            self.port_handler, scs.PortHandler
        )

        # 创建数据包处理器 - 处理协议封装/解析
        self.packet_handler = scs.PacketHandler(protocol_version)

        # 创建同步读取器 - 高效读取多个电机数据
        # 参数0,0表示起始地址和长度在后续调用中指定
        self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)

        # 创建同步写入器 - 高效写入多个电机数据
        self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

        # 缓存通信状态常量 - 避免频繁查表
        self._comm_success = scs.COMM_SUCCESS      # 通信成功状态码
        self._no_error = 0x00                        # 无错误状态码

        # 验证所有电机使用相同协议版本
        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise ValueError(f"Some motors are incompatible with protocol_version={self.protocol_version}")

    def _assert_same_protocol(self) -> None:
        """
        内部方法: 验证所有电机使用相同协议

        抛出:
        --------
        RuntimeError: 当检测到不同电机使用不同协议版本时

        注意:
        --------
        Protocol 0和Protocol 1不能混合使用
        因为它们使用不同的帧格式和通信时序
        """
        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise RuntimeError("Some motors use an incompatible protocol.")

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        """
        内部方法: 验证特定指令与当前协议兼容

        参数:
        -----------
        instruction_name : str
            指令名称,必须是以下之一:
            - "sync_read": 同步读取指令
            - "broadcast_ping": 广播ping指令

        异常:
        --------
        NotImplementedError: 当协议不支持该指令时

        注意:
        --------
        Protocol 1不支持sync_read和broadcast_ping
        需要改用轮询方式逐个读取/ping
        """
        if instruction_name == "sync_read" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Sync Read' is not available with Feetech motors using Protocol 1. Use 'Read' sequentially instead."
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Broadcast Ping' is not available with Feetech motors using Protocol 1. Use 'Ping' sequentially instead."
            )

    def _assert_same_firmware(self) -> None:
        """
        内部方法: 验证所有电机固件版本一致

        抛出:
        --------
        RuntimeError: 当检测到不同固件版本时

        用途:
        --------
        不同固件版本可能存在寄存器地址/行为差异
        统一固件版本可以避免兼容性问题

        解决建议:
        --------
        使用Feetech官方软件更新固件
        官网: https://www.feetechrc.com/software
        """
        firmware_versions = self._read_firmware_version(self.ids, raise_on_error=True)
        if len(set(firmware_versions.values())) != 1:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(firmware_versions)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    def _handshake(self) -> None:
        """
        内部方法: 执行电机握手/发现流程

        初始化时的电机存在性检查
        调用链: __init__ → configure_motors → _handshake
        """
        self._assert_motors_exist()
        self._assert_same_firmware()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """
        内部方法: 查找单个电机的ID和波特率

        参数:
        -----------
        motor : str
            电机名称(在motors字典中的键)

        initial_baudrate : int | None
            初始波特率,如果为None则遍历所有可能的波特率

        返回:
        --------
        tuple[int, int]
            (波特率, 电机ID)的元组

        抛出:
        --------
        RuntimeError: 找不到电机或型号不匹配时

        注意:
        --------
        Protocol 0和Protocol 1使用不同的发现算法
        """
        if self.protocol_version == 0:
            return self._find_single_motor_p0(motor, initial_baudrate)
        else:
            return self._find_single_motor_p1(motor, initial_baudrate)

    def _find_single_motor_p0(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """
        内部方法: Protocol 0方式查找单个电机

        Protocol 0使用broadcast_ping广播查询
        所有电机响应后逐一匹配型号

        参数: 同_find_single_motor

        返回: (波特率, 电机ID)
        """
        model = self.motors[motor].model
        # 确定要扫描的波特率列表
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            # 广播ping,返回{id: model_number}字典
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def _find_single_motor_p1(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """
        内部方法: Protocol 1方式查找单个电机

        Protocol 1需要逐个ID查询,因为不支持broadcast_ping

        参数: 同_find_single_motor

        返回: (波特率, 电机ID)
        """
        import scservo_sdk as scs

        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            # 遍历所有可能的ID(0-254)
            for id_ in range(scs.MAX_ID + 1):
                found_model = self.ping(id_)
                if found_model is not None:
                    if found_model != expected_model_nb:
                        raise RuntimeError(
                            f"Found one motor on {baudrate=} with id={id_} but it has a "
                            f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                            f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                        )
                    return baudrate, id_

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(
        self,
        return_delay_time: int = 0,
        maximum_acceleration: int = 254,
        acceleration: int = 254
    ) -> None:
        """
        配置电机基本参数

        在电机使用前调用,设置通信延迟和运动参数

        参数:
        -----------
        return_delay_time : int
            响应延迟时间,范围0-254
            实际延迟 = return_delay_time × 2µs
            默认0 = 最小延迟2µs
            注意: Feetech默认500µs(值250),这里优化到最小

        maximum_acceleration : int
            最大加速度,范围1-254
            值越大加速越快
            仅Protocol 0支持
            默认254

        acceleration : int
            加速度常数,范围0-254
            值越小加速越快
            默认254

        示例:
        -----------
        # 使用最快响应配置
        bus.configure_motors(return_delay_time=0, acceleration=254)
        """
        for motor in self.motors:
            # 设置响应延迟时间为最小值(2µs)
            # 默认500µs会造成不必要延迟
            self.write("Return_Delay_Time", motor, return_delay_time)

            # 设置最大加速度加快响应
            if self.protocol_version == 0:
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            self.write("Acceleration", motor, acceleration)

    @property
    def is_calibrated(self) -> bool:
        """
        属性: 检查校准是否有效

        返回:
        --------
        bool
            True = 校准有效, False = 未校准或校准数据不匹配

        检查项目:
        1. 校准文件是否存在
        2. 校准的电机与配置的电机是否一致
        3. 位置范围(range_min/max)是否匹配
        4. 零点偏移(homing_offset)是否匹配 (Protocol 0)
        """
        if not self.calibration:
            print("Calibration file not found!")
            return False
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            print("Calibration joints mismatch!")
            return False

        # 检查位置范围
        same_ranges = all(
            self.calibration[motor].range_min == cal.range_min
            and self.calibration[motor].range_max == cal.range_max
            for motor, cal in motors_calibration.items()
        )
        if self.protocol_version == 1:
            # Protocol 1不使用homing_offset
            if not same_ranges:
                print("Calibration ranges mismatch!")
            return same_ranges

        # Protocol 0还需要检查零点偏移
        same_offsets = all(
            self.calibration[motor].homing_offset == cal.homing_offset
            for motor, cal in motors_calibration.items()
        )
        if not same_offsets:
            print("Calibration offsets mismatch!")
        return same_ranges and same_offsets

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """
        从电机读取校准数据

        读取电机寄存器中的实际校准值

        返回:
        --------
        dict[str, MotorCalibration]
            电机名称到校准数据的字典

        读取的寄存器:
        -----------
        - Min_Position_Limit: 位置下限
        - Max_Position_Limit: 位置上限
        - Homing_Offset: 零点偏移 (仅Protocol 0)

        计算公式:
        -----------
        实际位置 = 寄存器位置 - Homing_Offset
        """
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            # 读取位置限位(原始值,不归一化)
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            # 读取零点偏移,Protocol 1没有此寄存器
            offsets[motor] = (
                self.read("Homing_Offset", motor, normalize=False) if self.protocol_version == 0 else 0
            )

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,                  # 电机ID
                drive_mode=0,             # 驱动模式(暂不支持)
                homing_offset=offsets[motor],  # 零点偏移
                range_min=mins[motor],    # 位置下限
                range_max=maxes[motor],   # 位置上限
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """
        将校准数据写入电机

        参数:
        -----------
        calibration_dict : dict[str, MotorCalibration]
            校准数据字典

        cache : bool
            是否缓存到实例,默认True
            设为False仅写入电机不更新内存

        写入的寄存器:
        -----------
        - Homing_Offset: 零点偏移 (仅Protocol 0)
        - Min_Position_Limit: 位置下限
        - Max_Position_Limit: 位置上限
        """
        for motor, calibration in calibration_dict.items():
            if self.protocol_version == 0:
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        内部方法: 计算半圈偏移的Home位置

        用于机械臂等需要计算对称位置的场景

        原理:
        --------
        假设电机当前位置在中间值(max_res/2),计算相对这个"假想零位"的偏移
        由于电机位置寄存器是循环的(0 ~ max_res-1),需要处理边界情况

        例如:
        - 电机最大分辨率4096,当前位置2000
        - 假想零位=2048
        - 目标偏移=2000-2048=-48

        参数:
        -----------
        positions : dict[NameOrID, Value]
            电机名称/ID到位置值的字典

        返回:
        --------
        dict[NameOrID, Value]
            计算后的偏移值字典
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1

            # 计算相对于中间位置的目标偏移
            target_offset = pos - int(max_res / 2)

            # 从编码表获取Homing_Offset的位数(符号位位置)
            encoding_table = self.model_encoding_table.get(model, {})
            homing_offset_bits = encoding_table.get("Homing_Offset", 11)  # 默认11位

            # 计算调整值: 2^(bits+1) 用于处理循环边界
            adjustment_value = 1 << (homing_offset_bits + 1)
            max_offset = (1 << homing_offset_bits) - 1  # 最大偏移 2^bits - 1

            # 确保偏移在合理范围内,超出则绕回
            # 这处理了位置寄存器循环的问题
            while target_offset > max_offset:
                target_offset -= adjustment_value
            while target_offset < -max_offset:
                target_offset += adjustment_value

            half_turn_homings[motor] = target_offset

        return half_turn_homings

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        禁用电机扭矩输出

        禁用后电机可以自由转动,常用于:
        - 手动调整位置
        - 校准过程
        - 紧急停止

        参数:
        -----------
        motors : str | list[str] | None
            要禁用的电机
            str: 单个电机名称
            list[str]: 多个电机名称列表
            None: 所有电机 (默认)

        num_retry : int
            通信失败重试次数,默认0

        示例:
        -----------
        # 禁用所有电机
        bus.disable_torque()

        # 禁用特定电机
        bus.disable_torque("shoulder_pan")
        bus.disable_torque(["shoulder_pan", "shoulder_lift"])
        """
        for motor in self._get_motors_list(motors):
            # 禁用扭矩使能
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)
            # 锁定电机参数,防止意外修改
            self.write("Lock", motor, 0, num_retry=num_retry)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        """
        内部方法: 禁用单个电机的扭矩(低级API)

        参数:
        -----------
        motor_id : int
            电机ID (非名称)

        model : str
            电机型号

        num_retry : int
            重试次数
        """
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)
        addr, length = get_address(self.model_ctrl_table, model, "Lock")
        self._write(addr, length, motor_id, 0, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        使能电机扭矩输出

        使能后电机保持当前位置,对抗外部力量

        参数:
        -----------
        motors : str | list[str] | None
            要使能的电机,同disable_torque
        num_retry : int
            通信失败重试次数,默认0

        示例:
        -----------
        bus.enable_torque()  # 使能所有电机
        """
        for motor in self._get_motors_list(motors):
            # 使能扭矩
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)
            # 解锁电机参数
            self.write("Lock", motor, 1, num_retry=num_retry)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        内部方法: 对数据进行符号编码

        用于将带符号整数编码为电机寄存器格式

        参数:
        -----------
        data_name : str
            数据名称,如"Homing_Offset", "Goal_Velocity"
            决定使用哪个符号位位置

        ids_values : dict[int, int]
            电机ID到值的字典 (就地修改)

        返回:
        --------
        dict[int, int]
            编码后的值字典

        符号编码示例:
        -----------
        假设符号位=11 (12位数)
        原值 +2048 → 0x0800 (不编码,正数)
        原值 -2048 → 0xF800 (最高位为1,负数)
        """
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """
        内部方法: 对数据进行符号解码

        用于将电机寄存器值解码为带符号整数

        参数:
        -----------
        data_name : str
            数据名称

        ids_values : dict[int, int]
            电机ID到原始值的字典 (就地修改)

        返回:
        --------
        dict[int, int]
            解码后的值字典
        """
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """
        内部方法: 将值拆分为字节列表

        是_split_into_byte_chunks函数的封装
        用于实例方法调用
        """
        return _split_into_byte_chunks(value, length)

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        """
        内部方法: 执行广播ping (Protocol 0)

        广播ping是电机发现的高效方式
        所有电机收到ping后同时响应,ID不同避免冲突

        返回:
        --------
        tuple[dict[int, int], int]
            (电机ID到错误状态的字典, 通信状态码)

        通信状态码:
        - COMM_SUCCESS: 成功
        - COMM_RX_TIMEOUT: 接收超时
        - COMM_RX_CORRUPT: 数据损坏
        """
        import scservo_sdk as scs

        data_list = {}  # 存储响应:{id: error_status}

        status_length = 6  # 每个响应包的长度(字节)

        rx_length = 0
        wait_length = status_length * scs.MAX_ID  # 最大可能接收长度

        txpacket = [0] * 6

        # 计算传输每个字节的时间(毫秒)
        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        # 构造ping包
        txpacket[scs.PKT_ID] = scs.BROADCAST_ID  # 广播ID (0xFE)
        txpacket[scs.PKT_LENGTH] = 2              # 长度: 仅指令
        txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING  # PING指令

        # 发送数据包
        result = self.packet_handler.txPacket(self.port_handler, txpacket)
        if result != scs.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list, result

        # 设置接收超时
        self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0)

        # 接收响应
        rxpacket = []
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

        self.port_handler.is_using = False

        if rx_length == 0:
            return data_list, scs.COMM_RX_TIMEOUT

        # 解析响应包
        while True:
            if rx_length < status_length:
                return data_list, scs.COMM_RX_CORRUPT

            # 查找包头(0xFF 0xFF)
            for idx in range(0, (rx_length - 1)):
                if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
                    break

            if idx == 0:  # 包头在起始位置
                # 校验checksum
                checksum = 0
                for idx in range(2, status_length - 1):  # 排除包头和校验和
                    checksum += rxpacket[idx]
                checksum = ~checksum & 0xFF

                if rxpacket[status_length - 1] == checksum:
                    result = scs.COMM_SUCCESS
                    # 提取电机ID和错误状态
                    data_list[rxpacket[scs.PKT_ID]] = rxpacket[scs.PKT_ERROR]
                    # 移除已解析的包
                    del rxpacket[0:status_length]
                    rx_length = rx_length - status_length
                    if rx_length == 0:
                        return data_list, result
                else:
                    result = scs.COMM_RX_CORRUPT
                    # 移除无效的包头
                    del rxpacket[0:2]
                    rx_length = rx_length - 2
            else:
                # 移除错误位置的包
                del rxpacket[0:idx]
                rx_length = rx_length - idx

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """
        广播ping电机发现

        高效发现总线上所有电机的ID和型号

        参数:
        -----------
        num_retry : int
            失败重试次数,默认0

        raise_on_error : bool
            是否在通信失败时抛出异常,默认False(返回None)

        返回:
        --------
        dict[int, int] | None
            成功: {电机ID: 型号编号}字典
            失败且raise_on_error=False: None

        示例:
        -----------
        result = bus.broadcast_ping()
        # {1: 777, 2: 2825} 表示ID=1是型号777(STS3215),ID=2是型号2825(STS3250)
        """
        self._assert_protocol_is_compatible("broadcast_ping")

        # 带重试的ping
        for n_try in range(1 + num_retry):
            ids_status, comm = self._broadcast_ping()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        # 检查电机错误状态
        ids_errors = {id_: status for id_, status in ids_status.items() if self._is_error(status)}
        if ids_errors:
            display_dict = {id_: self.packet_handler.getRxPacketError(err) for id_, err in ids_errors.items()}
            logger.error(f"Some motors found returned an error status:\n{pformat(display_dict, indent=4)}")

        # 读取并返回电机型号
        return self._read_model_number(list(ids_status), raise_on_error)

    def _read_firmware_version(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, str]:
        """
        内部方法: 读取电机固件版本

        参数:
        -----------
        motor_ids : list[int]
            要读取的电机ID列表

        raise_on_error : bool
            是否在读取失败时抛出异常

        返回:
        --------
        dict[int, str]
            电机ID到固件版本字符串的字典
            版本格式: "主版本.次版本" (如 "0.1")
        """
        firmware_versions = {}
        for id_ in motor_ids:
            # 读取主版本号
            firm_ver_major, comm, error = self._read(
                *FIRMWARE_MAJOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            # 读取次版本号
            firm_ver_minor, comm, error = self._read(
                *FIRMWARE_MINOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            # 组合版本字符串
            firmware_versions[id_] = f"{firm_ver_major}.{firm_ver_minor}"

        return firmware_versions

    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        """
        内部方法: 读取电机型号编号

        参数:
        -----------
        motor_ids : list[int]
            要读取的电机ID列表

        raise_on_error : bool
            是否在读取失败时抛出异常

        返回:
        --------
        dict[int, int]
            电机ID到型号编号的字典
            可与MODEL_NUMBER_TABLE比对得到具体型号名称
        """
        model_numbers = {}
        for id_ in motor_ids:
            model_nb, comm, error = self._read(*MODEL_NUMBER, id_, raise_on_error=raise_on_error)
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            model_numbers[id_] = model_nb

        return model_numbers
