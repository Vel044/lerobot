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
Feetech 电机控制表定义模块

本文件定义了Feetech各系列舵机的控制寄存器表(Control Table)、波特率表、编码表等配置信息。
控制表是电机通信的核心 - 每个寄存器由(地址, 字节长度)元组表示。

寄存器分类:
- EPROM区: 可读写,断电保存 (如ID、波特率、位置限位等)
- SRAM区: 可读写,断电丢失 (如扭矩使能、目标位置、当前位置等)
- 只读区: 只能读取 (如固件版本、当前温度等)

编码说明:
- 所有电机型号共用相同的寄存器地址,但字节长度可能不同
- STS/SMS系列: 位置精度4096 (12位分辨率)
- SCS系列: 位置精度1024 (10位分辨率)
"""

# ============================================================
# 固件版本寄存器地址 - 固件版本用于检查电机兼容性
# 格式:(地址, 字节长度)
# ============================================================
FIRMWARE_MAJOR_VERSION = (0, 1)  # 主版本号地址,占1字节
FIRMWARE_MINOR_VERSION = (1, 1)   # 次版本号地址,占1字节
MODEL_NUMBER = (3, 2)              # 型号寄存器地址,占2字节 (用于识别具体电机型号)

# TODO(Steven): 考虑使用枚举重构控制表,方便IDE类型检查和代码提示
# 示例: 使用枚举替代字符串key,可以在编译期检查错误的key名
# from enum import Enum
# class MyControlTableKey(Enum):
#   ID = "ID"
#   GOAL_SPEED = "Goal_Speed"
# MY_CONTROL_TABLE = {MyControlTableKey.ID.value: (5,1), ...}

# ============================================================
# STS/SMS 系列控制表 (Position Control Mode)
# 适用于: sts_series, sms_series, sts3215, sts3250, sm8512bl
# 官方手册: http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
# ============================================================

# EPROM区 (电可擦除可编程只读存储器,断电保存)
STS_SMS_SERIES_CONTROL_TABLE = {
    # 只读寄存器
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # 固件主版本号,只读
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # 固件次版本号,只读
    "Model_Number": MODEL_NUMBER,                      # 电机型号编号,只读

    # 基本配置
    "ID": (5, 1),                    # 电机ID地址,1字节,范围0-254
    "Baud_Rate": (6, 1),            # 波特率配置,1字节,对应波特率表
    "Return_Delay_Time": (7, 1),    # 响应延迟时间,1字节,默认250(500µs)
    "Response_Status_Level": (8, 1), # 响应状态级别,控制何时返回状态包

    # 位置限位 (安全保护)
    "Min_Position_Limit": (9, 2),   # 最小位置限位,2字节,防止超调
    "Max_Position_Limit": (11, 2),  # 最大位置限位,2字节,防止超调

    # 保护限位
    "Max_Temperature_Limit": (13, 1), # 最大温度限位,1字节,超过此温度报警(单位:℃)
    "Max_Voltage_Limit": (14, 1),      # 最大电压限位,1字节,超过此电压报警(单位:0.1V)
    "Min_Voltage_Limit": (15, 1),      # 最小电压限位,1字节,低于此电压报警(单位:0.1V)
    "Max_Torque_Limit": (16, 2),       # 最大扭矩限位,2字节,限制输出扭矩

    # 相位与卸荷
    "Phase": (18, 1),               # 相位配置,用于电机换向
    "Unloading_Condition": (19, 1), # 卸荷条件,控制何时卸除扭矩

    # 报警配置
    "LED_Alarm_Condition": (20, 1), # LED报警条件,各bit表示不同报警类型

    # PID系数 (位置/速度闭环控制)
    "P_Coefficient": (21, 1),       # 比例系数P,1字节,范围0-255
    "D_Coefficient": (22, 1),       # 微分系数D,1字节,范围0-255
    "I_Coefficient": (23, 1),       # 积分系数I,1字节,范围0-255

    # 启动与死区
    "Minimum_Startup_Force": (24, 2),  # 最小启动扭矩,2字节,克服静摩擦
    "CW_Dead_Zone": (26, 1),            # 顺时针死区,1字节,顺时针方向最小指令
    "CCW_Dead_Zone": (27, 1),           # 逆时针死区,1字节,逆时针方向最小指令

    # 保护电流
    "Protection_Current": (28, 2),       # 保护电流阈值,2字节

    # 角度分辨率
    "Angular_Resolution": (30, 1),       # 角度分辨率,1字节

    # 原点偏移 (关键!用于多电机同步时的相对位置校准)
    "Homing_Offset": (31, 2),           # 原点偏移,2字节,范围-2047~2048
                                        # 计算公式: Present_Position = Actual_Position - Homing_Offset

    # 运行模式
    "Operating_Mode": (33, 1),          # 运行模式,见OperatingMode枚举

    # 保护参数
    "Protective_Torque": (34, 1),       # 防护扭矩,1字节
    "Protection_Time": (35, 1),          # 保护时间,1字节,触发保护的持续时间
    "Overload_Torque": (36, 1),         # 过载扭矩,1字节

    # 速度环PID (仅速度模式)
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1), # 速度环P
    "Over_Current_Protection_Time": (38, 1),                   # 过流保护时间
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),  # 速度环I

    # ============================================================
    # SRAM区 (静态随机存储器,断电丢失,运行时可修改)
    # ============================================================

    # 扭矩控制 (最常用!)
    "Torque_Enable": (40, 1),          # 扭矩使能,1字节,1=使能扭矩输出,0=禁用(电机可自由转动)
    "Acceleration": (41, 1),           # 加速时间常数,1字节,0-254,值越小加速越快

    # 位置控制 (目标位置)
    "Goal_Position": (42, 2),          # 目标位置,2字节,范围0-4095(STS系列)
    "Goal_Time": (44, 2),              # 目标运行时间,2字节,单位ms
    "Goal_Velocity": (46, 2),          # 目标速度,2字节,用于速度约束的位置模式

    # 扭矩限制
    "Torque_Limit": (48, 2),           # 扭矩限制,2字节,与Max_Torque_Limit联动

    # 锁定电机 (写入0x00锁定,防止参数被意外修改)
    "Lock": (55, 1),                   # 锁定寄存器,写入0可锁定,0x00以外的值为解锁

    # 只读: 当前位置/速度/负载 (实时反馈)
    "Present_Position": (56, 2),       # 当前实际位置,2字节,只读
    "Present_Velocity": (58, 2),       # 当前实际速度,2字节,只读
    "Present_Load": (60, 2),           # 当前负载,2字节,只读
    "Present_Voltage": (62, 1),       # 当前电压,1字节,只读,单位0.1V
    "Present_Temperature": (63, 1),   # 当前温度,1字节,只读,单位℃
    "Status": (65, 1),                 # 状态寄存器,只读,包含错误标志
    "Moving": (66, 1),                 # 运动状态,只读,1=正在运动,0=停止
    "Present_Current": (69, 2),       # 当前实际电流,2字节,只读

    # 工厂参数 (出厂预设,一般不修改)
    "Moving_Velocity": (80, 1),        # 运动速度
    "Moving_Velocity_Threshold": (80, 1), # 运动速度阈值
    "DTs": (81, 1),                    # DTs参数,单位ms
    "Velocity_Unit_factor": (82, 1),  # 速度单位因子
    "Hts": (83, 1),                    # Hts参数,单位ns,仅固件>=2.54有效
    "Maximum_Velocity_Limit": (84, 1), # 最大速度限位
    "Maximum_Acceleration": (85, 1),  # 最大加速度
    "Acceleration_Multiplier ": (86, 1), # 加速度乘数,当acceleration为0时生效
}

# ============================================================
# SCS 系列控制表 (另一种协议,Protocol 1)
# 适用于: scs_series, scs0009
# 官方手册: http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SCSCL-emanual-cbcc8ab2e3384282a01d4bf3
# ============================================================
SCS_SERIES_CONTROL_TABLE = {
    # EPROM区
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,
    "Model_Number": MODEL_NUMBER,
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protective_Torque": (37, 1),      # 注意:地址与STS系列不同!
    "Protection_Time": (38, 1),

    # SRAM区
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Running_Time": (44, 2),           # 注意: SCS用Running_Time而非Goal_Time
    "Goal_Velocity": (46, 2),
    "Lock": (48, 1),                  # 注意: SCS的Lock地址是48,而非55

    # 只读寄存器
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Sync_Write_Flag": (64, 1),
    "Status": (65, 1),
    "Moving": (66, 1),

    # 工厂参数
    "PWM_Maximum_Step": (78, 1),
    "Moving_Velocity_Threshold*50": (79, 1),
    "DTs": (80, 1),
    "Minimum_Velocity_Limit*50": (81, 1),
    "Maximum_Velocity_Limit*50": (82, 1),
    "Acceleration_2": (83, 1),
}

# ============================================================
# 波特率配置表 - 将波特率值转换为电机寄存器中的索引
# 格式: {实际波特率: 寄存器值}
# ============================================================
STS_SMS_SERIES_BAUDRATE_TABLE = {
    1_000_000: 0,   # 1Mbps
    500_000: 1,     # 500Kbps
    250_000: 2,     # 250Kbps
    128_000: 3,     # 128Kbps
    115_200: 4,     # 115.2Kbps
    57_600: 5,      # 57.6Kbps
    38_400: 6,      # 38.4Kbps
    19_200: 7,      # 19.2Kbps
}

SCS_SERIES_BAUDRATE_TABLE = {
    # SCS系列使用相同的波特率表
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

# ============================================================
# 型号控制表映射 - 将电机型号映射到对应的控制表
# ============================================================
MODEL_CONTROL_TABLE = {
    "sts_series": STS_SMS_SERIES_CONTROL_TABLE,  # STS通用系列
    "scs_series": SCS_SERIES_CONTROL_TABLE,        # SCS通用系列
    "sms_series": STS_SMS_SERIES_CONTROL_TABLE,   # SMS系列(与STS相同)
    "sts3215": STS_SMS_SERIES_CONTROL_TABLE,      # STS3215型号
    "sts3250": STS_SMS_SERIES_CONTROL_TABLE,      # STS3250型号
    "scs0009": SCS_SERIES_CONTROL_TABLE,          # SCS0009型号
    "sm8512bl": STS_SMS_SERIES_CONTROL_TABLE,     # SM8512BL型号
}

# ============================================================
# 电机分辨率表 - 表示电机一圈的脉冲/刻度数量
# 这决定了位置精度: 分辨率越高,位置控制越精确
# ============================================================
MODEL_RESOLUTION = {
    "sts_series": 4096,   # STS系列: 12位分辨率,一圈4096个刻度
    "sms_series": 4096,   # SMS系列: 12位分辨率
    "scs_series": 1024,   # SCS系列: 10位分辨率,一圈1024个刻度
    "sts3215": 4096,      # STS3215: 12位分辨率
    "sts3250": 4096,      # STS3250: 12位分辨率
    "sm8512bl": 4096,     # SM8512BL: 12位分辨率
    "scs0009": 1024,      # SCS0009: 10位分辨率
}

# ============================================================
# 电机波特率表映射 - 将型号映射到波特率配置表
# ============================================================
MODEL_BAUDRATE_TABLE = {
    "sts_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sms_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sm8512bl": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3215": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3250": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs0009": SCS_SERIES_BAUDRATE_TABLE,
}

# ============================================================
# 符号-幅度编码表 (Sign-Magnitude Encoding)
# 用于处理带符号数据的高位表示方式
#
# 背景: 标准二进制中,最高位为1表示负数。但某些电机寄存器
#       使用"符号-幅度"编码:最高位是独立的符号位,其余位表示幅度
# 例如: 16位寄存器中, 0x8001 表示 +1, 0xFFFF 表示 -1
#
# 编码表指定了各型号电机的"符号位"位置(从右数第几位)
# ============================================================
STS_SMS_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": 11,      # 原点偏移: 11位符号 (范围-2047~2048)
    "Goal_Velocity": 15,     # 目标速度: 15位符号 (最高位为方向位)
    "Present_Velocity": 15,  # 当前速度: 15位符号
    "Present_Position": 15,  # 当前位置: 15位符号
}

# 合并到型号映射表
MODEL_ENCODING_TABLE = {
    "sts_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sms_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs_series": {},         # SCS系列不使用符号编码,留空
    "sts3215": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sts3250": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sm8512bl": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs0009": {},
}

# ============================================================
# 电机扫描波特率列表 - 自动发现电机时尝试的波特率
# 按从高到低的顺序扫描,高性能率放在前面可以加快发现速度
# ============================================================
SCAN_BAUDRATES = [
    4_800,     # 最低波特率
    9_600,
    14_400,
    19_200,
    38_400,
    57_600,
    115_200,
    128_000,
    250_000,
    500_000,
    1_000_000, # 最高波特率,默认推荐
]

# ============================================================
# 电机型号编号表 - 用于识别连接的电机具体型号
# 通过读取MODEL_NUMBER寄存器获取,与此表比对确认型号
# ============================================================
MODEL_NUMBER_TABLE = {
    "sts3215": 777,    # STS3215型号编号
    "sts3250": 2825,   # STS3250型号编号
    "sm8512bl": 11272, # SM8512BL型号编号
    "scs0009": 1284,   # SCS0009型号编号
}

# ============================================================
# 电机协议版本表 - 指定每种型号使用的通信协议
# Protocol 0: 用于STS/SMS系列 (半双工通信)
# Protocol 1: 用于SCS系列 (全双工通信)
# ============================================================
MODEL_PROTOCOL = {
    "sts_series": 0,   # STS通用系列使用协议0
    "sms_series": 0,   # SMS系列使用协议0
    "scs_series": 1,   # SCS通用系列使用协议1
    "sts3215": 0,      # STS3215使用协议0
    "sts3250": 0,      # STS3250使用协议0
    "sm8512bl": 0,     # SM8512BL使用协议0
    "scs0009": 1,      # SCS0009使用协议1
}
