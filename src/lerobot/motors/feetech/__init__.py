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
Feetech 电机驱动模块 - 初始化文件

本模块是Feetech品牌舵机/电机的驱动接口,继承自通用MotorsBus类。
Feetech是一家国产舵机品牌,产品包括STS系列、SCS系列、SMS系列等。

导出内容:
- DriveMode: 驱动模式枚举(正转/反转)
- FeetechMotorsBus: Feetech电机总线类,用于与电机通信
- OperatingMode: 运行模式枚举(位置/速度/PWM/步进模式)
- TorqueMode: 扭矩模式枚举(使能/禁用)
- tables模块中的所有常量(控制表、波特率表、编码表等)
"""

from .feetech import DriveMode, FeetechMotorsBus, OperatingMode, TorqueMode
from .tables import *
