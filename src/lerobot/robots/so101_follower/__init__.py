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
SO-101 Follower 机械臂模块

该模块是 SO-101 Follower 机械臂的入口点，提供以下公开接口：

类:
    SO101Follower: 机械臂主控制类
    SO101FollowerConfig: 机械臂配置类

导出说明:
    通过 `from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig`
    可以直接导入这两个类，无需关心内部实现细节。

配置示例:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.cameras import CameraConfig

    config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        disable_torque_on_disconnect=True,
        max_relative_target=0.1,
        cameras={
            "handeye": CameraConfig(type="opencv", index_or_path=0, width=640, height=360, fps=30),
            "fixed": CameraConfig(type="opencv", index_or_path=2, width=640, height=360, fps=30),
        },
        use_degrees=False,
    )

使用示例:
    robot = SO101Follower(config)
    robot.connect()
    obs = robot.get_observation()
    robot.send_action({"shoulder_pan.pos": 0.5, ...})
    robot.disconnect()
"""

from .config_so101_follower import SO101FollowerConfig
from .so101_follower import SO101Follower
