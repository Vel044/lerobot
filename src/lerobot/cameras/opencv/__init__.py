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
OpenCV相机模块入口文件。

功能说明：
    本模块提供使用OpenCV捕获摄像头帧的功能，主要包括：
    1. OpenCVCamera：核心相机类，负责连接、配置、读取摄像头
    2. OpenCVCameraConfig：相机配置数据类

使用示例：
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    # 创建配置并连接相机
    config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    camera = OpenCVCamera(config)
    camera.connect()

    # 同步读取一帧
    frame = camera.read()

    # 断开连接
    camera.disconnect()
"""

# 导出核心类供外部使用
from .camera_opencv import OpenCVCamera  # 核心OpenCV相机类
from .configuration_opencv import OpenCVCameraConfig  # 相机配置类
