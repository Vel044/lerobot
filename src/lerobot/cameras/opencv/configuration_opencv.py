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
OpenCV相机配置模块 - 定义OpenCV相机的配置数据类。

功能说明：
    本模块定义了OpenCV相机的配置参数，包括：
    1. 设备索引或视频文件路径
    2. 分辨率（宽度、高度）
    3. 帧率（FPS）
    4. 颜色模式（RGB或BGR）
    5. 图像旋转角度
    6. 预热时间

配置验证：
    __post_init__方法会验证color_mode和rotation参数的合法性，
    确保相机在连接前配置就是有效的。

依赖：
    - dataclasses: Python数据类装饰器
    - pathlib.Path: 文件路径处理（支持视频文件）
    - CameraConfig: 父级配置基类（定义通用配置项）
    - ColorMode: 颜色模式枚举（RGB/BGR）
    - Cv2Rotation: 旋转角度枚举
"""

from dataclasses import dataclass
from pathlib import Path

# 从父级configs模块导入相机配置的基类、颜色模式枚举和旋转枚举
# CameraConfig: 通用相机配置基类，包含width, height, fps等通用属性
# ColorMode: 颜色模式枚举，RGB或BGR
# Cv2Rotation: OpenCV旋转角度枚举，0°/90°/180°/270°
from ..configs import CameraConfig, ColorMode, Cv2Rotation


# 使用CameraConfig基类的register_subclass装饰器注册为"opencv"子类型
# 这样可以通过CameraConfig.from_dict({"type": "opencv", ...})方式创建实例
@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    OpenCV相机配置类 - 存储OpenCV相机的所有配置参数。

    继承自CameraConfig，提供OpenCV特有的配置项。

     Attributes (属性说明):
        index_or_path (int | Path):
            相机设备标识，可以是：
            - int: 相机设备索引，如0表示第一个相机
            - Path: 视频文件路径，用于读取视频而非相机
            注意：Linux下设备路径通常是/dev/video0等形式

        color_mode (ColorMode):
            输出图像的颜色模式，默认为ColorMode.RGB
            - ColorMode.RGB: 输出RGB格式（OpenCV默认是BGR，会自动转换）
            - ColorMode.BGR: 输出BGR格式（与OpenCV原生格式一致，无需转换）

        rotation (Cv2Rotation):
            图像旋转角度，默认为Cv2Rotation.NO_ROTATION (0°)
            - Cv2Rotation.NO_ROTATION: 不旋转
            - Cv2Rotation.ROTATE_90: 旋转90°
            - Cv2Rotation.ROTATE_180: 旋转180°
            - Cv2Rotation.ROTATE_270: 旋转270°

        warmup_s (int):
            连接相机后的预热时间（秒），默认1秒
            预热期间会多次读取帧以稳定相机输出
            对于有些相机需要若干帧才能达到正常曝光/对焦

    示例:
        # 使用相机索引，1280x720分辨率，30fps
        config = OpenCVCameraConfig(index_or_path=0, fps=30, width=1280, height=720)

        # 使用设备路径
        config = OpenCVCameraConfig(index_or_path="/dev/video4", fps=60, width=640, height=480)

        # 读取视频文件
        config = OpenCVCameraConfig(index_or_path=Path("video.mp4"), fps=30)

        # 带旋转和BGR输出
        config = OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.ROTATE_90
        )
    """

    # 设备索引或文件路径（必需参数，无默认值）
    # 类型: int(相机索引如0/1/2) 或 Path(视频文件路径)
    index_or_path: int | Path

    # 颜色模式：RGB或BGR，默认RGB（人类习惯颜色顺序）
    # OpenCV原生返回BGR格式，设置RGB会触发颜色通道转换
    color_mode: ColorMode = ColorMode.RGB

    # 旋转角度：0°/90°/180°/270°，默认不旋转
    # 用于校正相机安装方向导致的图像倒置
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION

    # 预热时间（秒）：连接后等待相机稳定的时间，默认1秒
    # 有些相机需要若干帧才能自动曝光/对焦稳定
    warmup_s: int = 1

    def __post_init__(self):
        """
        配置验证钩子 - 在对象创建后自动调用进行参数校验。

        验证逻辑：
            1. 检查color_mode是否为有效的RGB或BGR值
            2. 检查rotation是否为支持的旋转角度

        异常：
            ValueError: 当color_mode或rotation为无效值时抛出
        """
        # 验证颜色模式是否为有效的RGB或BGR
        # ColorMode枚举定义了RGB和BGR两种模式
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        # 验证旋转角度是否为支持的4种角度之一
        # Cv2Rotation枚举定义了NO_ROTATION/ROTATE_90/ROTATE_180/ROTATE_270
        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )
