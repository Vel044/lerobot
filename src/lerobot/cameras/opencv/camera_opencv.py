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
OpenCV相机核心实现模块 - 提供基于OpenCV的相机捕获功能。

功能说明：
    本模块是OpenCV相机模块的核心，实现了以下功能：
    1. 相机连接与断开管理
    2. 同步帧读取（阻塞式）
    3. 异步帧读取（非阻塞式，后台线程）
    4. 图像后处理（颜色转换、旋转、尺寸验证）
    5. 相机发现与枚举

架构设计：
    - 继承自Camera基类
    - 使用VideoCapture进行底层图像采集
    - 可选后台线程实现异步读取
    - 线程安全的帧存储（Lock/Event）

线程安全：
    - 帧读取和存储使用threading.Lock保护
    - 新帧通知使用threading.Event

异常处理：
    - DeviceAlreadyConnectedError: 重复连接已连接的相机
    - DeviceNotConnectedError: 对未连接相机进行操作
    - ConnectionError: 无法打开相机
    - RuntimeError: 配置参数应用失败或读取失败
    - TimeoutError: 异步读取超时

依赖：
    - cv2 (OpenCV): 底层图像采集库
    - numpy: 图像数据格式
    - threading: 异步读取的后台线程

作者: HuggingFace Inc.
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

# Fix MSMF hardware transform compatibility for Windows before importing cv2
# 修复Windows平台MSMF后端的硬件加速兼容性问题
# MSMF (Microsoft Media Foundation) 是Windows的媒体处理框架
# 设置为0可以避免某些情况下的崩溃或性能问题
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

# 从lerobot错误模块导入自定义异常类
# DeviceAlreadyConnectedError: 设备已连接时尝试再次连接
# DeviceNotConnectedError: 设备未连接时尝试操作
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# 从父级camera模块导入Camera基类
from ..camera import Camera
# 从上级utils模块导入OpenCV工具函数
# get_cv2_backend(): 获取OpenCV推荐的视频捕获后端
# get_cv2_rotation(): 将Cv2Rotation枚举转换为cv2旋转常量
from ..utils import get_cv2_backend, get_cv2_rotation
# 从本模块导入配置类
from .configuration_opencv import ColorMode, OpenCVCameraConfig


# NOTE(Steven): 最大OpenCV设备索引的说明
# 在不同操作系统上，相机设备索引的分配方式不同：
# - MacOS: 如果有3个相机，索引依次为0, 1, 2（连续）
# - Ubuntu(Linux): 索引可能不连续，如6, 16, 23
# - USB端口变化或重启后，索引可能发生变化
# 因此本常量设置为较高值60以覆盖可能的设备范围
MAX_OPENCV_INDEX = 60

# 获取本模块的日志记录器
logger = logging.getLogger(__name__)


class OpenCVCamera(Camera):
    """
    OpenCV相机类 - 使用OpenCV管理相机交互和帧采集。

    继承自Camera基类，提供基于OpenCV VideoCapture的高效帧录制功能。

    功能特性：
        1. 支持物理相机设备和视频文件两种输入源
        2. 支持同步读取（阻塞）和异步读取（非阻塞）两种模式
        3. 支持配置FPS、分辨率、颜色模式、旋转角度
        4. 异步模式使用独立后台线程持续采集最新帧

    使用方式：
        1. 创建配置对象OpenCVCameraConfig
        2. 创建相机实例并传入配置
        3. 调用connect()连接相机
        4. 使用read()同步读取或async_read()异步读取帧
        5. 调用disconnect()断开连接释放资源

    帧数据格式：
        返回的帧为numpy.ndarray，shape为(height, width, channels)
        - channels=3 表示彩色图像（RGB或BGR）
        - 像素值范围0-255（uint8）
        - 经过配置的rotation旋转后输出

    注意：
        - 树莓派等ARM平台上，USB相机索引可能不稳定
        - 建议使用设备路径（如/dev/video0）而非索引
        - 可使用lerobot-find-cameras opencv命令查找可用相机

    示例:
        ```python
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig, ColorMode, Cv2Rotation

        # 基础用法：使用相机索引0
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        camera.connect()

        # 同步读取一帧（阻塞等待）
        color_image = camera.read()
        print(color_image.shape)  # e.g., (480, 640, 3)

        # 异步读取一帧（非阻塞）
        async_image = camera.async_read()

        # 断开连接
        camera.disconnect()

        # 自定义配置示例
        custom_config = OpenCVCameraConfig(
            index_or_path='/dev/video0',  # 或使用索引如0
            fps=30,                         # 目标帧率30fps
            width=1280,                     # 宽度1280像素
            height=720,                     # 高度720像素
            color_mode=ColorMode.RGB,       # 输出RGB格式
            rotation=Cv2Rotation.ROTATE_90  # 旋转90度
        )
        custom_camera = OpenCVCamera(custom_config)
        custom_camera.connect()
        frame = custom_camera.read()
        custom_camera.disconnect()
        ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        初始化OpenCV相机实例。

        初始化逻辑：
            1. 调用父类Camera的初始化方法
            2. 从配置对象复制各项参数
            3. 初始化OpenCV VideoCapture对象为None
            4. 设置MJPG四字符编码（常见的视频编码格式）
            5. 初始化线程同步原语（Lock, Event）
            6. 计算旋转后的捕获尺寸

        传入参数:
            config (OpenCVCameraConfig): OpenCV相机的配置对象
                - index_or_path: 设备索引(int)或视频文件路径(Path)
                - fps: 目标帧率
                - width/height: 分辨率
                - color_mode: RGB或BGR
                - rotation: 旋转角度
                - warmup_s: 预热秒数

        初始化属性说明:
            videocapture (cv2.VideoCapture | None):
                OpenCV视频捕获对象，None表示未连接
            fourcc (cv2.VideoWriter_fourcc):
                四字符编码格式，这里使用MJPG(Motion-JPEG)
                'M','J','P','G'四个字符对应VideoWriter_fourcc类型
            thread (Thread | None):
                异步读取的后台线程对象
            stop_event (Event):
                线程停止信号事件
            frame_lock (Lock):
                保护最新帧的线程锁
            latest_frame (np.ndarray | None):
                异步模式下最新采集的帧
            new_frame_event (Event):
                新帧到达通知事件
            rotation (int | None):
                cv2旋转常量，用于cv2.rotate()
            backend (int):
                OpenCV视频捕获后端ID

        旋转尺寸计算逻辑:
            如果配置了90度或270度旋转，捕获时宽高需要交换
            因为旋转后会从竖屏变横屏或反之
        """
        # 调用父类初始化方法
        super().__init__(config)

        # 保存配置对象引用
        self.config = config
        # 保存设备索引或路径（用于日志和唯一标识）
        self.index_or_path = config.index_or_path

        # 从配置复制运行时参数
        self.fps = config.fps              # 目标帧率
        self.color_mode = config.color_mode  # 颜色模式
        self.warmup_s = config.warmup_s    # 预热秒数

        # 初始化OpenCV VideoCapture对象为None（未连接状态）
        self.videocapture: cv2.VideoCapture | None = None

        # 设置四字符编码为MJPG
        # MJPG是Motion-JPEG，适合实时传输，分辨率支持好
        # 其他常见选项：'YUYV', 'UYVY', 'GRBG'等
        self.fourcc: cv2.VideoWriter_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        # 异步读取相关初始化
        self.thread: Thread | None = None          # 后台读取线程
        self.stop_event: Event | None = None       # 停止信号
        self.frame_lock: Lock = Lock()              # 帧数据保护锁
        self.latest_frame: np.ndarray | None = None # 最新帧缓存
        self.new_frame_event: Event = Event()       # 新帧通知

        # 将配置中的Cv2Rotation枚举转换为cv2旋转常量
        # get_cv2_rotation返回cv2.ROTATE_90_CLOCKWISE等常量
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        # 获取OpenCV推荐的视频捕获后端
        # 不同平台后端不同：Linux用V4L2，MacOS用AVFOUNDATION，Windows用MSMF
        self.backend: int = get_cv2_backend()

        # 计算捕获尺寸（考虑旋转）
        if self.height and self.width:
            # 记录原始配置尺寸
            self.capture_width, self.capture_height = self.width, self.height
            # 如果配置了90度或270度旋转，需要交换宽高
            # 因为竖屏相机旋转后会变成横屏
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        """
        返回相机的字符串表示。

        返回格式: "OpenCVCamera(设备索引或路径)"
        用于日志和调试输出。
        """
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """
        检查相机是否已连接并打开。

        判断逻辑：
            1. videocapture对象存在
            2. VideoCapture.isOpened()返回True

        返回:
            bool: True表示已连接，False表示未连接
        """
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True):
        """
        连接OpenCV相机。

        连接流程：
            1. 检查是否已连接（避免重复连接）
            2. 设置OpenCV线程数为1（避免多线程冲突）
            3. 创建VideoCapture对象打开相机
            4. 验证相机是否成功打开
            5. 应用配置的捕获参数（FPS、分辨率等）
            6. 可选：执行预热流程（多次读取帧使相机稳定）

        传入参数:
            warmup (bool): 是否执行预热流程，默认True
                预热期间会多次调用read()读取帧

        异常:
            DeviceAlreadyConnectedError:
                如果相机已经连接，尝试再次连接时抛出
            ConnectionError:
                无法打开指定索引/路径的相机时抛出
                可能原因：设备不存在、设备被占用、权限不足
            RuntimeError:
                相机打开成功但无法应用配置的参数时抛出

        资源管理:
            连接失败时会自动释放已分配的VideoCapture资源
        """
        # 检查是否已连接
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # 设置OpenCV使用单线程
        # 原因：避免多线程环境下OpenCV操作冲突
        # 特别是在数据采集等高并发场景下，单线程更稳定
        cv2.setNumThreads(1)

        # 创建VideoCapture对象
        # index_or_path可以是int(设备索引如0)或str(Path，设备路径如"/dev/video0")
        self.videocapture = cv2.VideoCapture(self.index_or_path)

        # 验证相机是否成功打开
        if not self.videocapture.isOpened():
            # 打开失败，释放资源并抛出异常
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        # 应用配置的捕获参数（FPS、分辨率等）
        self._configure_capture_settings()

        # 执行预热流程
        if warmup:
            start_time = time.time()
            # 在指定预热时间内持续读取帧
            while time.time() - start_time < self.warmup_s:
                self.read()  # 同步读取（会阻塞）
                time.sleep(0.1)  # 短暂休眠避免太频繁

        logger.info(f"{self} connected.")

    def _configure_capture_settings(self) -> None:
        """
        应用相机捕获参数。

        配置流程：
            1. 获取相机默认分辨率（用于未指定分辨率时）
            2. 如果未指定分辨率，使用默认值
            3. 验证并设置四字符编码（fourcc）
            4. 验证并设置FPS（如果指定了）

        异常:
            DeviceNotConnectedError:
                相机未连接时调用此方法
            RuntimeError:
                无法设置FPS、分辨率或编码格式时抛出
        """
        # 检查相机是否已连接
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")

        # 获取相机当前默认分辨率
        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 如果配置中未指定分辨率，使用相机默认值
        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            # 如果配置了旋转且为90°/270°，需要交换默认宽高
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            # 配置中指定了分辨率，进行验证
            self._validate_width_and_height()

        # 验证并设置编码格式
        self._validate_fourcc()

        # 设置FPS
        if self.fps is None:
            # 未指定FPS，使用相机默认值
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            # 指定了FPS，进行验证
            self._validate_fps()

    def _validate_fps(self) -> None:
        """
        验证并设置相机帧率。

        验证逻辑：
            1. 尝试设置目标FPS
            2. 读取实际FPS
            3. 比较目标值和实际值（允许微小误差1e-3）

        异常:
            RuntimeError: 设置FPS失败或实际FPS与目标不符
        """
        # 尝试设置FPS
        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        # 读取实际FPS
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        # 使用math.isclose进行浮点数比较（考虑精度误差）
        # rel_tol=1e-3 表示相对误差容忍度为0.1%
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_width_and_height(self) -> None:
        """
        验证并设置相机捕获分辨率。

        验证逻辑：
            1. 尝试设置宽度和高度
            2. 读取实际宽度和高度
            3. 比较实际值与目标值

        异常:
            RuntimeError: 设置分辨率失败或实际分辨率不符
        """
        # 尝试设置宽度
        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        # 尝试设置高度
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        # 读取实际宽度
        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        # 读取实际高度
        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    def _validate_fourcc(self) -> None:
        """
        验证并设置相机编码格式。

        验证逻辑：
            1. 尝试设置fourcc编码
            2. 读取实际编码
            3. 比较实际编码与目标编码

        异常:
            RuntimeError: 设置编码失败或实际编码不符
        """
        # 尝试设置fourcc编码
        fourcc_succ = self.videocapture.set(cv2.CAP_PROP_FOURCC, self.fourcc)
        # 读取实际fourcc编码（返回的是整数形式的fourcc码）
        actual_fourcc = self.videocapture.get(cv2.CAP_PROP_FOURCC)
        if not fourcc_succ or actual_fourcc != self.fourcc:
            raise RuntimeError(f"{self} failed to set fourcc={self.fourcc} ({actual_fourcc=}, {fourcc_succ=}).")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        发现系统中可用的OpenCV相机。

        扫描策略：
            - Linux: 扫描/dev/video*路径
            - 其他系统(Mac/Windows): 扫描索引0-MAX_OPENCV_INDEX

        返回数据结构:
            list[dict[str, Any]]: 相机信息列表，每个字典包含：
                - name: 显示名称
                - type: 相机类型（固定为"OpenCV"）
                - id: 设备索引或路径
                - backend_api: OpenCV后端名称
                - default_stream_profile: 默认流配置
                    - format: 像素格式
                    - width: 默认宽度
                    - height: 默认高度
                    - fps: 默认帧率

        性能注意:
            每次调用都会打开和关闭每个设备探测
            频繁调用可能影响相机状态

        示例返回:
            [
                {
                    "name": "OpenCV Camera @ 0",
                    "type": "OpenCV",
                    "id": 0,
                    "backend_api": "V4L2",
                    "default_stream_profile": {
                        "format": 0.0,
                        "width": 640,
                        "height": 480,
                        "fps": 30.0
                    }
                }
            ]
        """
        found_cameras_info = []

        # 根据操作系统选择扫描方式
        if platform.system() == "Linux":
            # Linux: 扫描/dev/video*路径
            # Path.glob返回匹配路径列表，sorted按名称排序
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            # 转换为字符串列表
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            # MacOS/Windows: 扫描数字索引
            targets_to_scan = list(range(MAX_OPENCV_INDEX))

        # 遍历每个可能的设备
        for target in targets_to_scan:
            # 尝试打开设备
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                # 打开成功，读取设备信息
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                # 释放资源（探测完成后关闭）
                camera.release()

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        同步读取一帧图像（阻塞式）。

        读取流程：
            1. 检查相机是否已连接
            2. 调用VideoCapture.read()获取下一帧
            3. 验证帧是否有效
            4. 进行图像后处理（颜色转换、旋转）

        传入参数:
            color_mode (ColorMode | None):
                指定这帧的颜色模式，可选参数
                - None: 使用实例默认的self.color_mode
                - ColorMode.RGB: 输出RGB格式
                - ColorMode.BGR: 输出BGR格式
                用于临时覆盖配置的场景

        返回:
            np.ndarray: 处理的图像帧
                - shape: (height, width, 3) 表示彩色图像
                - dtype: uint8 像素值0-255
                - 颜色顺序由color_mode决定
                - 已应用rotation旋转

        异常:
            DeviceNotConnectedError: 相机未连接
            RuntimeError: 读取帧失败或帧尺寸不符
            ValueError: 指定了无效的color_mode

        性能注意:
            这是阻塞调用，会等待下一帧到达
            30fps相机约33ms返回一次
            建议使用async_read()进行异步读取
        """
        # 检查相机连接状态
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        # 从OpenCV读取下一帧
        # ret: bool 是否成功读取
        # frame: np.ndarray 原始图像（BGR格式）
        ret, frame = self.videocapture.read()

        # 验证读取结果
        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        # 进行图像后处理（颜色转换、旋转、尺寸验证）
        processed_frame = self._postprocess_image(frame, color_mode)

        # 记录读取耗时（毫秒）
        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        # 可选的调试日志
        # logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        对原始图像进行后处理。

        处理流程：
            1. 确定目标颜色模式（参数优先，其次实例配置）
            2. 验证颜色模式合法性
            3. 验证图像尺寸与配置一致
            4. 验证图像通道数为3
            5. 如果需要，进行BGR到RGB的颜色转换
            6. 如果配置了旋转，进行图像旋转

        传入参数:
            image (np.ndarray): 原始图像
                - 通常是BGR格式（OpenCV默认）
                - shape: (height, width, 3)
            color_mode (ColorMode | None):
                目标颜色模式
                - None: 使用self.color_mode
                - ColorMode.RGB: 转换为RGB
                - ColorMode.BGR: 保持BGR

        返回:
            np.ndarray: 处理后的图像
                - 已颜色转换
                - 已应用旋转
                - shape: (height, width, 3)

        异常:
            ValueError: color_mode无效
            RuntimeError: 图像尺寸或通道数不符
        """
        # 确定使用的颜色模式（参数优先）
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        # 验证颜色模式
        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        # 获取图像尺寸
        h, w, c = image.shape

        # 验证图像尺寸与捕获配置一致
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        # 验证通道数（只支持彩色图像）
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        # 颜色转换：BGR -> RGB（如果需要）
        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            # cv2.cvtColor进行颜色空间转换
            # COLOR_BGR2RGB: BGR格式转RGB格式
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用旋转（如果有配置）
        # 支持90°、270°（需要交换宽高）和180°旋转
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        异步读取的后台循环（运行在线程中）。

        循环逻辑：
            1. 检查停止信号（stop_event）
            2. 调用read()获取新帧
            3. 线程安全地更新latest_frame
            4. 通知新帧到达（new_frame_event）
            5. 异常处理：
               - DeviceNotConnectedError: 停止循环
               - 其他异常: 记录警告并继续

        线程安全：
            - 使用frame_lock保护latest_frame的读写
            - 使用new_frame_event通知主线程

        异常处理：
            - 设备断开连接：正常退出循环
            - 其他读取错误：记录警告不中断循环
        """
        while not self.stop_event.is_set():
            try:
                # 同步读取一帧
                color_image = self.read()

                # 线程安全地更新最新帧
                with self.frame_lock:
                    self.latest_frame = color_image
                # 通知等待者新帧已到达
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                # 设备断开，退出循环
                break
            except Exception as e:
                # 其他异常，记录警告
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """
        启动或重启异步读取线程。

        启动逻辑：
            1. 如果已有线程且在运行，等待其结束（最多0.1秒）
            2. 设置停止事件
            3. 创建并启动新线程

        线程配置：
            - daemon=True: 随主线程自动结束
            - name: 线程名称（包含相机标识）
        """
        # 如果线程存在且在运行，等待其结束
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        # 设置停止事件
        if self.stop_event is not None:
            self.stop_event.set()

        # 创建新的停止事件
        self.stop_event = Event()
        # 创建后台线程，目标为_read_loop方法
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True  # 守护线程
        self.thread.start()  # 启动线程

    def _stop_read_thread(self) -> None:
        """
        停止异步读取线程。

        停止逻辑：
            1. 设置停止事件（信号线程退出）
            2. 等待线程结束（最多2秒）
            3. 清理线程相关资源
        """
        # 设置停止事件
        if self.stop_event is not None:
            self.stop_event.set()

        # 等待线程结束
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # 清理资源
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        异步读取最新帧（非阻塞式）。

        异步机制：
            1. 后台线程持续调用read()采集帧
            2. 主线程通过async_read()获取最新已采集的帧
            3. 使用Event实现新帧通知

        传入参数:
            timeout_ms (float):
                等待新帧的最大超时时间（毫秒）
                默认200ms（约5fps的控制周期）
                如果最近已有帧，可能立即返回

        返回:
            np.ndarray: 最新的图像帧
                - shape: (height, width, 3)
                - 已应用颜色转换和旋转
                - 可能不是最新采集的帧，而是最近一次写入的帧

        异常:
            DeviceNotConnectedError: 相机未连接
            TimeoutError: 等待超时没有新帧
            RuntimeError: 内部错误（Event被设置但没有帧）

        性能提示:
            - 不会阻塞等待相机硬件
            - 只阻塞等待后台线程写入的帧
            - timeout应大于帧间隔（如30fps为33ms）
            - 建议超时时间设为帧间隔的5-6倍
        """
        # 检查连接状态
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 确保后台线程正在运行
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # 记录等待开始时间
        wait_start = time.perf_counter()
        has_cached_frame = self.latest_frame is not None

        # 等待新帧事件
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            # 等待超时
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )
        wait_ms = (time.perf_counter() - wait_start) * 1e3

        # 线程安全地获取最新帧
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()  # 清除事件状态

        # 验证帧存在
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        # 调试信息：
        # wait_ms: 实际等待时间
        # has_cached_frame: 等待前是否已有缓存帧
        # 如果等待时间接近帧间隔（约33ms@30fps），说明控制循环在等待相机
        # logger.debug(
        #     f"{self} async_read wait={wait_ms:.1f}ms (cached_before_wait={has_cached_frame})"
        # )

        return frame

    def disconnect(self):
        """
        断开相机连接并释放资源。

        断开流程：
            1. 检查连接状态（至少有其一项：is_connected或thread存在）
            2. 停止异步读取线程
            3. 释放VideoCapture资源
            4. 清空实例状态

        异常:
            DeviceNotConnectedError: 相机未连接

        资源清理：
            - 停止后台线程
            - 释放OpenCV VideoCapture句柄
            - 清空帧缓存

        注意：
            建议在不再需要相机时调用
            即使发生异常也会尽可能清理资源
        """
        # 检查连接状态
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        # 停止异步读取线程
        if self.thread is not None:
            self._stop_read_thread()

        # 释放VideoCapture资源
        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        logger.info(f"{self} disconnected.")
