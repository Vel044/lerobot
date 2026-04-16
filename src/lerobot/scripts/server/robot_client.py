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
Example command:
```shell
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


class RobotClient:
    # 整体架构：两线程流水线
    #   主线程 control_loop()：① 每 tick 从 action_queue pop 一帧 action 发给机器人
    #                           ② 队列剩余不足时采观测发给 policy_server 触发下一次推理
    #   后台线程 receive_actions()：阻塞等待 policy_server 返回 action chunk，
    #                               写入 action_queue，同时做 temporal ensemble（多 chunk 融合）
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        # make_robot_from_config：根据 config.robot.type 字符串（如 "so101_follower"）
        # 实例化对应的 Robot 子类，connect() 打开串口并初始化电机
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        # lerobot_features: dict[str, FeatureType]
        # 键为机器人特征名（如 "observation.state", "action"），值描述张量形状和类型
        # 用于 policy_server 确认输入输出格式是否和模型匹配
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        if config.verify_robot_cameras:
            # Load policy config for validation
            policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
            policy_image_features = policy_config.image_features

            # 检查机器人挂载的相机名称和分辨率是否与模型训练时的配置一致
            # 不一致时直接报错，避免推理时维度不匹配
            validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        # RemotePolicyConfig 打包发给 policy_server 的初始化信息，包含：
        #   policy_type: str           —— 模型类型，如 "act"
        #   pretrained_name_or_path: str —— HuggingFace 模型路径或本地路径
        #   lerobot_features: dict     —— 机器人输入输出特征描述
        #   actions_per_chunk: int     —— policy_server 每次推理返回多少帧 action（即 chunk_size）
        #   device: str                —— policy_server 推理设备，如 "cuda", "mps", "cpu"
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        # grpc_channel_options 设置 gRPC 重连参数，initial_backoff 用 environment_dt（帧间隔）
        # 确保断线后重连速度和控制频率匹配
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        # stub: AsyncInferenceStub，由 protobuf 生成的 gRPC 客户端代理
        # 通过它调用 Ready / SendPolicyInstructions / SendObservations / GetActions 四个 RPC
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        # latest_action: int，当前已执行的最新帧号（timestep）
        # 初始为 -1，每次 control_loop_action() 执行完一帧后更新
        # receive_actions() 用它过滤掉比当前帧号旧的 action（时间戳已过期的帧直接丢弃）
        self.latest_action = -1
        # action_chunk_size: int，policy_server 每次推理返回的帧数
        # 首次收到 action chunk 后根据实际长度更新，-1 表示还没收到过任何 chunk
        # 用于 _ready_to_send_observation() 计算队列剩余比例
        self.action_chunk_size = -1

        # chunk_size_threshold: float，触发新一轮推理的队列剩余比例阈值
        # 判断条件：action_queue.qsize() / action_chunk_size <= threshold
        # 例：threshold=0.5, action_chunk_size=100 → 队列剩 50 帧时发观测给 server
        # 意图：让新 chunk 在旧 chunk 耗尽前到货，避免机器人停下来等推理
        self._chunk_size_threshold = config.chunk_size_threshold

        # action_queue: Queue[TimedAction]，主线程和后台线程共享的 action 缓冲
        # TimedAction 结构：
        #   timestamp: float   —— 该 action 被 server 生成时的 unix 时间戳（用于延迟计算）
        #   timestep: int      —— 该 action 对应的全局帧号（用于去重和 temporal ensemble）
        #   action: Tensor     —— shape=(action_dim,)，各关节目标位置，单位度
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # 保护 action_queue 的读写，主线程和后台线程都会访问
        # action_queue_size: list[int]，每帧执行前记录的队列长度，episode 结束后用于可视化分析
        self.action_queue_size = []
        # start_barrier: Barrier(2)，保证主线程 control_loop 和后台线程 receive_actions 同时启动
        # 两者都调用 wait() 后才会继续，避免一方先跑导致时序混乱
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # must_go: Event，协调"何时必须强制触发推理"的开关
        # 正常路径：队列剩余 ≤ threshold 时提前发观测（此时 must_go 不影响行为）
        # 兜底路径：推理赶不上执行，队列空了，must_go 被置位，
        #           下次采到的观测会带 must_go=True 发给 server，
        #           server 收到后立刻推理而不放入待处理队列，避免机器人长时间停住
        # 收到新 action chunk 后 must_go 重置为 set（为下一次空队列做准备）
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        """client 是否仍在运行。shutdown_event.set() 后变为 False，两个线程的 while 循环据此退出。"""
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # 第一步：握手，确认 policy_server 已就绪
            # stub.Ready() 发一个空包过去，server 回空包，RPC 本身是同步阻塞的
            # 如果 server 还没起来会抛 RpcError，由外层捕获
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # 第二步：把模型配置序列化发给 server，让它加载对应的 policy
            # pickle.dumps(self.policy_config) → bytes，包含模型路径、chunk_size、设备等
            # PolicySetup(data=bytes) 是 protobuf 消息体，server 收到后反序列化并加载模型
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """停止 client：通知两个线程退出循环 → 断开机器人 → 关闭 gRPC channel。"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise.

        Args:
            obs: TimedObservation —— 带时间戳的观测包，结构如下：
                timestamp: float     unix 时间，用于和 server 计算网络延迟
                timestep: int        当前帧号（= latest_action，表示"这帧观测是在执行完第 N 帧 action 后采的"）
                observation: dict    包含：
                    "observation.images.cam_high": np.ndarray (H, W, 3)  相机图像（uint8 BGR）
                    "observation.state": np.ndarray (action_dim,)        各关节当前位置（度）
                    "task": str      任务描述文本，送给 language-conditioned policy 用
                must_go: bool        True 时 server 立刻推理，False 时 server 可以排队等
        """
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        # pickle 序列化整个 TimedObservation 对象为 bytes
        # 观测包含图像（较大，约数十 KB），所以后面用 send_bytes_in_chunks 分块发送
        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            # send_bytes_in_chunks 把 observation_bytes 按 gRPC 最大包大小切成多个
            # services_pb2.Observation 消息体，返回一个生成器（iterator）
            # stub.SendObservations() 是 client-streaming RPC，接受一个消息流
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.info(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        """调试用：查看 action_queue 当前的大小和所有帧号。持锁快照，不修改队列。

        Returns:
            (int, list[int])  —— (队列长度, 排序后的帧号列表)
        """
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Temporal Ensemble：把新到的 action chunk 和队列里已有的 action 按帧号对齐融合。

        Args:
            incoming_actions: list[TimedAction]  policy_server 本次推理返回的 N 帧 action
                TimedAction 结构：
                    timestamp: float  server 生成该 action 时的 unix 时间（用于延迟统计）
                    timestep: int     全局帧号（和观测的 timestep 对应）
                    action: Tensor    shape=(action_dim,)，各关节目标位置（度）
            aggregate_fn: (Tensor, Tensor) -> Tensor
                x1 = 队列里已有的旧预测，x2 = 新 chunk 的预测
                默认取 x2（直接用新的覆盖旧的）
                典型替代：weighted_average，对同一时刻的多次预测加权平均，能减少抖动
        """
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        # 新建一个空队列收融合结果，最后原子替换 self.action_queue
        # 不直接修改旧队列是因为 Queue 不支持随机访问，只能重建
        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        # current_action_queue: dict[int, Tensor]
        # 键=帧号，值=当前队列里对应帧的 action tensor
        # 用于快速判断新来的帧号是否和老预测重叠
        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # 情况 1：新 action 的帧号 ≤ 已执行帧号 → 这帧已经过去了，直接丢弃
            if new_action.get_timestep() <= latest_action:
                continue

            # 情况 2：新 action 的帧号不在旧队列里 → 纯未来帧，直接入新队列
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # 情况 3：新旧帧号重叠 → temporal ensemble，用 aggregate_fn 融合两次预测
            # x1 = 旧队列里这帧的 action tensor（来自上一个 chunk 的预测）
            # x2 = 新 chunk 对同一帧的预测
            # 融合结果写入新 TimedAction，时间戳用新的（server 生成时间更新鲜）
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        # 原子替换旧队列，主线程下次取 action 时拿到的就是融合后的结果
        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """后台线程：持续从 policy_server 拉取推理结果，写入 action_queue。
        此方法运行在独立的 daemon 线程里，和主线程 control_loop() 并发执行。
        """
        # start_barrier.wait()：等主线程也 ready 后同时开始，避免后台线程先跑
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # stub.GetActions() 是同步阻塞 RPC：
                #   发一个空请求给 server，server 把最新推理好的 action chunk 打包返回
                #   如果 server 还没推理完，返回空包（data 为空），继续下一轮轮询
                # 返回值 actions_chunk: services_pb2.Actions，有一个 data: bytes 字段
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # pickle.loads → list[TimedAction]
                # 每个 TimedAction: {timestamp: float, timestep: int, action: Tensor(action_dim,)}
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # 更新 action_chunk_size：取历史最大值，防止偶发短包导致 threshold 计算失真
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # 把新 chunk 融合进 action_queue（temporal ensemble）
                # timed_actions: list[TimedAction]，过期帧丢弃，重叠帧加权平均，纯未来帧直接入队
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                # 收到新 chunk 后重置 must_go 为 set
                # 含义：如果队列再次空掉，下次采观测时会标 must_go=True，强制 server 立刻推理
                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        """把 policy 输出的 action tensor 转成机器人接受的字典格式。

        Args:
            action_tensor: Tensor(action_dim,)  各关节目标位置，按 robot.action_features 的顺序排列

        Returns:
            dict[str, float]  如 {"shoulder_pan": 45.0, "shoulder_lift": -30.0, ...}
        """
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """从 action_queue 取出一帧 action 发给机器人执行。每个控制 tick 最多调用一次。

        Returns:
            dict[str, float]  —— 键为关节名（如 "shoulder_pan"），值为目标位置（度）
                               和 robot.action_features 的键顺序一致
        """
        # get_nowait()：非阻塞取队首，如果队列空了抛 Empty 异常
        # 调用方 control_loop() 已用 actions_available() 先检查，所以这里正常不会抛
        # 持锁时间尽量短：只取元素，不做其他操作
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        # timed_action.get_action() → Tensor(action_dim,)，各关节目标位置
        # _action_tensor_to_action_dict() 把 tensor 展开成 {"joint_name": float} 字典
        # robot.send_action() 通过串口 sync_write 把目标位置发给 Feetech 舵机
        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        # 更新已执行帧号，receive_actions() 会用此值过滤早于当前的旧 action
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """判断当前是否应该触发下一次推理（采观测发给 server）。

        触发条件：action_queue.qsize() / action_chunk_size <= chunk_size_threshold
        例：qsize=45, chunk_size=100, threshold=0.5 → 0.45 ≤ 0.5 → True，触发
        设计意图：提前触发，让新 chunk 在旧 chunk 耗尽前到货
        注意：action_chunk_size 初始为 -1，首次 chunk 到货前除以 -1 结果为负数，
              满足 ≤ threshold，等价于"一开始就触发推理"，这是正确的启动行为
        """
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        """采一帧观测发给 policy_server 触发推理。由 control_loop() 在阈值条件满足时调用。

        Args:
            task: str  任务描述文本，如 "pick up the red block"，传给 language-conditioned policy
            verbose: bool  是否打印详细日志

        Returns:
            RawObservation: dict，包含相机图像和关节位置的原始观测数据（不含时间戳）
        """
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            # robot.get_observation() 采集一帧观测：
            #   1. 调用所有相机的 cam.async_read() 拿最新帧（后台线程持续读，这里只取缓存）
            #   2. 调用 bus.sync_read("Present_Position") 读取所有关节当前位置
            # 返回值：RawObservation = dict，键如：
            #   "observation.images.cam_high": np.ndarray (H, W, 3) uint8 BGR
            #   "observation.state": np.ndarray (action_dim,) float64 度
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            # TimedObservation：给 raw_observation 包一层时间信息
            #   timestamp: float  —— unix 时间，用于跨进程计算网络延迟
            #   observation: dict —— 上面采到的原始观测
            #   timestep: int     —— max(latest_action, 0)
            #       = 已执行完的帧号，告诉 server "这帧观测是在执行完第 N 帧 action 之后采的"
            #       首次调用时 latest_action=-1，max(-1,0)=0，从第 0 帧开始
            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # must_go 标志：仅当 must_go 事件已 set 且队列为空时才置 True
            # must_go.is_set() 在两种情况下为 True：
            #   1. 初始化时 set（首次触发）
            #   2. receive_actions() 收到新 chunk 后 set（为下一次空队列做准备）
            # action_queue.empty()：队列已空，机器人即将无动作可执行
            # 两者同时为 True → 这帧观测标记 must_go=True，server 端会立刻推理
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            # 发送观测给 server（非阻塞，发完就继续，不等推理结果）
            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must_go 已用于本次观测，清除标志，等下次收到 action chunk 后再重新 set
                # 防止后续每帧观测都带 must_go=True，给 server 造成不必要的压力
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """主控制循环：每 tick 交替执行 action 和触发推理。

        一个 tick 的流程：
          1. 如果 action_queue 不空 → pop 一帧 action 发给舵机
          2. 如果队列剩余 ≤ threshold → 采观测发给 policy_server 触发下一次推理
          3. sleep 补齐到 environment_dt（如 33ms → 30 FPS）

        两个条件是独立的 if，同一个 tick 里可以既执行 action 又发观测。
        两个条件也可以都不满足（比如队列满且不空），此时只做 sleep。

        Args:
            task: str  任务描述，透传给 control_loop_observation()
            verbose: bool  是否打印详细日志

        Returns:
            (Observation, Action) 最后一次的观测和动作（用于外部状态检查）
        """
        # start_barrier：等后台 receive_actions 线程也到这一步，然后同时开始
        # 保证两边同时启动，避免 control_loop 在 receive_actions 还没准备好时就发观测
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # environment_dt: float，目标帧间隔（秒），如 1/30 ≈ 0.0333
            # 如果本 tick 实际耗时 < environment_dt，sleep 补齐差值，保持稳定的控制频率
            # 如果耗时 > environment_dt（串口慢或推理卡），sleep 0，下一 tick 立刻开始
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    """入口函数：启动 RobotClient 的两线程流水线。

    线程布局：
      主线程      → control_loop()       每 tick 执行 action + 按需触发推理
      daemon 线程 → receive_actions()    持续从 server 拉取 action chunk

    启动顺序：
      1. 创建 RobotClient(cfg) → 连接机器人硬件、建立 gRPC channel
      2. client.start() → 和 policy_server 握手、发送模型配置
      3. 启动 receive_actions 后台线程
      4. 主线程进入 control_loop（两者在 start_barrier 处同步）
      5. Ctrl+C 或外部调用 stop() → shutdown_event.set() → 两个线程退出循环

    Args:
        cfg: RobotClientConfig  由 draccus 从命令行参数解析，包含：
            robot: RobotConfig              机器人类型、串口、相机配置
            server_address: str             policy_server 地址，如 "127.0.0.1:8080"
            policy_type: str                模型类型，如 "act"
            pretrained_name_or_path: str    模型路径
            actions_per_chunk: int          每次推理返回帧数（chunk_size）
            chunk_size_threshold: float     触发推理的队列剩余比例
            aggregate_fn_name: str          融合函数名，如 "weighted_average"
            environment_dt: float           目标帧间隔（秒）
            fps: float                      目标 FPS
            debug_visualize_queue_size: bool 是否在结束后画队列长度图
    """
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # daemon=True：主线程退出时后台线程自动终止，不需要手动 join 来保证退出
        # target=receive_actions：后台线程运行 receive_actions() 的 while 循环
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            # control_loop() 内部有 start_barrier.wait()，会等后台线程就绪后才真正开始
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            # join() 等后台线程处理完最后一个 action chunk 后退出
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                # visualize_action_queue_size：画一张 action_queue_size 随时间变化的图
                # 横轴=tick 数，纵轴=队列长度，可以直观看到 stall 和流水线效率
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
