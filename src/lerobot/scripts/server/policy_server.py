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
Example:
```shell
python src/lerobot/scripts/server/policy_server.py \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import SUPPORTED_POLICIES
from lerobot.scripts.server.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    # gRPC 服务端，实现 AsyncInferenceServicer 的四个 RPC：
    #   Ready()                  → 握手，client 连接时调用
    #   SendPolicyInstructions() → 接收模型配置并加载 policy
    #   SendObservations()       → 接收 client 发来的观测（client-streaming）
    #   GetActions()             → 推理并返回 action chunk（unary RPC）
    #
    # 核心数据流：
    #   client 采观测 → SendObservations → observation_queue(maxsize=1)
    #                                           ↓
    #   client 调 GetActions ← 取队首观测 → _predict_action_chunk() → 返回 list[TimedAction]
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        """初始化 server 状态。模型本身在 SendPolicyInstructions() 里才加载。

        Args:
            config: PolicyServerConfig，包含：
                host: str              监听地址，如 "127.0.0.1"
                port: int              监听端口，如 8080
                fps: float             目标 FPS（用于延迟统计）
                inference_latency: float  人为注入的最小推理延迟（秒），用于模拟真实场景
                obs_queue_timeout: float  GetActions 等待观测的超时（秒）
                environment_dt: float    帧间隔（秒），如 1/30 ≈ 0.0333
        """
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        # observation_queue: Queue[TimedObservation], maxsize=1
        # 只保留最新的一条观测，旧的被覆盖（因为推理速度 < 采样速度）
        # GetActions() 从这里 get()，SendObservations() 往这里 put()
        self.observation_queue = Queue(maxsize=1)

        # _predicted_timesteps: set[int]，已经推理过的帧号集合
        # 用于去重：如果 client 因网络抖动重发了同一帧观测，直接跳过
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        # last_processed_obs: TimedObservation | None
        # 上一次成功入队的观测，用于 observations_similar() 判断新观测是否和上次"太像"
        self.last_processed_obs = None

        # 以下属性在 SendPolicyInstructions() 收到 client 配置后才赋值
        self.device = None                # str，推理设备 "cpu" / "cuda" / "mps"
        self.policy_type = None           # str，模型类型 "act" / "pi0" / "diffusion" 等
        self.lerobot_features = None      # dict[str, PolicyFeature]，输入输出特征描述
        self.actions_per_chunk = None     # int，每次推理返回多少帧 action（= chunk_size）
        self.policy = None                # Policy 子类实例，如 ACTPolicy

    @property
    def running(self):
        """server 是否在运行。shutdown_event.set() 后变为 False。"""
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        """返回模型期望的图像特征描述，如 {"observation.images.cam_high": PolicyFeature(shape=(3,480,640))}。"""
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """新 client 连接时重置 server 状态：清空观测队列和已推理帧号集合。"""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        """RPC 1/4：握手。client 连上后第一个调用。

        Args:
            request: services_pb2.Empty  空请求
            context: grpc.ServicerContext  包含 context.peer() = "ipv4:127.0.0.1:xxxxx"

        Returns:
            services_pb2.Empty
        """
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """RPC 2/4：接收 client 发来的模型配置，加载 policy 并放到指定设备。

        Args:
            request: services_pb2.PolicySetup
                request.data: bytes = pickle.dumps(RemotePolicyConfig)
                RemotePolicyConfig 结构：
                    policy_type: str           如 "act"
                    pretrained_name_or_path: str  模型路径
                    lerobot_features: dict[str, PolicyFeature]  输入输出特征描述
                    actions_per_chunk: int     chunk_size
                    device: str                如 "cpu"
            context: grpc.ServicerContext

        Returns:
            services_pb2.Empty
        """

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        # request.data 是 client 端 pickle.dumps(RemotePolicyConfig) 的结果
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # get_policy_class: 根据字符串（如 "act"）返回对应的 Policy 子类（如 ACTPolicy）
        policy_class = get_policy_class(self.policy_type)

        # from_pretrained: 从 HuggingFace hub 或本地路径加载模型权重和配置
        # policy.to(device): 把模型参数移到指定设备
        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """RPC 3/4：接收 client 发来的观测。client-streaming RPC（client 分块发，server 拼接）。

        收到观测后做三件事：
          1. 拼接分块字节 → pickle 反序列化为 TimedObservation
          2. 计算网络延迟和 FPS
          3. 调用 _enqueue_observation() 放入 observation_queue（可能被过滤掉）

        Args:
            request_iterator: Iterator[services_pb2.Observation]
                client 端 send_bytes_in_chunks 产出的消息流，每个消息的 data 字段是字节片段
            context: grpc.ServicerContext

        Returns:
            services_pb2.Empty
        """
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        # receive_bytes_in_chunks: 从 request_iterator 里逐个取 protobuf 消息，
        # 把 data 字段拼接成完整 bytes。阻塞式调用，直到 client 发完所有分块。
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        # 反序列化为 TimedObservation：
        #   timestamp: float   client 采观测时的 unix 时间
        #   timestep: int      当前帧号
        #   observation: RawObservation  = dict[str, Tensor]  包含图像和关节位置
        #   must_go: bool       是否强制推理
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            # 观测被过滤：可能帧号重复、或和上次观测太相似、或 must_go=False 且队列满
            self.logger.info(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """RPC 4/4：client 轮询调用，返回推理结果（action chunk）。Unary RPC。

        流程：
          1. 从 observation_queue 阻塞等待一条观测（超时则返回空包）
          2. 调用 _predict_action_chunk() 推理 → list[TimedAction]
          3. pickle 序列化后返回给 client
          4. 按 inference_latency 参数 sleep，控制最小推理延迟

        Args:
            request: services_pb2.Empty  client 每次发空请求来拉取结果
            context: grpc.ServicerContext

        Returns:
            services_pb2.Actions  data 字段 = pickle.dumps(list[TimedAction])
            或 services_pb2.Empty  超时或出错时返回空包
        """
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            # observation_queue.get(): 阻塞等待，超时 obs_queue_timeout 秒后抛 Empty
            # 因为 maxsize=1，队列里最多只有一条观测
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            # 记录已推理帧号，避免同一帧被重复推理
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            # _predict_action_chunk: 预处理 → 推理 → 后处理，返回 list[TimedAction]
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            # pickle 序列化 list[TimedAction] 为 bytes，通过网络发给 client
            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            # inference_latency 人为注入延迟，模拟真实推理耗时（调试用）
            # 如果实际推理已经超过了 inference_latency，sleep 0
            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            # 超时没收到观测 → 返回空包，client 端 continue 轮询
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """检查观测是否值得推理。两个过滤条件：
        1. 帧号已推理过 → 跳过（去重）
        2. 和上次推理的观测太相似（关节位置差 < atol） → 跳过（减少无效推理）

        Args:
            obs: TimedObservation  本次收到的观测
            previous_obs: TimedObservation  上一次成功入队的观测

        Returns:
            bool  True 表示可以入队推理，False 表示应过滤掉
        """
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        # 条件 1：帧号重复（网络重传或 client 逻辑问题）
        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        # 条件 2：关节位置几乎没变化 → 推理结果会和上次差不多，浪费算力
        # observations_similar: 把两个观测的 "observation.state" tensor 做 L2 范数比较
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """尝试把观测放入 observation_queue。不满足条件时跳过。

        入队条件（满足任一即可）：
          1. obs.must_go = True（client 要求强制推理）
          2. last_processed_obs is None（首条观测）
          3. _obs_sanity_checks 通过（帧号不重复 + 观测有变化）

        因为 observation_queue maxsize=1，如果队列已满，先 pop 旧的再 put 新的。
        保证队列里永远是"最新"的观测。

        Args:
            obs: TimedObservation  待入队的观测

        Returns:
            bool  True 入队成功，False 被过滤掉
        """

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """把 policy 输出的 tensor 列表包装成带时间戳的 TimedAction 列表。

        Args:
            t_0: float  观测的 unix 时间戳，作为第一个 action 的基准时间
            action_chunk: list[Tensor]  每个 Tensor shape=(action_dim,)，各关节目标位置
            i_0: int  观测的帧号，action 的 timestep 从 i_0 开始递增

        Returns:
            list[TimedAction]，每个元素：
                timestamp: t_0 + i * environment_dt（按帧间隔递增）
                timestep: i_0 + i
                action: Tensor(action_dim,)
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """把 RawObservation 转换为 policy 推理需要的格式。

        转换内容：
          1. 键名映射：motor 原始键 → lerobot 标准键（如 "observation.state"）
          2. 图像缩放：原始分辨率 → policy 训练时的分辨率（如 480×640）
          3. 数据类型：int8 [0,255] 图像 → float32 [0,1]
          4. shape 调整：(H,W,C) → (1,C,H,W) 加 batch 维度
          5. 移到 device

        Args:
            observation_t: TimedObservation
                .observation = RawObservation = dict[str, Tensor/ndarray]
                    键如 "observation.images.cam_high": (H, W, 3) uint8
                         "observation.state": (action_dim,) float64

        Returns:
            Observation = dict[str, Tensor]，所有值在 device 上，图像已加 batch 维度
                键如 "observation.images.cam_high": (1, C, H, W) float32
                     "observation.state": (1, action_dim) float32
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """调用 policy 推理，返回 action chunk tensor。

        Args:
            observation: Observation = dict[str, Tensor]
                policy 的输入，键和 _prepare_observation 返回的一致

        Returns:
            Tensor, shape=(B, actions_per_chunk, action_dim)
                B=1（单 batch）
                actions_per_chunk ≤ chunk_size（截取前 N 帧）
                action_dim = 关节数
        """
        # policy.predict_action_chunk: 内部走 ACT 的 encoder-decoder 推理
        # 返回 shape=(B, chunk_size, action_dim) 或 (chunk_size, action_dim)
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        # 截取前 actions_per_chunk 帧，丢弃多余的
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """完整的推理流水线：预处理 → 推理 → 后处理。由 GetActions() 调用。

        Args:
            observation_t: TimedObservation  带时间戳的原始观测

        Returns:
            list[TimedAction]  推理出的 action chunk，每个带 timestamp 和 timestep
        """
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        preprocessing_time = time.perf_counter() - start_time

        # 更新 last_processed_obs，后续观测会跟它做 similarity 检查
        self.last_processed_obs: TimedObservation = observation_t

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # action_tensor: (1, actions_per_chunk, action_dim) → squeeze(0) → (actions_per_chunk, action_dim)
        # 移到 CPU 再序列化（pickle 不支持 CUDA tensor）
        action_tensor = action_tensor.cpu().squeeze(0)

        # 把 tensor 列表包装成 TimedAction 列表：
        #   timestamp = 观测时间戳 + i * 帧间隔
        #   timestep = 观测帧号 + i
        #   action = Tensor(action_dim,)
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms | "
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms | "
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms | "
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """停止 server：重置所有状态（清空队列和已推理帧号集合）。"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """policy_server 入口：创建 gRPC server 并阻塞等待终止。

    启动流程：
      1. 创建 PolicyServer 实例（此时不加载模型）
      2. 创建 gRPC server，线程池 max_workers=4（最多同时处理 4 个 RPC）
      3. 注册 AsyncInferenceServicer（Ready/SendPolicyInstructions/SendObservations/GetActions）
      4. 监听 host:port，等待 client 连接

    典型调用：
      python policy_server.py --host=127.0.0.1 --port=8080 --fps=30

    Args:
        cfg: PolicyServerConfig，由 draccus 从命令行参数解析，包含：
            host: str              监听地址
            port: int              监听端口
            fps: float             目标 FPS
            inference_latency: float  最小推理延迟（秒）
            obs_queue_timeout: float  观测队列等待超时（秒）
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # grpc.server: 用线程池处理并发 RPC 请求
    # max_workers=4：同时最多 4 个线程处理 client 请求
    # 实际场景中只有一个 client，但 SendObservations 是 client-streaming
    # 可能和 GetActions 并发，所以需要多线程
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    # 把 PolicyServer 注册为 AsyncInferenceServicer 的实现
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    # 阻塞等待，直到 server.stop() 被调用或进程被 kill
    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
