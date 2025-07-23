import gymnasium as gym
import torch
import torch.nn as nn
from typing import Union, Dict, Any, Optional
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np

from ..utils.config_parser import ConfigParser


class TianshouInterface:
    """Tianshou中间层接口，负责连接配置和底层训练"""

    def __init__(self, env: Union[str, gym.Env], config_path: str):
        """
        初始化Tianshou接口

        Args:
            env: 环境名称字符串或gym环境实例
            config_path: 配置文件路径
        """
        self._config = ConfigParser(config_path)
        self._env_check(env)
        self._create_logger()
        self._env = None
        self._policy = None
        self._trainer = None
        self._collector = None

    def _env_check(self, env: Union[str, gym.Env]):
        """检查环境有效性"""
        if isinstance(env, str):
            if env not in gym.envs.registry:
                raise ValueError(f"环境 {env} 在gym.envs.registry中未找到")
            self._task_name = env
        else:
            if not gym.check_env(env):
                raise ValueError(f"环境 {env} 不是有效的gymnasium环境")
            self._task_name = env.__class__.__name__
        self._env_spec = env

    def _create_logger(self):
        """创建日志记录器"""
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        logger_path = os.path.join(
            self._config.get("logger.path", "/tmp/zyrl/logs"), self._task_name, time_str
        )
        os.makedirs(logger_path, exist_ok=True)
        self._logger = TensorboardLogger(SummaryWriter(logger_path))

    def create_env(self, num_envs: Optional[int] = None) -> gym.Env:
        """创建环境"""
        if num_envs is None:
            num_envs = self._config.get("env.num_envs", 1)

        if isinstance(self._env_spec, str):
            env_fn = lambda: gym.make(self._env_spec)
        else:
            env_fn = lambda: self._env_spec

        if num_envs == 1:
            self._env = env_fn()
        else:
            # 使用向量化环境
            env_type = self._config.get("env.vector_type", "dummy")
            if env_type == "subproc":
                self._env = SubprocVectorEnv([env_fn for _ in range(num_envs)])
            else:
                self._env = DummyVectorEnv([env_fn for _ in range(num_envs)])

        return self._env

    def create_network(self) -> nn.Module:
        """创建神经网络"""
        network_config = self._config.get("network", {})
        network_type = network_config.get("type", "mlp")

        # 从环境中获取维度信息
        if self._env is None:
            self.create_env()

        if hasattr(self._env, "observation_space"):
            obs_space = self._env.observation_space
            if hasattr(obs_space, "shape"):
                input_dim = (
                    obs_space.shape[0]
                    if len(obs_space.shape) == 1
                    else np.prod(obs_space.shape)
                )
            else:
                input_dim = network_config.get("input_dim", 4)
        else:
            input_dim = network_config.get("input_dim", 4)

        if hasattr(self._env, "action_space"):
            action_space = self._env.action_space
            if hasattr(action_space, "n"):
                output_dim = action_space.n
            else:
                output_dim = network_config.get("output_dim", 2)
        else:
            output_dim = network_config.get("output_dim", 2)

        # 更新配置
        network_config["input_dim"] = input_dim
        network_config["output_dim"] = output_dim

        if network_type == "mlp":
            return self._create_mlp_network(network_config)
        elif network_type == "cnn":
            return self._create_cnn_network(network_config)
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")

    def _create_mlp_network(self, config: Dict[str, Any]) -> nn.Module:
        """创建MLP网络"""
        input_dim = config.get("input_dim")
        output_dim = config.get("output_dim")
        hidden_sizes = config.get("hidden_sizes", [256, 256])
        activation = config.get("activation", "relu")

        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            activation_fn = nn.ReLU

        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, hidden_size), activation_fn()])
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _create_cnn_network(self, config: Dict[str, Any]) -> nn.Module:
        """创建CNN网络"""
        # 这里可以根据需要实现CNN网络
        raise NotImplementedError("CNN网络暂未实现")

    def create_policy(self, network: nn.Module) -> BasePolicy:
        """创建策略"""
        policy_config = self._config.get("policy", {})
        policy_type = policy_config.get("type", "dqn")

        if policy_type == "dqn":
            return self._create_dqn_policy(network, policy_config)
        elif policy_type == "ppo":
            return self._create_ppo_policy(network, policy_config)
        else:
            raise ValueError(f"不支持的策略类型: {policy_type}")

    def _create_dqn_policy(
        self, network: nn.Module, config: Dict[str, Any]
    ) -> BasePolicy:
        """创建DQN策略"""
        from tianshou.policy import DQNPolicy

        return DQNPolicy(
            model=network,
            optim=torch.optim.Adam(network.parameters(), lr=config.get("lr", 1e-3)),
            discount_factor=config.get("gamma", 0.99),
            estimation_step=config.get("n_step", 1),
            target_update_freq=config.get("target_update_freq", 500),
        )

    def _create_ppo_policy(
        self, network: nn.Module, config: Dict[str, Any]
    ) -> BasePolicy:
        """创建PPO策略"""
        from tianshou.policy import PPOPolicy

        return PPOPolicy(
            actor=network,
            critic=network,  # 简化处理，实际应该分开
            optim=torch.optim.Adam(network.parameters(), lr=config.get("lr", 3e-4)),
            dist_fn=None,  # 需要根据环境类型设置
            discount_factor=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            vf_coef=config.get("vf_coef", 0.5),
            ent_coef=config.get("ent_coef", 0.01),
            action_space=self._env.action_space,
        )

    def create_collector(self, policy: BasePolicy) -> Collector:
        """创建数据收集器"""
        buffer_config = self._config.get("buffer", {})
        buffer_size = buffer_config.get("size", 100000)

        buffer = VectorReplayBuffer(buffer_size, len(self._env))

        self._collector = Collector(policy, self._env, buffer, exploration_noise=True)

        return self._collector

    def create_trainer(self, policy: BasePolicy, collector: Collector) -> BaseTrainer:
        """创建训练器"""
        trainer_config = self._config.get("trainer", {})
        trainer_type = trainer_config.get("type", "offpolicy")

        if trainer_type == "offpolicy":
            return self._create_offpolicy_trainer(policy, collector, trainer_config)
        elif trainer_type == "onpolicy":
            return self._create_onpolicy_trainer(policy, collector, trainer_config)
        else:
            raise ValueError(f"不支持的训练器类型: {trainer_type}")

    def _create_offpolicy_trainer(
        self, policy: BasePolicy, collector: Collector, config: Dict[str, Any]
    ):
        """创建离策略训练器"""
        from tianshou.trainer import OffpolicyTrainer

        return OffpolicyTrainer(
            policy=policy,
            collector=collector,
            test_collector=None,  # 可以添加测试收集器
            max_epoch=config.get("max_epoch", 100),
            step_per_epoch=config.get("step_per_epoch", 1000),
            step_per_collect=config.get("step_per_collect", 10),
            update_per_step=config.get("update_per_step", 0.1),
            batch_size=config.get("batch_size", 64),
            train_fn=None,
            test_fn=None,
            stop_fn=None,
            save_best_fn=None,
            logger=self._logger,
        )

    def _create_onpolicy_trainer(
        self, policy: BasePolicy, collector: Collector, config: Dict[str, Any]
    ):
        """创建在策略训练器"""
        from tianshou.trainer import OnpolicyTrainer

        return OnpolicyTrainer(
            policy=policy,
            collector=collector,
            test_collector=None,
            max_epoch=config.get("max_epoch", 100),
            step_per_epoch=config.get("step_per_epoch", 1000),
            repeat_per_collect=config.get("repeat_per_collect", 2),
            episode_per_test=config.get("episode_per_test", 100),
            batch_size=config.get("batch_size", 64),
            step_per_collect=config.get("step_per_collect", 1000),
            train_fn=None,
            test_fn=None,
            stop_fn=None,
            save_best_fn=None,
            logger=self._logger,
        )

    def setup(self):
        """设置完整的训练流程"""
        # 创建环境
        self.create_env()

        # 创建网络
        network = self.create_network()

        # 创建策略
        policy = self.create_policy(network)

        # 创建收集器
        collector = self.create_collector(policy)

        # 创建训练器
        trainer = self.create_trainer(policy, collector)

        self._policy = policy
        self._trainer = trainer
        self._collector = collector

        return trainer

    def train(self):
        """开始训练"""
        if self._trainer is None:
            self.setup()

        result = self._trainer.run()
        return result

    def test(self, num_episodes: int = 10):
        """测试策略"""
        if self._policy is None:
            raise ValueError("请先调用setup()或train()方法")

        # 创建测试环境
        test_env = self.create_env(num_envs=1)

        # 创建测试收集器
        test_collector = Collector(self._policy, test_env)

        # 运行测试
        result = test_collector.collect(n_episode=num_episodes)

        return result
