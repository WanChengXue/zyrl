import gymnasium as gym
import tianshou as ts
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from functools import partial

from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo
from zyrl.utils.config_parser import ConfigParser
from zyrl.utils.file_utils import load_yaml, save_yaml


def stop_fn(reward_threshold: float, mean_reward: float) -> bool:
    if mean_reward >= reward_threshold:
        return True
    else:
        return False


def save_fn(policy, path: str):
    torch.save(policy.state_dict(), path)


class Pipeline:
    """统一的Pipeline类，根据配置自动选择LowLevel或HighLevel模式"""

    def __init__(self, env: str | gym.Env, config_path: str):
        """
        初始化Pipeline

        Args:
            env: 环境名称字符串或gym环境实例
            config_path: 配置文件路径
        """
        self._env_check(env)
        self._config = ConfigParser(config_path)

        # 根据配置确定运行模式
        self._mode = self._config.get("base.mode", "low_level")  # high_level, low_level
        self._create_pipeline()

    def _env_check(self, env: str | gym.Env):
        if isinstance(env, str):
            if env not in gym.envs.registry:
                raise ValueError(f"Environment {env} not found in gym.envs.registry")
            self._task_name = env
        else:
            # gymnasium no longer has check_env, we'll assume it's valid if it has required methods
            if not hasattr(env, "reset") or not hasattr(env, "step"):
                raise ValueError(
                    f"Environment {env} is not a valid gymnasium environment"
                )
            self._task_name = env.__qualname__
        self._env = env

    def _create_pipeline(self):
        if self._mode == "high_level":
            self._pipeline = HighLevelPipeline(self._env, self._task_name, self._config)
        else:
            self._pipeline = LowLevelPipeline(self._env, self._task_name, self._config)

    def fit(self):
        """训练模型"""
        self._pipeline.fit()

    def render(self):
        pass

    def test(self, num_episodes: int = 10):
        pass

    def save_checkpoint(self, path: str):
        pass

    def load_checkpoint(self, path: str):
        pass

    def resume_training(self, checkpoint_path: str, **kwargs):
        pass

    def get_training_history(self):
        pass

    def get_evaluation_history(self):
        pass


# 为了向后兼容，保留原有的类名
class HighLevelPipeline(Pipeline):
    """高级Pipeline，提供更简洁的接口（向后兼容）"""

    def __init__(self, env: str | gym.Env, task_name: str, config: ConfigParser):
        # 强制设置为high_level模式
        pass


class LowLevelPipeline(Pipeline):
    """底层Pipeline，提供更细粒度的控制（向后兼容）"""

    def __init__(self, env: str | gym.Env, task_name: str, config: ConfigParser):
        self._env = env
        self._task_name = task_name
        self._config = config
        self._load_default_config()
        self._init()
        save_yaml(self._config.get_all(), f"{self._logger_path}/config.yaml")

    def _init(self):
        self._create_logger()
        self._create_net()
        self._create_optimizer()
        self._create_policy()
        self._create_collector()
        self._create_trainer()

    def _create_logger(self):
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self._logger_path = os.path.join(
            self._config.get("base.logger_path", "/tmp/zyrl/logs"),
            self._task_name,
            time_str,
        )
        self._logger = ts.utils.TensorboardLogger(SummaryWriter(self._logger_path))

    def _create_net(self):
        if isinstance(self._env, str):
            env = gym.make(self._env)
        else:
            env = self._env()
        self._space_info = SpaceInfo.from_env(env)
        state_shape = self._space_info.observation_info.obs_shape
        action_shape = self._space_info.action_info.action_shape
        self._action_sapce = env.action_space
        net_type = self._config.get("net.type")
        if net_type == "mlp":
            hidden_sizes = self._config.get("net.hidden_sizes")
            self._net = Net(state_shape, action_shape, hidden_sizes=hidden_sizes)
        elif net_type == "rnn":
            pass

    def _create_optimizer(self):
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self._config.get("algo.lr")
        )

    def _load_default_config(self):
        algo_name = self._config.get("base.algo")
        common_default_config = ConfigParser(f"config/common_config.yaml")
        algo_default_config = ConfigParser(f"config/algo_config/{algo_name}.yaml")
        common_default_config.merge(algo_default_config)
        self._config.merge(common_default_config, overwrite=False)

    def _create_policy(self):
        if self._config.get("base.algo") == "dqn":
            self._policy = ts.policy.DQNPolicy(
                model=self._net,
                optim=self._optimizer,
                action_space=self._action_sapce,
                discount_factor=self._config.get("algo.discount_factor"),
                estimation_step=self._config.get("algo.estimation_step"),
                target_update_freq=self._config.get("algo.target_update_freq"),
            )
        elif self._config.get("base.algo") == "ppo":
            pass

    def _create_collector(self):
        buffer_size = self._config.get("trainer.buffer_size")
        train_num = self._config.get("trainer.train_num")
        if isinstance(self._env, str):
            train_envs = ts.env.RayVectorEnv(
                [lambda: gym.make(self._env) for _ in range(train_num)]
            )
        else:
            train_envs = ts.env.RayVectorEnv(
                [lambda: self._env() for _ in range(train_num)]
            )
        self._train_collector = ts.data.Collector(
            self._policy,
            train_envs,
            ts.data.VectorReplayBuffer(buffer_size, train_num),
            exploration_noise=True,
        )

        test_num = self._config.get("trainer.test_num")
        if isinstance(self._env, str):
            test_envs = ts.env.RayVectorEnv(
                [lambda: gym.make(self._env) for _ in range(test_num)]
            )
        else:
            test_envs = ts.env.RayVectorEnv(
                [lambda: self._env() for _ in range(test_num)]
            )

        self._test_collector = ts.data.Collector(
            self._policy, test_envs, exploration_noise=True
        )

    def _create_trainer(self):
        if self._config.get("base.reward_threshold") == "none":
            _stop_fn = None
        else:
            _stop_fn = partial(stop_fn, self._config.get("base.reward_threshold"))
        self._trainer = ts.trainer.OffpolicyTrainer(
            policy=self._policy,
            train_collector=self._train_collector,
            test_collector=self._test_collector,
            max_epoch=self._config.get("trainer.max_epoch"),
            step_per_epoch=self._config.get("trainer.step_per_epoch"),
            step_per_collect=self._config.get("trainer.step_per_collect"),
            episode_per_test=self._config.get("trainer.test_num"),
            batch_size=self._config.get("trainer.batch_size"),
            update_per_step=1 / self._config.get("trainer.step_per_collect"),
            train_fn=lambda epoch, env_step: self._policy.set_eps(
                self._config.get("trainer.eps_train")
            ),
            test_fn=lambda epoch, env_step: self._policy.set_eps(
                self._config.get("trainer.eps_test")
            ),
            stop_fn=_stop_fn,
            logger=self._logger,
        )

    def fit(self):
        self._trainer.run()
