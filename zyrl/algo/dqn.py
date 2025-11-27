import os
import torch
import tianshou as ts
import ray
from torch.utils.tensorboard import SummaryWriter
from zyrl.algo.base_trainer import BaseTrainer
from tianshou.utils.net.common import Net


class DQNTrainer(BaseTrainer):
    def __init__(self, *, algo_config: dict):
        super().__init__(algo_config=algo_config)
        self._algo_config = algo_config
        self._input_dim = algo_config.get("input_dim", 3)
        self._hidden_dim = algo_config.get("hidden_dim", 128)
        self._output_dim = algo_config.get("output_dim", 4)
        self._init_trainer_config()
        self._create_model_and_optimizer()

    def _init_trainer_config(self):
        self._checkpoint_folder = self._algo_config.get("checkpoint_folder", None)
        self._lr = self._algo_config.get("lr", 0.001)
        self._epoch = self._algo_config.get("epoch", 100)
        self._batch_size = self._algo_config.get("batch_size", 256)
        self._train_num = self._algo_config.get("train_num", 10)
        self._test_num = self._algo_config.get("test_num", 20)
        self._gamma = self._algo_config.get("gamma", 0.99)
        self._n_step = self._algo_config.get("n_step", 3)
        self._target_freq = self._algo_config.get("target_freq", 320)
        self._buffer_size = self._algo_config.get("buffer_size", 80000)
        self._eps_train = self._algo_config.get("eps_train", 0.05)
        self._eps_test = self._algo_config.get("eps_test", 0.0)
        self._step_per_epoch = self._algo_config.get("step_per_epoch", 4000)
        self._step_per_collect = self._algo_config.get(
            "step_per_collect", self._train_num
        )
        self._device = self._algo_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._logger = ts.utils.TensorboardLogger(
            SummaryWriter(f"{self._checkpoint_folder}/dqn")
        )

    def get_device(self):
        return self._device

    def _print_trainer_config(self):
        # 优雅输出训练配置信息
        print("\n" + "=" * 80)
        print("DQN Training Configuration")
        print("=" * 80)
        print(
            f"{'Device':<20} | {'Learning Rate':<15} | {'Batch Size':<12} | {'Epoch':<8} | {'Gamma':<8}"
        )
        print(
            f"{self._device:<20} | {self._lr:<15.6f} | {self._batch_size:<12} | {self._epoch:<8} | {self._gamma:<8.4f}"
        )
        print("-" * 80)
        print(
            f"{'Train Num':<20} | {'Test Num':<15} | {'Step/Epoch':<12} | {'N-Step':<8} | {'Buffer Size':<12}"
        )
        print(
            f"{self._train_num:<20} | {self._test_num:<15} | {self._step_per_epoch:<12} | {self._n_step:<8} | {self._buffer_size:<12}"
        )
        print("-" * 80)
        print(
            f"{'Eps Train':<20} | {'Eps Test':<15} | {'Target Freq':<12} | {'Input Dim':<8} | {'Hidden Dim':<12}"
        )
        print(
            f"{self._eps_train:<20.4f} | {self._eps_test:<15.4f} | {self._target_freq:<12} | {self._input_dim:<8} | {self._hidden_dim:<12}"
        )
        print("-" * 80)
        print("=" * 80 + "\n")

    def _create_model_and_optimizer(self):
        self._model = Net(
            state_shape=self._input_dim,
            action_shape=self._output_dim,
            hidden_sizes=[self._hidden_dim, self._hidden_dim],
            device=self._device,
        )
        self._model.to(device=self._device, dtype=torch.float32)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=100, gamma=0.1
        )

    def train(self, env_cls, env_config):
        self._print_trainer_config()
        self._checkpoint_path = os.path.join(self._checkpoint_folder, "model_best.pth")

        def stop_fn(mean_rewards: float) -> bool:
            if mean_rewards >= 2.2:
                return True
            return False

        env = env_cls(env_config)
        observation_space = env.observation_space
        action_space = env.action_space
        policy = ts.policy.DQNPolicy(
            model=self._model,
            optim=self._optimizer,
            discount_factor=self._gamma,
            action_space=action_space,
            estimation_step=self._n_step,
            target_update_freq=self._target_freq,
        )
        train_envs = ts.env.RayVectorEnv(
            [lambda: env_cls(env_config) for _ in range(self._train_num)]
        )
        test_envs = ts.env.RayVectorEnv(
            [lambda: env_cls(env_config) for _ in range(self._test_num)]
        )

        train_collector = ts.data.Collector(
            policy,
            train_envs,
            ts.data.VectorReplayBuffer(self._buffer_size, self._train_num),
            exploration_noise=True,
        )
        test_collector = ts.data.Collector(
            policy,
            test_envs,
            exploration_noise=True,
        )
        result = ts.trainer.OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=self._epoch,
            step_per_epoch=self._step_per_epoch,
            step_per_collect=self._step_per_collect,
            episode_per_test=self._test_num,
            batch_size=self._batch_size,
            update_per_step=1 / self._step_per_collect,
            train_fn=lambda epoch, env_step: policy.set_eps(self._eps_train),
            test_fn=lambda epoch, env_step: policy.set_eps(self._eps_test),
            stop_fn=stop_fn,
            logger=self._logger,
        ).run()
        print(f"Finished training in {result.timing.total_time} seconds")
        torch.save(self._model.state_dict(), self._checkpoint_path)
        ray.shutdown()

    def evaluate(self, data, label):
        pass

    def save_checkpoint(self):
        if not os.path.exists(self._checkpoint_folder):
            os.makedirs(self._checkpoint_folder)
        torch.save(self._model.state_dict(), f"{self._checkpoint_folder}/model.pth")

    def load_checkpoint(self, checkpoint_path: str):
        # 确保map_location能正确处理设备映射
        # 如果设备是CUDA但CUDA不可用，则映射到CPU
        if isinstance(self._device, str) and "cuda" in self._device:
            if torch.cuda.is_available():
                map_location = self._device
            else:
                map_location = torch.device("cpu")
        else:
            map_location = self._device
        model_dict = torch.load(checkpoint_path, map_location=map_location)
        self._model.load_state_dict(model_dict)

    def get_model(self):
        return self._model

    def get_checkpoint_path(self):
        return self._checkpoint_path
