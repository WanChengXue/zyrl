from zyrl.algo.dqn import DQNTrainer


class RLTrainer:
    def __init__(self, *, rl_config: dict):
        self._rl_config = rl_config
        self._rl_algo = self._rl_config.get("algo", "DQN")
        self._create_trainer()

    def _create_trainer(self):
        if self._rl_algo == "DQN":
            self._trainer = DQNTrainer(algo_config=self._rl_config)

    def load_checkpoint(self, checkpoint_folder: str):
        self._trainer.load_checkpoint(checkpoint_folder)

    def save_checkpoint(self):
        self._trainer.save_checkpoint()

    def get_checkpoint_path(self):
        return self._trainer.get_checkpoint_path()

    def get_model(self):
        return self._trainer.get_model()

    def train(self, env_cls, env_config):
        self._trainer.train(env_cls, env_config)

    def evaluate(self, env_cls, env_config):
        self._trainer.evaluate(env_cls, env_config)

    def get_device(self):
        return self._trainer.get_device()
