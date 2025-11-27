from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        algo_config: dict,
    ):
        super().__init__()

    @abstractmethod
    def train(self, data, label):
        pass

    @abstractmethod
    def evaluate(self, data, label):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_folder: str):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_folder: str):
        pass

    @abstractmethod
    def get_model(self):
        pass
