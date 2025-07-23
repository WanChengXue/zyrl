import torch
from typing import Dict, Any, Optional
from .interface import TianshouInterface


class TianshouTrainer:
    """高级训练器，提供更便捷的训练接口"""

    def __init__(self, interface: TianshouInterface):
        self.interface = interface
        self._training_history = []

    def train(self, **kwargs) -> Dict[str, Any]:
        """开始训练"""
        # 更新配置
        for key, value in kwargs.items():
            self.interface._config.update(f"trainer.{key}", value)

        # 开始训练
        result = self.interface.train()
        self._training_history.append(result)

        return result

    def resume_training(self, checkpoint_path: str, **kwargs):
        """从检查点恢复训练"""
        # 加载检查点
        self.interface._policy.load_state_dict(torch.load(checkpoint_path))

        # 继续训练
        return self.train(**kwargs)

    def get_training_history(self):
        """获取训练历史"""
        return self._training_history

    def save_checkpoint(self, path: str):
        """保存检查点"""
        if self.interface._policy is not None:
            torch.save(self.interface._policy.state_dict(), path)
        else:
            raise ValueError("没有可保存的策略")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        if self.interface._policy is not None:
            self.interface._policy.load_state_dict(torch.load(path))
        else:
            raise ValueError("策略未初始化")
