from typing import Dict, Any, List
import numpy as np
from .interface import TianshouInterface


class TianshouCollector:
    """数据收集器，提供数据收集和评估功能"""

    def __init__(self, interface: TianshouInterface):
        self.interface = interface
        self._evaluation_results = []

    def collect_data(
        self, num_episodes: int = 100, render: bool = False
    ) -> Dict[str, Any]:
        """收集训练数据"""
        if self.interface._collector is None:
            raise ValueError("收集器未初始化，请先调用setup()")

        # 收集数据
        result = self.interface._collector.collect(n_episode=num_episodes)

        return {
            "episodes": num_episodes,
            "total_steps": result["n/st"],
            "total_episodes": result["n/ep"],
            "mean_reward": result["rew"],
            "mean_length": result["len"],
        }

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """评估策略性能"""
        if self.interface._policy is None:
            raise ValueError("策略未初始化，请先调用setup()")

        # 运行评估
        result = self.interface.test(num_episodes)

        evaluation_result = {
            "episodes": num_episodes,
            "mean_reward": result["rew"],
            "std_reward": result.get("rew_std", 0),
            "mean_length": result["len"],
            "std_length": result.get("len_std", 0),
            "success_rate": result.get("success_rate", 0),
        }

        self._evaluation_results.append(evaluation_result)
        return evaluation_result

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self._evaluation_results

    def benchmark(
        self, num_runs: int = 5, episodes_per_run: int = 10
    ) -> Dict[str, Any]:
        """基准测试，多次运行取平均值"""
        results = []

        for i in range(num_runs):
            result = self.evaluate(episodes_per_run)
            results.append(result)

        # 计算统计信息
        rewards = [r["mean_reward"] for r in results]
        lengths = [r["mean_length"] for r in results]

        benchmark_result = {
            "num_runs": num_runs,
            "episodes_per_run": episodes_per_run,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "individual_results": results,
        }

        return benchmark_result
