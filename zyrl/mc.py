"""Monte Carlo reinforcement learning implementation with Ray distributed computing.

This module provides a Monte Carlo method implementation for reinforcement learning
that uses Ray for distributed computation to improve performance.
"""

import copy
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ray
from collections import deque


class SlidingWindowQCalculator:
    """基于滑动窗口的Q值计算器

    维护一个固定长度的窗口，当窗口满时自动丢弃最老的值，
    始终计算窗口内所有值的平均值作为Q值。
    """

    def __init__(self, window_size: int = 30000):
        """
        初始化滑动窗口Q值计算器

        Args:
            window_size: 窗口大小，默认30000
        """
        self.window_size = window_size
        self.q_windows = {}  # {(state_index, action): deque}
        self.q_values = {}  # {(state_index, action): float}

    def add_sample(self, state_index: int, action: int, q_values):
        """
        添加新的Q值样本到滑动窗口

        Args:
            state_index: 状态索引
            action: 动作
            q_values: Q值或Q值列表
        """
        key = (state_index, action)

        # 如果该状态-动作对还没有窗口，创建一个新的deque
        if key not in self.q_windows:
            self.q_windows[key] = deque(maxlen=self.window_size)
            self.q_values[key] = 0.0

        # 确保q_values是列表
        if isinstance(q_values, (int, float)):
            q_values = [q_values]
        elif not isinstance(q_values, (list, np.ndarray)):
            q_values = list(q_values)

        # 添加所有新值到窗口
        for q_value in q_values:
            self.q_windows[key].append(q_value)

        # 计算窗口内所有值的平均值
        self.q_values[key] = np.mean(self.q_windows[key])

    def get_q_value(self, state_index: int, action: int) -> float:
        """
        获取指定状态-动作对的Q值

        Args:
            state_index: 状态索引
            action: 动作

        Returns:
            该状态-动作对的Q值，如果不存在则返回0.0
        """
        key = (state_index, action)
        return self.q_values.get(key, 0.0)

    def get_all_q_values(self) -> dict:
        """
        获取所有状态-动作对的Q值

        Returns:
            包含所有Q值的字典
        """
        return self.q_values.copy()

    def get_window_size(self, state_index: int, action: int) -> int:
        """
        获取指定状态-动作对的当前窗口大小

        Args:
            state_index: 状态索引
            action: 动作

        Returns:
            当前窗口大小
        """
        key = (state_index, action)
        return len(self.q_windows.get(key, []))

    def clear(self):
        """清空所有窗口数据"""
        self.q_windows.clear()
        self.q_values.clear()


class MCFromStart:
    """Monte Carlo reinforcement learning agent that learns from complete episodes.

    This class implements Monte Carlo control with exploring starts, using distributed
    computing via Ray to collect experience data and update Q-values.
    """

    def __init__(
        self,
        env_class,
        init_q_table: pd.DataFrame,
        env_config: dict,
        mc_config: dict,
        debug_mode: bool = False,
    ):
        self._debug_mode = debug_mode
        self._saved_q_table_path = mc_config.get("saved_q_table_path")
        self._saved_count_table_path = mc_config.get("saved_count_table_path")
        self._mask_table = mc_config.get("mask_table", None)
        self._mc_config = mc_config
        self._env_config = env_config
        self._gamma = mc_config.get("gamma", 0.99)
        self._q_table = init_q_table
        self._env_class = env_class
        self._loss_list = []
        # 新增收敛控制参数
        self._learning_rate = mc_config.get("learning_rate", 0.1)
        self._min_learning_rate = mc_config.get("min_learning_rate", 0.01)
        self._learning_rate_decay = mc_config.get("learning_rate_decay", 0.995)
        self._convergence_window = mc_config.get("convergence_window", 10)
        self._use_max_error = mc_config.get("use_max_error", False)
        self._patience = mc_config.get("patience", 50)
        self._no_improvement_count = 0
        self._best_mse = float("inf")
        self._loss_history = []
        self._sample_env_num = mc_config.get("sample_env_num", 10)
        # 滑动窗口Q值计算器
        self._use_sliding_window = mc_config.get("use_sliding_window", False)
        self._window_size = mc_config.get("window_size", 20000)
        if self._use_sliding_window:
            self._sliding_window_calculator = SlidingWindowQCalculator(
                self._window_size
            )
            print(f"启用滑动窗口模式，窗口大小: {self._window_size}")
        else:
            self._sliding_window_calculator = None

        self._init_count_table()
        self._saved_q_table()

    def _init_count_table(self):
        self._sample_point_count_table = pd.DataFrame(
            index=self._q_table.index, columns=self._q_table.columns
        )
        self._sample_point_count_table.fillna(0, inplace=True)

    def run(self):
        end_flag = False
        iteration = 0
        while not end_flag:
            iteration += 1
            print(f"\n=== 开始第 {iteration} 次迭代 ===")
            before_update_q_table = copy.deepcopy(self._q_table)
            # 清理历史采样数据，确保每次迭代都基于最新的Q表格
            self._clear_historical_data()
            # 使用最新Q表格进行采样
            self._collect_data()
            # 检查收敛
            end_flag = self._end_check(before_update_q_table, self._q_table)
            if iteration >= 15:
                end_flag = True
            # 保存结果
            self._saved_q_table()
            plt.plot(self._loss_list)
            plt.savefig(self._mc_config.get("loss_plot_path"))
            plt.close()
            print(f"第 {iteration} 次迭代完成，loss: {self._loss_list[-1]:.6f}")
        print(f"训练完成，总共进行了 {iteration} 次迭代")

    def _clear_historical_data(self):
        """
        清理历史采样数据，确保每次迭代都基于最新的Q表格
        因为Q表格更新后，基于旧Q表格的历史采样数据已经失效

        注意：计数表不需要清除，因为它记录的是总采样次数，这个信息是有用的
        """
        print("准备使用最新Q表格进行采样（保留计数表信息）")

    def _saved_q_table(self):
        self._q_table.to_csv(self._saved_q_table_path)
        self._sample_point_count_table.to_csv(self._saved_count_table_path)

    def _end_check(self, current_state: pd.DataFrame, next_state: pd.DataFrame) -> bool:
        mse_error = np.mean((current_state.values - next_state.values) ** 2)
        max_error = np.max(np.abs(current_state.values - next_state.values))
        current_max_action = np.argmax(current_state.values, 1)
        next_max_action = np.argmax(next_state.values, 1)
        action_diff = sum(next_max_action != current_max_action)
        if self._use_max_error:
            error_val = max_error
        else:
            error_val = mse_error
        self._loss_list.append(error_val)
        self._loss_history.append(error_val)

        # 更新学习率
        if error_val < self._best_mse:
            self._best_mse = error_val
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        # 自适应学习率衰减
        if self._no_improvement_count > 10:
            self._learning_rate = max(
                self._min_learning_rate, self._learning_rate * self._learning_rate_decay
            )

        print(
            f"mse_error: {mse_error:.6f}, max_error: {max_error:.6f}, \n"
            f"action_diff: {action_diff}, learning_rate: {self._learning_rate:.4f}, \n"
            f"current_max_q_table_value: {np.max(current_state.values)}, current_min_q_table_value: {np.min(current_state.values)}, \n"
            f"next_max_q_table_value: {np.max(next_state.values)}, next_min_q_table_value: {np.min(next_state.values)}, \n"
            f"no_improvement: {self._no_improvement_count}"
        )

        # 如果使用滑动窗口，显示窗口统计信息
        if self._use_sliding_window:
            # 获取滑动窗口统计信息
            window_stats = self._get_sliding_window_stats()
            print(
                f"滑动窗口统计: 总窗口数={window_stats['total_windows']}, "
                f"平均窗口大小={window_stats['avg_window_size']:.1f}, "
                f"最大窗口大小={window_stats['max_window_size']}, "
                f"最小窗口大小={window_stats['min_window_size']}"
            )

        # 改进的收敛判断
        if mse_error < 1e-4:
            return True

        # if self._no_improvement_count >= self._patience:
        #     print(f"Early stopping: no improvement for {self._patience} iterations")
        #     return True
        # if len(self._loss_history) >= self._convergence_window:
        #     recent_mse = np.mean(self._loss_history[-self._convergence_window :])
        #     if recent_mse < 2e-1:
        #         return True

        return False

    def _update_q_table(self, return_dict: dict[int, list[float]]):
        """
        使用当前迭代的采样结果更新Q值
        支持滑动窗口模式和传统模式
        """
        for state_index in self._sample_point_count_table.index:
            for action_ind, action in enumerate(self._q_table.columns):
                if (state_index, action) not in return_dict:
                    continue

                # 计算当前迭代的采样平均值
                new_sample_mean = np.mean(return_dict[state_index, action])
                current_q_value = self._q_table.iloc[state_index, action_ind]

                if (
                    self._use_sliding_window
                    and self._sliding_window_calculator is not None
                ):
                    # 滑动窗口模式：将新样本列表添加到滑动窗口
                    sample_list = return_dict[state_index, action]
                    self._sliding_window_calculator.add_sample(
                        state_index, action, sample_list
                    )
                    # 从滑动窗口获取更新后的Q值
                    updated_q_value = self._sliding_window_calculator.get_q_value(
                        state_index, action
                    )
                else:
                    # 传统模式：使用学习率进行平滑更新
                    updated_q_value = (
                        1 - self._learning_rate
                    ) * current_q_value + self._learning_rate * new_sample_mean

                # 更新Q值
                self._q_table.iloc[state_index, action_ind] = updated_q_value

                # 累积计数表信息（保留总采样次数统计）
                current_count = self._sample_point_count_table.iloc[
                    state_index, action_ind
                ]
                new_samples = len(return_dict[state_index, action])
                self._sample_point_count_table.iloc[state_index, action_ind] = (
                    current_count + new_samples
                )

    def _get_sliding_window_stats(self):
        """
        获取滑动窗口统计信息

        Returns:
            包含窗口统计信息的字典
        """
        if not self._use_sliding_window or self._sliding_window_calculator is None:
            return {}

        stats = {
            "total_windows": len(self._sliding_window_calculator.q_windows),
            "window_sizes": {},
            "avg_window_size": 0,
        }

        if stats["total_windows"] > 0:
            window_sizes = []
            for (
                state_index,
                action,
            ), window in self._sliding_window_calculator.q_windows.items():
                size = len(window)
                window_sizes.append(size)
                stats["window_sizes"][(state_index, action)] = size

            stats["avg_window_size"] = np.mean(window_sizes)
            stats["max_window_size"] = max(window_sizes)
            stats["min_window_size"] = min(window_sizes)

        return stats

    def _collect_data(self):
        """
        使用当前最新的Q表格进行采样，确保每次迭代都基于最新的策略
        """
        return_dict = {}
        # 动态调整采样数量：如果MSE较大，增加采样数量
        base_sample_num = self._mc_config.get("sample_env_num", 10)
        if len(self._loss_history) > 0:
            recent_mse = (
                np.mean(self._loss_history[-5:])
                if len(self._loss_history) >= 5
                else self._loss_history[-1]
            )
            # 根据MSE调整采样数量
            if recent_mse > 0.1:
                self._sample_env_num = min(base_sample_num * 3, 1000)
            elif recent_mse > 0.01:
                self._sample_env_num = min(base_sample_num * 2, 500)
            else:
                self._sample_env_num = base_sample_num
        else:
            self._sample_env_num = base_sample_num

        print(
            f"Using {self._sample_env_num} samples per state-action pair with current Q-table"
        )

        # 确保使用当前最新的Q表格进行采样
        current_q_table = self._q_table.copy()
        self._debug_return_list = []
        for state_index in tqdm(self._q_table.index):
            for init_action in self._q_table.columns:
                if self._mask_table is not None:
                    if not self._mask_table.iloc[state_index, init_action]:
                        continue
                worker_list = []
                for _ in range(self._sample_env_num):
                    worker_config = {
                        "init_state_index": state_index,
                        "init_action": init_action,
                        "q_table": current_q_table,  # 使用当前最新的Q表格
                        "gamma": self._gamma,
                        "env_class": self._env_class,
                        "env_config": self._env_config,
                    }
                    if self._debug_mode:
                        worker = RayWorker(worker_config)
                        return_dict = worker.run()
                        self._debug_return_list.append(return_dict)
                    else:
                        worker_list.append(ray_worker.remote(worker_config))
                if self._debug_mode:
                    worker_return_dict_list = self._debug_return_list
                else:
                    worker_return_dict_list = ray.get(worker_list)
                for worker_return_dict in worker_return_dict_list:
                    for key in worker_return_dict:
                        if key not in return_dict:
                            return_dict[key] = [worker_return_dict[key]]
                        else:
                            return_dict[key].append(worker_return_dict[key])

        # 使用当前迭代的采样结果更新Q表格
        self._update_q_table(return_dict)


@ray.remote
def ray_worker(config):
    worker = RayWorker(config)
    return_dict = worker.run()
    return return_dict


class RayWorker:
    def __init__(self, config):
        self._config = config
        self._return_dict = {}
        self._env_class = config.get("env_class")
        self._init_action = config.get("init_action")
        self._init_state_index = config["init_state_index"]
        self._q_table = config.get("q_table")
        self._gamma = config.get("gamma", 0.99)
        self._return_container = StateIndexReturnContainer(self._gamma)
        self._env = self._env_class(config["env_config"])

    def run(self):
        init_action = self._init_action
        first_flag = True
        self._return_container.clear()
        done = False
        try:
            current_state, _ = self._env.reset(
                options={"init_state_index": self._init_state_index}
            )
            while True:
                state_index = self._env.get_state_index(current_state)
                q_list = self._q_table.iloc[state_index]

                # 添加调试信息
                if first_flag:
                    action = init_action
                    assert state_index == self._init_state_index
                else:
                    action = q_list.idxmax()

                first_flag = False

                next_state, reward, done, _, _ = self._env.step(action)
                self._return_container.add_reward(
                    reward,
                    state_index,
                    action,
                )
                current_state = next_state
                if done:
                    break
            return self._return_container.get_return()
        except Exception:
            return {}


class StateIndexReturnContainer:
    def __init__(self, gamma: float = 0.99, delta: float = 0.05):
        self._reward_list = []
        self._state_index_list = []
        self._gamma = gamma
        self._delta = delta

    def add_reward(self, item, state_index: int, action: int):
        self._reward_list.append(item)
        self._state_index_list.append((state_index, action))

    def get_return(self) -> dict[int, list[float]]:
        len_reward_list = len(self._reward_list) + 1e-7
        return_list = []
        return_value = 0
        for reward in reversed(self._reward_list):
            return_value = reward - self._delta + self._gamma * return_value
            return_list.append(return_value)
        reversed_return_list = return_list[::-1]
        index_return_dict = {}

        for index in range(len(self._state_index_list)):
            if self._state_index_list[index] not in index_return_dict:
                index_return_dict[self._state_index_list[index]] = reversed_return_list[
                    index
                ]

        return index_return_dict

    def _log_return(self):
        log_list = []
        return_dict = self.get_return()
        for ind, (state_index, action) in enumerate(self._state_index_list):
            log_list.append(
                [
                    ind,
                    state_index,
                    action,
                    self._reward_list[ind],
                    return_dict[(state_index, action)][0],
                ]
            )

        start_index, start_action = self._state_index_list[0]
        end_index, end_action = self._state_index_list[-1]
        log_csv = pd.DataFrame(
            log_list,
            columns=["ind", "state_index", "action", "reward", "return"],
        )

        log_csv.to_csv(
            f"./log/log_{start_index}_{start_action}_{end_index}_{end_action}_{time.time()}.csv",
            index=False,
        )

    def clear(self):
        self._reward_list = []
        self._state_index_list = []
