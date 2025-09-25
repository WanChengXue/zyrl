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
    ):
        self._saved_q_table_path = mc_config.get("saved_q_table_path")
        self._saved_count_table_path = mc_config.get("saved_count_table_path")
        self._mask_table = mc_config.get("mask_table", None)
        self._mc_config = mc_config
        self._env_config = env_config
        self._gamma = mc_config.get("gamma", 0.99)
        self._q_table = init_q_table
        self._env_class = env_class
        self._mse_loss_list = []
        self._init_count_table()
        self._saved_q_table()

    def _init_count_table(self):
        self._sample_point_count_table = pd.DataFrame(
            index=self._q_table.index, columns=self._q_table.columns
        )
        self._sample_point_count_table.fillna(0, inplace=True)

    def run(self):
        end_flag = False
        while not end_flag:
            before_update_q_table = copy.deepcopy(self._q_table)
            self._collect_data()
            end_flag = self._end_check(before_update_q_table, self._q_table)
            self._saved_q_table()
            plt.plot(self._mse_loss_list)
            plt.savefig(self._mc_config.get("mse_loss_plot_path"))
            plt.close()

    def _saved_q_table(self):
        self._q_table.to_csv(self._saved_q_table_path)
        self._sample_point_count_table.to_csv(self._saved_count_table_path)

    def _end_check(self, current_state: pd.DataFrame, next_state: pd.DataFrame) -> bool:
        mse_error = np.mean((current_state.values - next_state.values) ** 2)
        self._mse_loss_list.append(mse_error)
        print(f"mse_error: {mse_error}")
        if mse_error < 5e-5:
            return True
        return False

    def _update_q_table(self, return_dict: dict[int, list[float]]):
        for state_index in self._sample_point_count_table.index:
            for action_ind, action in enumerate(self._q_table.columns):
                if (state_index, action) not in return_dict:
                    continue
                count_num = self._sample_point_count_table.iloc[state_index][action]
                self._q_table.iloc[state_index, action_ind] = (
                    self._q_table.iloc[state_index][action] * count_num
                    + np.sum(return_dict[state_index, action])
                ) / (count_num + len(return_dict[state_index, action]))
                self._sample_point_count_table.iloc[state_index, action_ind] = (
                    count_num + len(return_dict[state_index, action])
                )

    def _collect_data(self):
        return_dict = {}
        self._sample_env_num = self._mc_config.get("sample_env_num", 10)
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
                        "q_table": self._q_table,
                        "gamma": self._gamma,
                        "env_class": self._env_class,
                        "env_config": self._env_config,
                    }
                    # worker = RayWorker(worker_config)
                    # worker_return_dict = worker.run(init_action)
                    # return_dict[state_index, init_action] = worker_return_dict
                    worker_list.append(ray_worker.remote(worker_config))

                worker_return_dict_list = ray.get(worker_list)
                for worker_return_dict in worker_return_dict_list:
                    for key in worker_return_dict:
                        if key not in return_dict:
                            return_dict[key] = [worker_return_dict[key]]
                        else:
                            return_dict[key].append(worker_return_dict[key])
        self._update_q_table(return_dict)


@ray.remote
def ray_worker(config):
    worker = RayWorker(config)
    init_action = config.get("init_action")
    return_dict = worker.run(init_action)
    return return_dict


class RayWorker:
    def __init__(self, config):
        self._config = config
        self._return_dict = {}
        self._env_class = config.get("env_class")

    def run(self, init_action: int):
        start_option = {
            "init_action": init_action,
        }
        worker_node = RayNode(self._config, self._env_class)
        ray_node_return = worker_node.run(start_option)
        return ray_node_return


class RayNode:
    def __init__(self, config, env_class):
        self._init_state_index = config["init_state_index"]
        self._q_table = config.get("q_table")
        self._return_dict = {}
        self._gamma = config.get("gamma", 0.99)
        self._return_container = StateIndexReturnContainer(self._gamma)
        self._env = env_class(config["env_config"])

    def run(self, start_option: dict):
        init_action = start_option["init_action"]
        first_flag = True
        self._return_container.clear()
        done = False
        try:
            current_state, info = self._env.reset(
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

                next_state, reward, done, _, info = self._env.step(action)
                # log_list.append([info["predict_value"], info["current_holding"], current_state["forward_value_index"].item(), action, reward])
                self._return_container.add_reward(
                    reward,
                    state_index,
                    action,
                )
                # (state_index, holding, action, reward)
                current_state = next_state
                if done:
                    break
            return self._return_container.get_return()
        except Exception as e:
            return {}


class StateIndexReturnContainer:
    def __init__(self, gamma: float = 0.99):
        self._reward_list = []
        self._state_index_list = []
        self._gamma = gamma

    def add_reward(self, item, state_index: int, action: int):
        self._reward_list.append(item)
        self._state_index_list.append((state_index, action))

    def get_return(self) -> dict[int, list[float]]:
        return_list = []
        return_value = 0
        for reward in reversed(self._reward_list):
            return_value = reward + self._gamma * return_value
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
