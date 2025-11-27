import ray
import os
import numpy as np
import copy
import gymnasium as gym

from zyrl.utils.table_utils import load_dataframe
from zyrl.env.env_utils import sorted_file_by_trade_time
from zyrl.env.short_long.short_long_env import ShortLongEnv


class Backtest:
    def __init__(self, config: dict):
        self._config = config
        self._backtest_result = {}

    def run(self):
        backtest_node_list = []
        file_list = os.listdir(self._config["env_config"]["training_data_path"])
        for file_name in file_list:
            config = copy.deepcopy(self._config)
            config["env_config"]["file_name"] = file_name
            backtest_node_list.append(backtest_node.remote(config))
        backtest_result_list = ray.get(backtest_node_list)
        for backtest_result in backtest_result_list:
            self._backtest_result.update(backtest_result)

        concatenated_reward_list = []
        for file_name in sorted_file_by_trade_time(self._backtest_result.keys()):
            concatenated_reward_list.extend(self._backtest_result[file_name])
        return concatenated_reward_list


@ray.remote
def backtest_node(config: dict):
    backtest_worker = BacktestWorker(config)
    return backtest_worker.run()


class BacktestWorker:
    def __init__(self, config: dict):
        self._config = config
        self._load_q_table()
        self._init_env()

    def _load_q_table(self):
        self._q_table = load_dataframe(self._config["q_table_path"])

    def _init_env(self):
        self._env = ShortLongEnv(self._config["env_config"])

    def _get_action(self, state: gym.spaces.Dict):
        q_table_index = self._env.get_state_index(state)
        q_list = self._q_table.iloc[q_table_index]
        action = q_list.idxmax()
        return int(action)

    def run(self):
        reward_list = []
        done = False
        current_state, info = self._env.reset(
            option={"file_name": self._config["env_config"]["file_name"]}
        )
        while not done:
            action = self._get_action(current_state)
            current_state, reward, done, _, info = self._env.step(action)
            reward_list.append(reward)
        return {self._config["env_config"]["file_name"]: reward_list}
