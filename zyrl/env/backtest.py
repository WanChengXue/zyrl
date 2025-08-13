import ray
import os
import numpy as np
import copy
import gymnasium as gym
from zyrl.env.short_long_env import ShortLongEnv
from zyrl.utils.table_utils import load_dataframe
from zyrl.env.env_utils import sorted_file_by_trade_time


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
        self._init_state_table_by_percentile()
        self._init_env()

    def _init_state_table_by_percentile(self):
        training_data_path = self._config["env_config"]["training_data_path"]
        file_list = os.listdir(training_data_path)
        predict_value_list = []
        self._percentile_dict = {}
        self._state_table = {}
        self._index_state_dict = {}
        for file_name in file_list:
            data = load_dataframe(os.path.join(training_data_path, file_name))
            predict_value_list.append(data["FW_label"].values)
        predict_value_list = np.concatenate(predict_value_list)
        percentile_list = [
            0.5,
            1,
            2,
            3,
            4,
            5,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            95,
            96,
            97,
            98,
            99,
            99.5,
        ]
        for index, percentile in enumerate(percentile_list):
            percentile_value = np.percentile(predict_value_list, percentile)
            self._percentile_dict[f"{100-percentile}%"] = percentile_value
            self._state_table[percentile_value] = len(percentile_list) - index
            self._index_state_dict[len(percentile_list) - index] = percentile_value
        self._config["env_config"]["percentile_dict"] = self._percentile_dict
        self._config["env_config"]["state_table"] = self._state_table
        self._config["env_config"]["index_state_dict"] = self._index_state_dict

    def _load_q_table(self):
        self._q_table = load_dataframe(self._config["q_table_path"])
        self._tuple_to_int_dict = {}
        for index, tuple_index in enumerate(self._q_table.index):
            self._tuple_to_int_dict[tuple_index] = index

    def _init_env(self):
        self._env = ShortLongEnv(self._config["env_config"])

    def _get_action(self, state: gym.spaces.Dict):
        state_index = state["forward_value_index"].item()
        holding = state["holding"].item()
        q_index = self._tuple_to_int_dict[f"({holding}, {state_index})"]
        q_list = self._q_table.iloc[q_index]
        action = q_list.idxmax()
        return int(action)

    def run(self):
        reward_list = []
        done = False
        current_state, info = self._env.reset()
        while not done:
            action = self._get_action(current_state)
            current_state, reward, done, _, info = self._env.step(action)
            reward_list.append(reward)
        return {self._config["env_config"]["file_name"]: reward_list}
