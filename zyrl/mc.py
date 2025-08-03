# from tianshou.env import RayVectorEnv
import pandas as pd
import os
import copy
from zyrl.env.short_long_env import ShortLongEnv


class MCFromStart:
    def __init__(self, config):
        self._file_path = config.get("file_path")
        self._saved_q_table_path = config.get("saved_q_table_path")
        self._gamma = config.get("gamma", 0.99)
        self._init_q_table()
        self._read_data_list()

    def _init_q_table(self):
        holding_list = [1, 0, -1]
        state_table_index_list = [i + 1 for i in range(22)]
        action_list = [-1, 0, 1]
        total_state = [
            (holding, state_index)
            for holding in holding_list
            for state_index in state_table_index_list
        ]
        self._q_table = pd.DataFrame(index=total_state, columns=action_list)
        self._q_table.fillna(0, inplace=True)
        for holding, state_index in total_state:
            for action in action_list:
                if holding == 1:
                    if state_index < 16 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 16 and state_index < 20 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 20 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1
                    else:
                        self._q_table.at[(holding, state_index), action] = 0
                elif holding == 0:
                    if state_index < 4 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 4 and state_index < 20 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 20 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1
                    else:
                        self._q_table.at[(holding, state_index), action] = 0

                elif holding == -1:
                    if state_index < 4 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 4 and state_index < 7 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1
                    elif state_index >= 7 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1
                    else:
                        self._q_table.at[(holding, state_index), action] = 0
        self._q_table.index.name = "state"
        self._q_table.columns.name = "action"

    def _get_state_table_index(self, state_table: dict) -> int:
        for key, value in state_table.items():
            if value < key:
                return value
        return len(state_table) + 1

    def _read_data_list(self):
        self._file_list = os.listdir(self._file_path)

    def run(self):
        end_flag = False
        while not end_flag:
            self._collect_data()
            self._update_q_table()
            end_flag = self._end_check()

    def _saved_q_table(self):
        self._q_table.to_csv(self._saved_q_table_path)

    def _end_check(self, current_state: pd.DataFrame, next_state: pd.DataFrame) -> bool:
        pass

    def _update_q_table(self, return_dict: dict[int, list[float]]):
        pass

    def _collect_data(self):
        return_dict = {}
        for init_state_index, init_holding in self._q_table.index:
            worker_list = []
            for file_name in self._file_list:
                worker_config = {
                    "file_name": file_name,
                    "init_state_index": init_state_index,
                    "init_holding": init_holding,
                    "q_table": self._q_table,
                    "gamma": self._gamma,
                }
                worker_list.append(RayWorker(worker_config))

            for worker in worker_list:
                worker_return_dict = worker.run()
                for key in worker_return_dict:
                    if key not in return_dict:
                        return_dict[key] = worker_return_dict[key]
                    else:
                        return_dict[key].extend(worker_return_dict[key])
        return return_dict


class RayWorker:
    def __init__(self, config):
        self._file_name = config.get("file_name")
        self._init_state_index = config.get("init_state_index", 1)
        self._init_holding = config.get("init_holding", 0)
        self._q_table = config.get("q_table")
        self._return_dict = {}
        self._gamma = config.get("gamma", 0.99)
        self._return_container = StateIndexReturnContainer(self._gamma)
        self._env = ShortLongEnv(config)

    def run(self):
        self._return_container.clear()
        return_dict = {}
        start_index_list = self._env.get_start_index_list(self._init_state_index)
        for start_index in start_index_list:
            current_state, info = self._env.reset(
                option={
                    "state_index": self._init_state_index,
                    "file_name": self._file_name,
                }
            )
            while True:
                table_key = (
                    current_state["current_holding"],
                    current_state["current_index"],
                )
                q_list = self._q_table.loc[current_state]
                action = q_list.idxmax()
                next_state, reward, done, _, info = self._env.step(action)
                self._return_container.add_reward(
                    reward, next_state["forward_value_index"].item()
                )
                current_state = next_state
                self._env.step(action)
                if done:
                    break
            return_dict.update(self._return_container.get_return())
        return return_dict


class StateIndexReturnContainer:
    def __init__(self, gamma: float = 0.99):
        self._reward_list = []
        self._state_index_list = []
        self._gamma = gamma

    def add_reward(self, item, state_index: int):
        self._reward_list.append(item)
        self._state_index_list.append(state_index)

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
                index_return_dict[self._state_index_list[index]] = [
                    reversed_return_list[index]
                ]
            else:
                index_return_dict[self._state_index_list[index]].append(
                    reversed_return_list[index]
                )
        return index_return_dict

    def clear(self):
        self._reward_list = []
        self._state_index_list = []
