# from tianshou.env import RayVectorEnv
import pandas as pd
import os
from zyrl.env.short_long_env import ShortLongEnv

class MCFromStart:
    def __init__(self, config):
        self._file_path = config.get("file_path")
        self._init_q_table()
        self._read_data_list()

    def _init_q_table(self):
        holding_list = [1, 0, -1]
        state_table_index_list = [i+1 for i in range(22)]
        action_list = [-1, 0, 1]
        total_state = [(holding, state_index) for holding in holding_list for state_index in state_table_index_list]
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
        return len(state_table)


    def _read_data_list(self):
        self._file_list = os.listdir(self._file_path)


    def run(self):
        pass


    def _collect_data(self):
        pass


class RayWorker:
    def __init__(self, config):
        self._file_name = config.get("file_name")
        self._init_state_index = config.get("init_state_index", 1)
        self._init_holding = config.get("init_holding", 0)
        self._q_table = config.get("q_table")
        self._return_dict = {}
        self._gamma = config.get("gamma", 0.99)
        self._return_container = ReturnContainer(self._gamma)
        self._env = ShortLongEnv(config)

    def run(self):
        self._return_container.clear()
        current_state, info = self._env.reset(option={"state_index": self._init_state_index, "file_name": self._file_name})
        while True:
            table_key = (current_state["current_holding"], current_state["current_index"])
            q_list = self._q_table.loc[current_state]
            action = q_list.idxmax()
            next_state, reward, done, _, info = self._env.step(action)
            self._return_container.add_reward(reward)
            current_state = next_state
            self._env.step(action)
            if done:
                break


class ReturnContainer:
    def __init__(self, gamma: float =0.99):
        self._reward_list = []
        self._gamma = gamma

    def add_reward(self, item):
        self._reward_list.append(item)


    def get_return(self):
        return_list = []
        return_value = 0
        for reward in reversed(self._reward_list):
            return_value = reward + self._gamma * return_value
            return_list.append(return_value)
        return return_list[::-1]

    def clear(self):
        self._reward_list = []
