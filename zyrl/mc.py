# from tianshou.env import RayVectorEnv
import pandas as pd
import os
import copy
from zyrl.env.short_long_env import ShortLongEnv
from zyrl.utils.table_utils import load_dataframe
from tqdm import tqdm
import numpy as np

import ray
import time


class MCFromStart:
    def __init__(
        self,
        env_class: type[ShortLongEnv],
        init_q_table: pd.DataFrame,
        env_config: dict,
        mc_config: dict,
    ):
        self._saved_q_table_path = mc_config.get("saved_q_table_path")
        self._saved_count_table_path = mc_config.get("saved_count_table_path")
        self._mc_config = mc_config
        self._env_config = env_config
        self._gamma = mc_config.get("gamma", 0.99)
        self._q_table = init_q_table
        self._env_class = env_class
        self._init_count_table()

    def _init_count_table(self):
        self._sample_point_count_table = pd.DataFrame(
            index=self._q_table.index, columns=self._q_table.columns
        )
        self._sample_point_count_table.fillna(0, inplace=True)

    def _init_q_table_bak(self):
        holding_list = [1, 0, -1]
        state_table_index_list = [i + 1 for i in range(22)]
        action_list = [-1, 0, 1]
        total_state = [
            (holding, state_index)
            for holding in holding_list
            for state_index in state_table_index_list
        ]
        self._q_table = pd.DataFrame(index=total_state, columns=action_list)
        self._sample_point_count_table = pd.DataFrame(
            index=total_state, columns=action_list
        )
        self._sample_point_count_table.fillna(0, inplace=True)
        self._q_table.fillna(0.0, inplace=True)
        for holding, state_index in total_state:
            for action in action_list:
                if holding == 1:
                    if state_index < 16 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 16 and state_index < 20 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 20 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    else:
                        self._q_table.at[(holding, state_index), action] = 0.0
                elif holding == 0:
                    if state_index < 4 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 4 and state_index < 20 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 20 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    else:
                        self._q_table.at[(holding, state_index), action] = 0.0

                elif holding == -1:
                    if state_index < 4 and action == 1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 4 and state_index < 8 and action == 0:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    elif state_index >= 8 and action == -1:
                        self._q_table.at[(holding, state_index), action] = 1.0
                    else:
                        self._q_table.at[(holding, state_index), action] = 0.0
        self._q_table.index.name = "state"
        self._q_table.columns.name = "action"
        self._convert_tuple_index_to_int()
        self._q_table.to_csv("./init_q_table.csv")

    def _init_state_table_by_percentile(self):
        training_data_path = self._config.get("training_data_path")
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

    def _get_state_table_index(self, state_table: dict) -> int:
        for key, value in state_table.items():
            if value < key:
                return value
        return len(state_table) + 1

    def run(self):
        end_flag = False
        while not end_flag:
            before_update_q_table = copy.deepcopy(self._q_table)
            self._collect_data()
            end_flag = self._end_check(before_update_q_table, self._q_table)
            self._saved_q_table()

    def _saved_q_table(self):
        self._q_table.to_csv(self._saved_q_table_path)
        self._sample_point_count_table.to_csv(self._saved_count_table_path)

    def _end_check(self, current_state: pd.DataFrame, next_state: pd.DataFrame) -> bool:
        mse_error = np.mean((current_state.values - next_state.values) ** 2)
        if mse_error < 1e-6:
            return True
        return False


    def _update_q_table(self, return_dict: dict[int, list[float]]):
        for state_index in self._sample_point_count_table.index:
            for action_ind, _ in enumerate(self._q_table.columns):
                if (state_index, action_ind) not in return_dict:
                    continue
                count_num = self._sample_point_count_table.iloc[state_index][action_ind]
                self._q_table.iloc[state_index][action_ind] = (
                    self._q_table.iloc[state_index][action_ind] * count_num
                    + np.sum(return_dict[state_index, action_ind])
                ) / (count_num + len(return_dict[state_index, action_ind]))
                self._sample_point_count_table.iloc[state_index][action_ind] = (
                    count_num + len(return_dict[state_index, action_ind])
                )

    def _collect_data(self):
        return_dict = {}
        self._sample_env_num = self._config.get("sample_env_num", 10)
        for state_index in tqdm(self._q_table.index):
            for init_action in self._q_table.columns:
                worker_list = []
                for _ in range(self._sample_env_num):
                    worker_config = {
                        "init_state_index": state_index,
                        "init_action": init_action,
                        "q_table": self._q_table,
                        "gamma": self._gamma,
                        "env_config": self._config.get("env_config"),
                    }
                    worker_list.append(ray_worker.remote(worker_config))
                worker_return_dict_list = ray.get(worker_list)
                for worker_return_dict in worker_return_dict_list:
                    for key in worker_return_dict:
                        if key not in return_dict:
                            return_dict[key] = worker_return_dict[key]
                        else:
                            return_dict[key].extend(worker_return_dict[key])
        self._update_q_table(return_dict)

    def _collect_data_bak(self):
        return_dict = {}
        for init_holding, init_state_index in tqdm(self._q_table.index):
            for init_action in self._q_table.columns:
                if not (init_holding, init_action) == (0, 0):
                    worker_list = []
                    for file_name in self._file_list:
                        worker_config = {
                            "file_name": file_name,
                            "init_state_index": init_state_index,
                            "init_holding": init_holding,
                            "q_table": self._q_table,
                            "gamma": self._gamma,
                            "training_data_path": self._config.get(
                                "training_data_path"
                            ),
                            "training_market_data_path": self._config.get(
                                "training_market_data_path"
                            ),
                            "test_data_path": self._config.get("test_data_path"),
                            "test_market_data_path": self._config.get(
                                "test_market_data_path"
                            ),
                            "commission_value": self._config.get(
                                "commission_value", 0.12
                            ),
                            "init_action": init_action,
                            "percentile_dict": self._percentile_dict,
                            "state_table": self._state_table,
                            "index_state_dict": self._index_state_dict,
                        }
                        worker_list.append(ray_worker.remote(worker_config))
                    worker_return_dict_list = ray.get(worker_list)
                    for worker_return_dict in worker_return_dict_list:
                        for key in worker_return_dict:
                            if key not in return_dict:
                                return_dict[key] = worker_return_dict[key]
                            else:
                                return_dict[key].extend(worker_return_dict[key])
                    # fix (state_index, holding, action), N个episodes, update Q(s,a)
                    self._update_q_table(return_dict)


@ray.remote
def ray_worker(config):
    worker = RayWorker(config)
    init_action = config.get("init_action")
    return_dict = worker.run(init_action)
    return return_dict


@ray.remote
def ray_node(config, start_option):
    ray_node = RayNode(config)
    return ray_node.run(start_option)


class RayWorker:
    def __init__(self, config):
        self._config = config
        self._return_dict = {}
        self._init_start_index_list()

    def _init_start_index_list(self):
        file_path = os.path.join(
            self._config["training_data_path"], self._config["file_name"]
        )
        state_index = self._config["init_state_index"]
        index_state_dict = self._config["index_state_dict"]
        self._start_index_list = ShortLongEnv.get_start_index_list(
            file_path, state_index, index_state_dict
        )

    def run(self, init_action: int):
        return_dict = {}
        ray_node_return_list = []
        for start_index in self._start_index_list:
            start_option = {
                "start_index": start_index,
                "init_action": init_action,
            }
            worker_node = RayNode(self._config)
            ray_node_return = worker_node.run(start_option)
            ray_node_return_list.append(ray_node_return)
        #     ray_node_list.append(ray_node.remote(self._config, start_option))
        # ray_node_return_list = ray.get(ray_node_list)
        for ray_node_return in ray_node_return_list:
            for key in ray_node_return:
                if key not in return_dict:
                    return_dict[key] = ray_node_return[key]
                else:
                    return_dict[key].extend(ray_node_return[key])
        return return_dict


class RayNode:
    def __init__(self, config):
        self._file_name = config.get("file_name")
        self._init_state_index = config.get("init_state_index", 1)
        self._init_holding = config["init_holding"]
        self._q_table = config.get("q_table")
        self._return_dict = {}
        self._gamma = config.get("gamma", 0.99)
        self._return_container = StateIndexReturnContainer(self._gamma)
        self._convert_tuple_index_to_int()
        self._env = ShortLongEnv(config)

    def _convert_tuple_index_to_int(self):
        self._tuple_to_int_dict = {}
        for index, tuple_index in enumerate(self._q_table.index):
            self._tuple_to_int_dict[tuple_index] = index

    def run(self, start_option: dict):
        start_index = start_option["start_index"]
        init_action = start_option["init_action"]
        first_flag = True
        self._return_container.clear()
        done = False
        current_state, info = self._env.reset(
            option={
                "start_index": start_index,
                "file_name": self._file_name,
            }
        )
        while True:
            table_key = (
                current_state["holding"].item(),
                current_state["forward_value_index"].item(),
            )

            q_list = self._q_table.iloc[self._tuple_to_int_dict[table_key]]

            # 添加调试信息
            action = init_action if first_flag else q_list.idxmax()

            first_flag = False

            next_state, reward, done, _, info = self._env.step(action)
            # log_list.append([info["predict_value"], info["current_holding"], current_state["forward_value_index"].item(), action, reward])
            self._return_container.add_reward(
                reward,
                current_state["forward_value_index"].item(),
                current_state["holding"].item(),
                action,
            )
            # (state_index, holding, action, reward)
            current_state = next_state
            if done:
                break
        return self._return_container.get_return()


class StateIndexReturnContainer:
    def __init__(self, gamma: float = 0.99):
        self._reward_list = []
        self._state_index_list = []
        self._gamma = gamma

    def add_reward(self, item, state_index: int, holding: int, action: int):
        self._reward_list.append(item)
        self._state_index_list.append((state_index, holding, action))

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
            # else:
            #     index_return_dict[self._state_index_list[index]].append(
            #         reversed_return_list[index]
            #     )
        return index_return_dict

    def _log_return(self):
        log_list = []
        return_dict = self.get_return()
        for ind, (state_index, holding, action) in enumerate(self._state_index_list):
            log_list.append(
                [
                    ind,
                    state_index,
                    holding,
                    action,
                    self._reward_list[ind],
                    return_dict[(state_index, holding, action)][0],
                ]
            )

        start_index, start_holding, start_action = self._state_index_list[0]
        end_index, end_holding, end_action = self._state_index_list[-1]
        log_csv = pd.DataFrame(
            log_list,
            columns=["ind", "state_index", "holding", "action", "reward", "return"],
        )

        log_csv.to_csv(
            f"./log/log_{start_index}_{start_holding}_{start_action}_{end_index}_{end_holding}_{end_action}_{time.time()}.csv",
            index=False,
        )

    def clear(self):
        self._reward_list = []
        self._state_index_list = []
