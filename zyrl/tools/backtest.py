import ray
import os
import numpy as np
import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import random
import pandas as pd
from zyrl.utils.table_utils import load_dataframe
from zyrl.env.env_utils import sorted_file_by_trade_time
from zyrl.env.open_close_env.env_backtest import SplitStateActionEnvBacktest


class Backtest:
    def __init__(self, config: dict):
        self._config = config
        self._backtest_result = {}

    def run(self):
        backtest_node_list = []
        file_list = os.listdir(self._config["env_config"]["training_data_path"])
        for file_name in sorted(file_list):
            config = copy.deepcopy(self._config)
            config["env_config"]["file_name"] = file_name
            # backtest_worker = BacktestWorker(config)
            # backtest_worker.run()
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
        self._use_benchmark = config.get("use_benchmark", False)
        if not self._use_benchmark:
            self._load_q_table()
        self._init_env()
        self._commission_value = self._config["env_config"]["commission_value"]

    def _load_q_table(self):
        self._q_table = {}
        self._q_table["open_long"] = load_dataframe(
            self._config["q_table_path"]["open_long_table_path"]
        )
        self._q_table["open_short"] = load_dataframe(
            self._config["q_table_path"]["open_short_table_path"]
        )
        self._q_table["close_long"] = load_dataframe(
            self._config["q_table_path"]["close_long_table_path"]
        )
        self._q_table["close_short"] = load_dataframe(
            self._config["q_table_path"]["close_short_table_path"]
        )

    def _init_env(self):
        self._env = SplitStateActionEnvBacktest(self._config["env_config"])

    def _get_action(self, state: gym.spaces.Dict) -> tuple[int, int]:
        q_table_index = self._env.get_state_index(state, self._env._rank)
        long_index = q_table_index["long"]
        short_index = q_table_index["short"]
        if self._env._holding == 0:
            long_q_list = self._q_table["open_long"].iloc[long_index]
            short_q_list = self._q_table["open_short"].iloc[short_index]
        if self._env._holding == 1:
            long_q_list = self._q_table["close_long"].iloc[long_index]
            short_q_list = self._q_table["open_short"].iloc[short_index]
        if self._env._holding == -1:
            long_q_list = self._q_table["open_long"].iloc[long_index]
            short_q_list = self._q_table["close_short"].iloc[short_index]

        long_action = long_q_list.idxmax()
        short_action = short_q_list.idxmax()
        # long_action = random.choice([0, 1, 2, 3])
        # short_action = random.choice([0, 1, 2, 3])
        action = (int(long_action), int(short_action))
        return action

    def run(self):
        detailed_table = pd.DataFrame(
            columns=[
                "ts",
                "holding",
                "action",
                "diff_reward",
                "grossReturn",
                "commissionFee",
                "cleanReturn",
            ]
        )
        reward_list = []
        done = False
        current_state, info = self._env.reset(
            options={"file_name": self._config["env_config"]["file_name"]}
        )
        diff_reward = 0
        prev_holding = 0
        op_index = 0
        while not done:
            if self._use_benchmark:
                action = self._env.get_best_action(current_state)
            else:
                action = self._get_action(current_state)
            current_state, reward, done, _, info = self._env.step(action)
            reward_list.append(reward)
            diff_reward += reward
            new_holding = info["current_holding"]
            if len(info["action_op"]) == 2:
                if (
                    info["action_op"]["long"] == "OpenLong"
                    and info["action_op"]["short"] == "OpenShort"
                ):
                    assert new_holding == 0
                    ts = info["current_ts_str"]
                    action_op = "open_short_long"
                    gross_return = (
                        -info["price_info"]["long"]["current_LP"]
                        + info["price_info"]["short"]["current_SP"]
                    )
                    clean_return = gross_return - self._commission_value * 2
                    detailed_table.loc[op_index] = [
                        ts,
                        new_holding,
                        action_op,
                        diff_reward.item(),
                        gross_return.item(),
                        self._commission_value,
                        clean_return.item(),
                    ]
                    op_index += 1
                    diff_reward = 0

            if new_holding != prev_holding:
                ts = info["current_ts_str"]
                action = info["action_op"]
                holding = new_holding
                if len(action) == 2:
                    action_op = (
                        "open_long" if action["long"] == "OpenLong" else "open_short"
                    )
                    if action_op == "open_long":
                        gross_return = -info["price_info"]["long"]["current_LP"]
                    else:
                        gross_return = info["price_info"]["short"]["current_SP"]
                else:
                    if "long" in action:
                        action_op = "open_long" if new_holding == 1 else "close_long"
                        if action_op == "close_long":
                            gross_return = info["price_info"]["long"]["current_SP"]
                        if action_op == "open_long":
                            gross_return = -info["price_info"]["long"]["current_LP"]
                    if "short" in action:
                        action_op = "open_short" if new_holding == -1 else "close_short"
                        if action_op == "close_short":
                            gross_return = -info["price_info"]["short"]["current_LP"]
                        if action_op == "open_short":
                            gross_return = info["price_info"]["short"]["current_SP"]

                clean_return = gross_return - self._commission_value
                detailed_table.loc[op_index] = [
                    ts,
                    holding,
                    action_op,
                    diff_reward.item(),
                    gross_return.item(),
                    self._commission_value,
                    clean_return.item(),
                ]
                op_index += 1
                prev_holding = new_holding
                diff_reward = 0
        # detailed_table.to_csv("./data/backtest_detailed_table.csv")
        return {self._config["env_config"]["file_name"]: reward_list}


if __name__ == "__main__":
    table_folder_name = "delta4/im_20250916_result/sequential"
    test_folder_name = "delta4/im_20250916_test"
    index_folder_name = "delta4/state_index_with_delta4"
    saved_folder_name = "delta4/im_20250916_test_result"
    rank = 4
    pred_using = False
    benchmark_using = False
    random_using = False
    config = {
        "q_table_path": {
            "open_long_table_path": f"./data/{table_folder_name}/open_long/q_table.csv",
            "open_short_table_path": f"./data/{table_folder_name}/open_short/q_table.csv",
            "close_long_table_path": f"./data/{table_folder_name}/close_long/q_table.csv",
            "close_short_table_path": f"./data/{table_folder_name}/close_short/q_table.csv",
        },
        "env_config": {
            "training_data_path": f"data/{test_folder_name}",
            "commission_value": 0.125,
            "index_state_dict_path": f"./data/{index_folder_name}/index_state_dict.npy",
            "state_index_mapping_path": f"./data/{index_folder_name}/state_index_mapping.npy",
            "rank": rank,
            "pred_using": pred_using,
        },
    }
    backtest = Backtest(config)
    reward_list = backtest.run()
    cum_reward_list = np.cumsum(reward_list)
    import os

    if not os.path.exists(f"./data/{saved_folder_name}"):
        os.makedirs(f"./data/{saved_folder_name}")
    if benchmark_using:
        npy_name = "benchmark_reward_list.npy"
        png_name = "cusum_reward_probenv_benchmark_test.png"
    else:
        npy_name = "q_reward_list.npy"
        png_name = "cusum_reward_probenv_q_test.png"
    if random_using:
        png_name = "cusum_reward_probenv_random_test.png"
        npy_name = "random_reward_list.npy"

    np.save(f"./data/{saved_folder_name}/{npy_name}", reward_list)
    plt.plot(cum_reward_list)
    plt.show()
    plt.savefig(f"./data/{saved_folder_name}/{png_name}")
    plt.close()
