from zyrl.mc import MCFromStart
from zyrl.env.backtest import Backtest
from zyrl.env.short_long_env import ShortLongEnv
from zyrl.utils.table_utils import load_dataframe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def init_q_table():
    holding_list = [1, 0, -1]
    state_table_index_list = [i + 1 for i in range(22)]
    action_list = [-1, 0, 1]
    total_state = [
        (holding, state_index)
        for holding in holding_list
        for state_index in state_table_index_list
    ]
    q_table = pd.DataFrame(index=total_state, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    for holding, state_index in total_state:
        for action in action_list:
            if holding == 1:
                if state_index < 16 and action == 1:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 16 and state_index < 20 and action == 0:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 20 and action == -1:
                    q_table.at[(holding, state_index), action] = 1.0
                else:
                    q_table.at[(holding, state_index), action] = 0.0
            elif holding == 0:
                if state_index < 4 and action == 1:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 4 and state_index < 20 and action == 0:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 20 and action == -1:
                    q_table.at[(holding, state_index), action] = 1.0
                else:
                    q_table.at[(holding, state_index), action] = 0.0

            elif holding == -1:
                if state_index < 4 and action == 1:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 4 and state_index < 8 and action == 0:
                    q_table.at[(holding, state_index), action] = 1.0
                elif state_index >= 8 and action == -1:
                    q_table.at[(holding, state_index), action] = 1.0
                else:
                    q_table.at[(holding, state_index), action] = 0.0
    q_table.index.name = "state"
    q_table.columns.name = "action"
    q_table.index = [i for i in range(len(q_table.index))]
    return q_table


def init_percentile_dict(
    training_data_path: str,
    state_table_saved_path: str,
    index_state_dict_saved_path: str,
    percentile_dict_saved_path: str,
):
    file_list = os.listdir(training_data_path)
    predict_value_list = []
    state_table = {}
    percentile_dict = {}
    index_state_dict = {}
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
        percentile_dict[f"{100-percentile}%"] = percentile_value
        state_table[percentile_value] = len(percentile_list) - index
        index_state_dict[len(percentile_list) - index] = percentile_value
    np.save(state_table_saved_path, state_table)
    np.save(index_state_dict_saved_path, index_state_dict)
    np.save(percentile_dict_saved_path, percentile_dict)


def test_mc():

    env_config = {
        "file_path": "data/predict_data",
        "training_data_path": "data/predict_data",
        "training_market_data_path": "data/market_data",
        "commission_value": 0.12,
        "state_table_path": "./data/state_table.npy",
        "index_state_dict_path": "./data/index_state_dict.npy",
        "percentile_dict_path": "./data/percentile_dict.npy",
        "state_index_mapping_path": "./data/state_index_mapping.npy",
    }

    mc_config = {
        "saved_q_table_path": "./q_table.csv",
        "saved_count_table_path": "./count_table.csv",
        "gamma": 0.99,
        "sample_env_num": 30,
    }

    mc = MCFromStart(ShortLongEnv, init_q_table(), env_config, mc_config)

    mc.run()


def test_backtest():
    config = {
        "q_table_path": "./q_table.csv",
        "env_config": {
            "training_data_path": "data/predict_data",
            "training_market_data_path": "data/market_data",
            "commission_value": 0.12,
            "util_termination": True,
            "state_index_mapping_path": "./data/state_index_mapping.npy",
            "state_table_path": "./data/state_table.npy",
            "index_state_dict_path": "./data/index_state_dict.npy",
            "percentile_dict_path": "./data/percentile_dict.npy",
        },
    }
    backtest = Backtest(config)
    reward_list = backtest.run()
    cum_reward_list = np.cumsum(reward_list)
    plt.plot(cum_reward_list)
    plt.show()
    plt.savefig("./figure/cusum_reward_20250818.png")
    plt.close()


test_backtest()
# init_percentile_dict(
#     training_data_path="data/predict_data",
#     state_table_saved_path="./data/state_table.npy",
#     index_state_dict_saved_path="./data/index_state_dict.npy",
#     percentile_dict_saved_path="./data/percentile_dict.npy",
# )

# test_mc()
# index = 0
# state_index_mapping = {
#     "index_to_state": {},
#     "state_to_index": {},
# }
# for act in [1, 0, -1]:
#     for i in range(1,23):
#         state_index_mapping["index_to_state"][index] = (i, act)
#         state_index_mapping["state_to_index"][(i, act)] = index
#         index += 1

# np.save("./data/state_index_mapping.npy", state_index_mapping)
