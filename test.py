import pandas as pd
import matplotlib.pyplot as plt
from zyrl.mc import MCFromStart
from zyrl.env.open_close_env.env import SplitStateActionEnv


def init_close_long_q_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                if delta2 > 0:
                    q_table.at[index, 2] = 1.0
                if delta2 == 0 and delta3 > 0:
                    q_table.at[index, 3] = 1.0
                index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_close_short_q_table():
    pass


def init_open_long_q_table():
    pass


def init_open_short_q_table():
    pass


def test_mc():
    config = {
        "data_path": "./data/split_state_action_data",
        "env_type": "close_long",
        "commission_value": 0.12,
        "index_state_dict_path": "./data/split_state_action_data_index/index_state_dict.npy",
        "state_index_mapping_path": "./data/split_state_action_data_index/state_index_mapping.npy",
    }

    mc_config = {
        "saved_q_table_path": "./data/split_state_action_data_index/q_table_close_long.csv",
        "saved_count_table_path": "./data/split_state_action_data_index/count_table_close_long.csv",
        "gamma": 0.99,
        "sample_env_num": 30,
        "mse_loss_plot_path": "./data/split_state_action_data_index/mse_loss_plot_close_long.png",
    }

    mc = MCFromStart(SplitStateActionEnv, init_close_long_q_table(), config, mc_config)
    mc.run()


if __name__ == "__main__":
    test_mc()
