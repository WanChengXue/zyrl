import pandas as pd
import matplotlib.pyplot as plt
from zyrl.mc import MCFromStart
from zyrl.env.open_close_env.env import SplitStateActionEnv


def init_mask_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    mask_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    mask_table.fillna(False, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                mask_table.at[index, 0] = True
                if delta1 != 0:
                    mask_table.at[index, 1] = True
                if delta2 != 0:
                    mask_table.at[index, 2] = True
                if delta3 != 0:
                    mask_table.at[index, 3] = True
                index += 1
    mask_table.index.name = "state_index"
    mask_table.columns.name = "action"
    return mask_table


def init_mask_table_with_pred():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 12)]
    mask_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    mask_table.fillna(False, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for region_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    mask_table.at[index, 0] = True
                    if delta1 != 0:
                        mask_table.at[index, 1] = True
                    if delta2 != 0:
                        mask_table.at[index, 2] = True
                    if delta3 != 0:
                        mask_table.at[index, 3] = True
                    index += 1
    mask_table.index.name = "state_index"
    mask_table.columns.name = "action"
    return mask_table


def init_close_long_q_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                if delta1 == 0:
                    q_table.at[index, 1] = -10.0
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 == 0:
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 == 0:
                    q_table.at[index, 2] = 1.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 > 0:
                    q_table.at[index, 2] = 1.0

                index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_close_short_q_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                if delta1 == 0:
                    q_table.at[index, 1] = -10.0
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 == 0:
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 == 0:
                    q_table.at[index, 2] = 1.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 > 0:
                    q_table.at[index, 2] = 1.0

                index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_long_q_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                if delta1 == 0:
                    q_table.at[index, 1] = -10.0
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 == 0:
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 == 0:
                    q_table.at[index, 2] = 1.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 > 0:
                    q_table.at[index, 2] = 1.0

                index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_short_q_table():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                if delta1 == 0:
                    q_table.at[index, 1] = -10.0
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 == 0:
                    q_table.at[index, 2] = -10.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 == 0:
                    q_table.at[index, 2] = 1.0
                    q_table.at[index, 3] = -10.0

                if delta1 > 0 and delta2 > 0 and delta3 > 0:
                    q_table.at[index, 2] = 1.0

                index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_close_long_q_table_with_pred():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 12)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for region_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_close_short_q_table_with_pred():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 12)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for region_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_long_q_table_with_pred():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 12)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for region_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_short_q_table_with_pred():
    action_list = [0, 1, 2, 3]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 12)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for region_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def test_mc(env_type, data_folder, pred_using=False):
    print("env_type: ", env_type, "pred_using", pred_using)
    if pred_using:
        if env_type == "open_short":
            q_table = init_open_short_q_table_with_pred()
        elif env_type == "open_long":
            q_table = init_open_long_q_table_with_pred()
        elif env_type == "close_short":
            q_table = init_close_short_q_table_with_pred()
        elif env_type == "close_long":
            q_table = init_close_long_q_table_with_pred()
        mask_table = init_mask_table_with_pred()
        table_folder = f"delta3/{data_folder}_pred_using"
    else:
        if env_type == "open_short":
            q_table = init_open_short_q_table()
        elif env_type == "open_long":
            q_table = init_open_long_q_table()
        elif env_type == "close_short":
            q_table = init_close_short_q_table()
        elif env_type == "close_long":
            q_table = init_close_long_q_table()
        mask_table = init_mask_table_with_pred()
        table_folder = f"delta3/{data_folder}"
    index_table_folder = "delta3/state_index_with_pred"
    config = {
        "data_path": f"./data/delta3/{data_folder}",
        "env_type": env_type,
        "pred_using": pred_using,
        "commission_value": 0.128,
        "rank": 3,
        "index_state_dict_path": f"./data/{index_table_folder}/index_state_dict.npy",
        "state_index_mapping_path": f"./data/{index_table_folder}/state_index_mapping.npy",
        "start_index_path": f"./data/{table_folder}_result",
        "fixed_commission": True,
    }

    mc_config = {
        "saved_q_table_path": f"./data/{table_folder}_result/q_table_{env_type}.csv",
        "saved_count_table_path": f"./data/{table_folder}_result/count_table_{env_type}.csv",
        "gamma": 0.99,
        "sample_env_num": 30,
        "mse_loss_plot_path": f"./data/{table_folder}_result/mse_loss_plot_{env_type}.png",
        "mask_table": mask_table,
    }

    mc = MCFromStart(SplitStateActionEnv, q_table, config, mc_config)
    mc.run()


def compare_figure(q_path, bechmark_path, data_folder):
    import numpy as np

    q_reward_list = np.load(q_path, allow_pickle=True)
    bechmark_reward_list = np.load(bechmark_path, allow_pickle=True)
    plt.plot(np.cumsum(q_reward_list))
    plt.plot(np.cumsum(bechmark_reward_list))
    plt.legend(["q", "bechmark"])
    plt.savefig(f"./data/{data_folder}_test_result/compare_figure.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default="close_short")
    parser.add_argument("--data_folder", type=str, default="im_20250905")
    parser.add_argument("--pred_using", type=bool, default=True)
    args = parser.parse_args()
    env_type = args.env_type
    # test_mc(env_type, args.data_folder, args.pred_using)
    compare_figure(
        f"./data/{args.data_folder}_test_result/q_reward_list.npy",
        f"./data/{args.data_folder}_test_result/benchmark_reward_list.npy",
        args.data_folder,
    )
    # q_table = init_close_long_q_table()
    # q_table.to_csv("./table/init_q_open_close.csv")
