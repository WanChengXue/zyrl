import pandas as pd
import matplotlib.pyplot as plt
from zyrl.env.open_close_env.env import SplitStateActionEnv
from zyrl.sequential.task import Task
from zyrl.sequential.edge import TaskLink
from zyrl.sequential.controller import Controller


def init_mask_table():
    action_list = [0, 1, 2, 3, 4]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 6)]
    mask_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    mask_table.fillna(False, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for delta4 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    mask_table.at[index, 0] = True
                    if delta1 != 0:
                        mask_table.at[index, 1] = True
                    if delta2 != 0:
                        mask_table.at[index, 2] = True
                    if delta3 != 0:
                        mask_table.at[index, 3] = True
                    if delta4 != 0:
                        mask_table.at[index, 4] = True
                    index += 1
    mask_table.index.name = "state_index"
    mask_table.columns.name = "action"
    return mask_table


def init_close_long_q_table():
    action_list = [0, 1, 2, 3, 4]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for delta4 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_close_short_q_table():
    action_list = [0, 1, 2, 3, 4]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for delta4 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_long_q_table():
    action_list = [0, 1, 2, 3, 4]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for delta4 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def init_open_short_q_table():
    action_list = [0, 1, 2, 3, 4]
    state_table_index_list = [i for i in range(6 * 6 * 6 * 6)]
    q_table = pd.DataFrame(index=state_table_index_list, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    index = 0
    for delta1 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for delta2 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for delta3 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                for delta4 in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    if delta1 == 0:
                        q_table.at[index, 1] = -10.0
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 == 0:
                        q_table.at[index, 2] = -10.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 3] = -10.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 == 0:
                        q_table.at[index, 2] = 1.0
                        q_table.at[index, 4] = -10.0

                    if delta1 > 0 and delta2 > 0 and delta3 > 0 and delta4 > 0:
                        q_table.at[index, 2] = 1.0

                    index += 1
    q_table.index.name = "state_index"
    q_table.columns.name = "action"
    return q_table


def get_config(env_type, file_path_dict, rank, pred_using):
    if env_type == "open_short":
        q_table = init_open_short_q_table()
    elif env_type == "open_long":
        q_table = init_open_long_q_table()
    elif env_type == "close_short":
        q_table = init_close_short_q_table()
    elif env_type == "close_long":
        q_table = init_close_long_q_table()
    data_folder_name = file_path_dict["data_folder"]
    index_folder_name = file_path_dict["index_folder"]
    result_folder_name = file_path_dict["result_folder"]

    mask_table = init_mask_table()
    config = {
        "data_path": f"./data/{data_folder_name}",
        "env_type": env_type,
        "commission_value": 0.125,
        "index_state_dict_path": f"./data/{index_folder_name}/index_state_dict.npy",
        "state_index_mapping_path": f"./data/{index_folder_name}/state_index_mapping.npy",
        "start_index_path": f"./data/{result_folder_name}",
        "long_table_path": f"./data/{result_folder_name}/sequential/close_long/q_table.csv",
        "short_table_path": f"./data/{result_folder_name}/sequential/close_short/q_table.csv",
        "rank": rank,
        "pred_using": pred_using,
    }

    mc_config = {
        "gamma": 0.99,
        "sample_env_num": 30,
        "mask_table": mask_table,
    }
    return config, mc_config, q_table


def test_sequential(file_path_dict, rank, pred_using):
    result_folder_name = file_path_dict["result_folder"]
    close_long_config, close_long_mc_config, close_long_q_table = get_config(
        "close_long", file_path_dict, rank, pred_using
    )
    open_long_config, open_long_mc_config, open_long_q_table = get_config(
        "open_long", file_path_dict, rank, pred_using
    )
    close_short_config, close_short_mc_config, close_short_q_table = get_config(
        "close_short", file_path_dict, rank, pred_using
    )
    open_short_config, open_short_mc_config, open_short_q_table = get_config(
        "open_short", file_path_dict, rank, pred_using
    )
    close_long_task = Task(
        SplitStateActionEnv,
        "close_long",
        close_long_config,
        close_long_q_table,
        close_long_mc_config,
    )
    open_long_task = Task(
        SplitStateActionEnv,
        "open_long",
        open_long_config,
        open_long_q_table,
        open_long_mc_config,
    )
    close_short_task = Task(
        SplitStateActionEnv,
        "close_short",
        close_short_config,
        close_short_q_table,
        close_short_mc_config,
    )
    open_short_task = Task(
        SplitStateActionEnv,
        "open_short",
        open_short_config,
        open_short_q_table,
        open_short_mc_config,
    )
    open_long_close_long_link = TaskLink([open_long_task, close_long_task])
    close_short_open_short_link = TaskLink([close_short_task, open_short_task])
    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, close_short_open_short_link],
        checkpoint_folder=f"./data/{result_folder_name}/sequential",
    )
    controller.train()


def test_sequential_evaluate(
    file_path_dict, rank, pred_using, test_folder_path, result_saved_folder_path
):
    result_folder_name = file_path_dict["result_folder"]
    close_long_config, close_long_mc_config, _ = get_config(
        "close_long", file_path_dict, rank, pred_using
    )
    open_long_config, open_long_mc_config, _ = get_config(
        "open_long", file_path_dict, rank, pred_using
    )
    close_short_config, close_short_mc_config, _ = get_config(
        "close_short", file_path_dict, rank, pred_using
    )
    open_short_config, open_short_mc_config, _ = get_config(
        "open_short", file_path_dict, rank, pred_using
    )
    close_long_task = Task(
        SplitStateActionEnv,
        "close_long",
        close_long_config,
    )
    open_long_task = Task(
        SplitStateActionEnv,
        "open_long",
        open_long_config,
    )
    close_short_task = Task(
        SplitStateActionEnv,
        "close_short",
        close_short_config,
    )
    open_short_task = Task(
        SplitStateActionEnv,
        "open_short",
        open_short_config,
    )
    open_long_close_long_link = TaskLink([open_long_task, close_long_task])
    close_short_open_short_link = TaskLink([open_short_task, close_short_task])
    checkpoint_folder = f"./data/{result_folder_name}/sequential"
    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, close_short_open_short_link],
        checkpoint_folder=checkpoint_folder,
    )
    controller.evaluate(checkpoint_folder, test_folder_path, result_saved_folder_path)


def compare_figure(q_path, bechmark_path, data_folder):
    import numpy as np

    q_reward_list = np.load(q_path, allow_pickle=True)
    bechmark_reward_list = np.load(bechmark_path, allow_pickle=True)
    update_ratio = sum(q_reward_list) / sum(bechmark_reward_list)
    plt.plot(np.cumsum(q_reward_list))
    plt.plot(np.cumsum(bechmark_reward_list))
    plt.legend(["q", "bechmark"])
    plt.title(f"update_ratio: {update_ratio}")
    plt.savefig(f"./data/{data_folder}_test_result/compare_figure.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="delta4/im_20250916")
    parser.add_argument(
        "--index_folder", type=str, default="delta4/state_index_with_delta4"
    )
    parser.add_argument(
        "--result_folder", type=str, default="delta4/im_20250916_result"
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pred_using", type=bool, default=False)
    parser.add_argument(
        "--test_folder_path", type=str, default="delta4/im_20250916_test"
    )
    parser.add_argument(
        "--result_saved_folder_path", type=str, default="delta4/im_20250916_test_result"
    )
    args = parser.parse_args()
    file_path_dict = {
        "data_folder": args.data_folder,
        "index_folder": args.index_folder,
        "result_folder": args.result_folder,
    }
    test_sequential(file_path_dict, args.rank, args.pred_using)
    # test_folder_path = f"./data/{args.test_folder_path}"
    # result_saved_folder_path = f"./data/{args.result_saved_folder_path}"
    # test_sequential_evaluate(
    #     file_path_dict,
    #     args.rank,
    #     args.pred_using,
    #     test_folder_path,
    #     result_saved_folder_path,
    # )
    # test_mc(env_type, file_path_dict, args.rank, args.pred_using)
    # compare_figure(
    #     f"./data/{args.data_folder}_test_result/q_reward_list.npy",
    #     f"./data/{args.data_folder}_test_result/benchmark_reward_list.npy",
    #     args.data_folder,
    # )
    # q_table = init_close_long_q_table()
    # q_table.to_csv("./table/init_q_open_close.csv")
