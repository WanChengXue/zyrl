# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zyrl.env.open_close_env.env import SplitStateActionEnv
from zyrl.sequential.task import Task
from zyrl.sequential.edge import TaskLink
from zyrl.sequential.controller import Controller
from tqdm import tqdm


# %%
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
                if delta1 > 0 and delta2 != 0:
                    mask_table.at[index, 2] = True
                if delta1 > 0 and delta2 > 0 and delta3 != 0:
                    mask_table.at[index, 3] = True
                index += 1
    mask_table.index.name = "state_index"
    mask_table.columns.name = "action"
    return mask_table


# %%
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


def generate_training_data():
    data_list = []
    label_list = []
    mask_list = []
    single_interval_point_number = 5
    delta_interval_list = [
        (0, 0),
        (0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1),
    ]
    for delta1 in tqdm(delta_interval_list):
        for delta2 in delta_interval_list:
            for delta3 in delta_interval_list:
                sub_data_list, sub_label_list, sub_mask_list = (
                    generate_single_interval_data(
                        delta1, delta2, delta3, single_interval_point_number
                    )
                )
                data_list.extend(sub_data_list)
                label_list.extend(sub_label_list)
                mask_list.extend(sub_mask_list)
    return np.array(data_list), np.array(label_list), np.array(mask_list)


def generate_single_interval_data(
    delta1_interval, delta2_interval, delta3_interval, number_of_point
):
    sub_data_list = []
    sub_label_list = []
    sub_mask_list = []

    def _generate_delta_list(delta_interval, number_of_point):
        if delta_interval[0] == 0 and delta_interval[1] == 0:
            delta_list = [0 for _ in range(number_of_point)]
        else:
            step_size = (delta_interval[1] - delta_interval[0]) / number_of_point
            delta_list = [
                delta_interval[0] + (1 + i) * step_size for i in range(number_of_point)
            ]
        return delta_list

    delta1_list = _generate_delta_list(delta1_interval, number_of_point)
    delta2_list = _generate_delta_list(delta2_interval, number_of_point)
    delta3_list = _generate_delta_list(delta3_interval, number_of_point)
    for delta1 in delta1_list:
        for delta2 in delta2_list:
            for delta3 in delta3_list:
                mask_list, label_list = _get_label_and_mask(delta1, delta2, delta3)
                sub_data_list.append((delta1, delta2, delta3))
                sub_label_list.append(label_list)
                sub_mask_list.append(mask_list)
    return sub_data_list, sub_label_list, sub_mask_list


def _get_label_and_mask(delta1, delta2, delta3):
    mask_list = [True, False, False, False]
    label_list = [0, -10, -10, -10]
    if delta1 != 0:
        mask_list[1] = True
        label_list[1] = 0
    if delta2 != 0 and delta1 != 0:
        mask_list[2] = True
        label_list[2] = 2
    if delta3 != 0 and delta2 != 0 and delta1 != 0:
        mask_list[3] = True
        label_list[3] = 0
    return mask_list, label_list


data_list, label_list, mask_list = generate_training_data()


# %%
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
        "data_path": data_folder_name,
        "env_type": env_type,
        "commission_value": 0.125,
        "index_state_dict_path": f"{index_folder_name}/index_state_dict.npy",
        "state_index_mapping_path": f"{index_folder_name}/state_index_mapping.npy",
        "start_index_path": f"{result_folder_name}",
        "rank": rank,
        "pred_using": pred_using,
    }

    mc_config = {
        "gamma": 0.99,
        "sample_env_num": 300,  # 基础采样数量，会根据收敛情况动态调整
        "mask_table": mask_table,
        # 新增收敛控制参数
        "learning_rate": 0.1,  # 初始学习率
        "min_learning_rate": 0.01,  # 最小学习率
        "learning_rate_decay": 0.98,  # 学习率衰减因子（更大，因为MC估计很准确）
        "convergence_window": 10,  # 收敛窗口大小
        "patience": 10,  # 早停耐心值
        # 使用窗口滑动计算
        "use_sliding_window": True,
        # 使用最大误差收敛
        "use_max_error": True,
    }

    rl_config = {
        "pretrained_config": {
            "trained_data": np.array(data_list),
            "label": np.array(label_list),
            "mask": np.array(mask_list),
            "batch_size": 2048,
            "learning_rate": 0.001,
            "device": "cpu",
        },
        "pretrained": True,
        "algo": "DQN",
        "device": "cpu",
        "batch_size": 128,
        "lr": 0.001,
        "epochs": 100,
        "gamma": 0.99,
        "epsilon": 0.1,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
    }
    return config, mc_config, q_table, rl_config


# %%
def test_sequential(file_path_dict, rank, pred_using):
    result_folder_name = file_path_dict["result_folder"]
    (
        close_long_config,
        close_long_mc_config,
        close_long_q_table,
        close_long_rl_config,
    ) = get_config("close_long", file_path_dict, rank, pred_using)
    open_long_config, open_long_mc_config, open_long_q_table, open_long_rl_config = (
        get_config("open_long", file_path_dict, rank, pred_using)
    )
    (
        close_short_config,
        close_short_mc_config,
        close_short_q_table,
        close_short_rl_config,
    ) = get_config("close_short", file_path_dict, rank, pred_using)
    (
        open_short_config,
        open_short_mc_config,
        open_short_q_table,
        open_short_rl_config,
    ) = get_config("open_short", file_path_dict, rank, pred_using)
    close_long_task = Task(
        SplitStateActionEnv,
        "close_long",
        close_long_config,
        close_long_q_table,
        close_long_mc_config,
        train_algo="RL",
        rl_config=close_long_rl_config,
    )
    open_long_task = Task(
        SplitStateActionEnv,
        "open_long",
        open_long_config,
        open_long_q_table,
        open_long_mc_config,
        train_algo="RL",
        rl_config=open_long_rl_config,
    )
    close_short_task = Task(
        SplitStateActionEnv,
        "close_short",
        close_short_config,
        close_short_q_table,
        close_short_mc_config,
        train_algo="RL",
        rl_config=close_short_rl_config,
    )
    open_short_task = Task(
        SplitStateActionEnv,
        "open_short",
        open_short_config,
        open_short_q_table,
        open_short_mc_config,
        train_algo="RL",
        rl_config=open_short_rl_config,
    )
    open_long_close_long_link = TaskLink([open_long_task, close_long_task])
    open_short_close_short_link = TaskLink([open_short_task, close_short_task])
    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, open_short_close_short_link],
        checkpoint_folder=f"{result_folder_name}/sequential_max_stable_action",
    )
    controller.train()


trained_data_path = "./data/delta3/im_20251101"
start_ts_saved_root_path = "./data/delta3/im_20251101_delta3_result"
mapping_saved_root_path = "./data/delta3/state_index_with_delta3"
model_saved_root_path = "./data/delta3/model"
rank = 3
pred_using = False


# %%
# 开始训练
file_path_dict = {
    "data_folder": trained_data_path,
    "index_folder": mapping_saved_root_path,
    "start_ts_saved_root_path": start_ts_saved_root_path,
    "result_folder": model_saved_root_path,
}
test_sequential(file_path_dict, rank, pred_using)

# %%
# file_path_dict


table_folder_name = f"{model_saved_root_path}/sequential_max_stable_action"
test_folder_name = "./data/delta3/im_20251101_test"
saved_folder_name = "./data/delta3/im_20251101_test_delta3_result"


# # %%
# # 并表处理
def test_sequential_evaluate(
    file_path_dict, rank, pred_using, test_folder_path, result_saved_folder_path
):
    result_folder_name = file_path_dict["result_folder"]
    train_algo = "RL"
    (
        close_long_config,
        close_long_mc_config,
        close_long_q_table,
        close_long_rl_config,
    ) = get_config("close_long", file_path_dict, rank, pred_using)
    open_long_config, open_long_mc_config, open_long_q_table, open_long_rl_config = (
        get_config("open_long", file_path_dict, rank, pred_using)
    )
    (
        close_short_config,
        close_short_mc_config,
        close_short_q_table,
        close_short_rl_config,
    ) = get_config("close_short", file_path_dict, rank, pred_using)
    (
        open_short_config,
        open_short_mc_config,
        open_short_q_table,
        open_short_rl_config,
    ) = get_config("open_short", file_path_dict, rank, pred_using)
    close_long_task = Task(
        SplitStateActionEnv,
        "close_long",
        close_long_config,
        close_long_q_table,
        close_long_mc_config,
        train_algo=train_algo,
        rl_config=close_long_rl_config,
    )
    open_long_task = Task(
        SplitStateActionEnv,
        "open_long",
        open_long_config,
        open_long_q_table,
        open_long_mc_config,
        train_algo=train_algo,
        rl_config=open_long_rl_config,
    )
    close_short_task = Task(
        SplitStateActionEnv,
        "close_short",
        close_short_config,
        close_short_q_table,
        close_short_mc_config,
        train_algo=train_algo,
        rl_config=close_short_rl_config,
    )
    open_short_task = Task(
        SplitStateActionEnv,
        "open_short",
        open_short_config,
        open_short_q_table,
        open_short_mc_config,
        train_algo=train_algo,
        rl_config=open_short_rl_config,
    )
    open_long_close_long_link = TaskLink([open_long_task, close_long_task])
    open_short_close_short_link = TaskLink([open_short_task, close_short_task])
    checkpoint_folder = f"{result_folder_name}/sequential_max_stable_action"
    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, open_short_close_short_link],
        checkpoint_folder=checkpoint_folder,
    )

    controller.evaluate(checkpoint_folder, test_folder_path, result_saved_folder_path)


# %%
test_folder_path = test_folder_name
result_saved_folder_path = saved_folder_name
test_sequential_evaluate(
    file_path_dict, rank, pred_using, test_folder_path, result_saved_folder_path
)


# # %%
# # 跑一个baseline的结果
from zyrl.tools.backtest import Backtest

# # %%
backtest_config = {
    "env_config": {
        "training_data_path": test_folder_name,
        "commission_value": 0.125,
        "index_state_dict_path": f"{mapping_saved_root_path}/index_state_dict.npy",
        "state_index_mapping_path": f"{mapping_saved_root_path}/state_index_mapping.npy",
        "rank": rank,
        "pred_using": pred_using,
    },
    "use_benchmark": True,
    "use_rl": True,
    "action_table_path": f"{result_saved_folder_path}",
}
# backtest = Backtest(backtest_config)
# reward_list = backtest.run()
# cum_reward_list = np.cumsum(reward_list)
import os

if not os.path.exists(saved_folder_name):
    os.makedirs(saved_folder_name)

npy_name = "bechmark_reward_list.npy"
png_name = "cusum_reward_probenv_benchmark_test.png"
benchmark_saved_path = f"{saved_folder_name}/{npy_name}"
# np.save(benchmark_saved_path, reward_list)

# %%
# 使用q表跑回测结果
backtest_config["use_benchmark"] = False
backtest = Backtest(backtest_config)
reward_list = backtest.run()
cum_reward_list = np.cumsum(reward_list)
import os

if not os.path.exists(saved_folder_name):
    os.makedirs(saved_folder_name)

npy_name = "q_reward_list.npy"
png_name = "cusum_reward_probenv_q_test.png"
q_saved_path = f"{saved_folder_name}/{npy_name}"
np.save(q_saved_path, reward_list)

# %%
# 评估，将test结果和benchmark结果进行对比
import numpy as np

q_reward_list = np.load(q_saved_path, allow_pickle=True)
bechmark_reward_list = np.load(benchmark_saved_path, allow_pickle=True)
update_ratio = sum(q_reward_list) / sum(bechmark_reward_list)
plt.plot(np.cumsum(q_reward_list))
plt.plot(np.cumsum(bechmark_reward_list))
plt.legend(["q", "bechmark"])
plt.title(f"update_ratio: {update_ratio}")
plt.savefig(f"{saved_folder_name}/compare_figure.png")
plt.close()
