# %%
import pandas as pd
import matplotlib.pyplot as plt
from zyrl.env.open_close_env.env import SplitStateActionEnv
from zyrl.sequential.task import Task
from zyrl.sequential.edge import TaskLink
from zyrl.sequential.controller import Controller


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


# 6nHWPTvhkrVx
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
                # if delta2 > 0:
                #     q_table.at[index, 2] = 1.0
                #     q_table.at[index, 0] = -10.0
                # else:
                #     q_table.at[index, 2] = -10.0
                #     q_table.at[index, 0] = 1.0
                # q_table.at[index, 3] = -10.0
                # q_table.at[index, 1] = -10.0

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
                # if delta2 > 0:
                #     q_table.at[index, 2] = 1.0
                #     q_table.at[index, 0] = -10.0
                # else:
                #     q_table.at[index, 2] = -10.0
                #     q_table.at[index, 0] = 1.0
                # q_table.at[index, 3] = -10.0
                # q_table.at[index, 1] = -10.0
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
                # if delta2 > 0:
                #     q_table.at[index, 2] = 1.0
                #     q_table.at[index, 0] = -10.0
                # else:
                #     q_table.at[index, 2] = -10.0
                #     q_table.at[index, 0] = 1.0
                # q_table.at[index, 3] = -10.0
                # q_table.at[index, 1] = -10.0
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
                # if delta2 > 0:
                #     q_table.at[index, 2] = 1.0
                #     q_table.at[index, 0] = -10.0
                # else:
                #     q_table.at[index, 2] = -10.0
                #     q_table.at[index, 0] = 1.0
                # q_table.at[index, 3] = -10.0
                # q_table.at[index, 1] = -10.0
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
    start_ts_saved_root_path = file_path_dict["start_ts_saved_root_path"]

    mask_table = init_mask_table()
    config = {
        "data_path": data_folder_name,
        "env_type": env_type,
        "commission_value": 0.1284,
        "index_state_dict_path": f"{index_folder_name}/index_state_dict.npy",
        "state_index_mapping_path": f"{index_folder_name}/state_index_mapping.npy",
        "start_index_path": f"{start_ts_saved_root_path}",
        "fixed_commission": True,
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
    return config, mc_config, q_table


# %%
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
    open_short_close_short_link = TaskLink([open_short_task, close_short_task])
    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, open_short_close_short_link],
        checkpoint_folder=result_folder_name,
    )
    controller.train()


# %%
# 计算state和index的mapping
import numpy as np

# 默认是先delta1， delta2， delta3，然后是pred值
from zyrl.tools.state_index_processor import generate_state_table_delta3

index_state_dict, state_index_mapping = generate_state_table_delta3()
import os

mapping_saved_root_path = "./data/delta3/state_index_with_delta3"
if not os.path.exists(mapping_saved_root_path):
    os.makedirs(mapping_saved_root_path, exist_ok=True)
index_state_dict_path = "./data/delta3/state_index_with_delta3/index_state_dict.npy"
state_index_mapping_path = (
    "./data/delta3/state_index_with_delta3/state_index_mapping.npy"
)
np.save(index_state_dict_path, index_state_dict)
np.save(state_index_mapping_path, state_index_mapping)

# %%
# 计算delta3 with pred的所有start点
from zyrl.tools.legal_start_index import get_legal_start_index

start_ts_saved_root_path = "./data/delta3/im_20251101_delta3_result"
if not os.path.exists(start_ts_saved_root_path):
    os.makedirs(start_ts_saved_root_path, exist_ok=True)
trained_data_path = "./data/delta3/im_20251101"
rank = 3
pred_using = False
config = {
    "data_path": trained_data_path,
    "saved_path": start_ts_saved_root_path,
    "state_index_mapping": state_index_mapping,
    "rank": rank,
    "pred_using": pred_using,
}
# get_legal_start_index(config)

# %%
# 开始训练
# table_result_folder = "./data/delta3/model/default_policy_q_table"
table_result_folder = "./data/delta3/model/adjusted_cost_q_table"
file_path_dict = {
    "data_folder": trained_data_path,
    "index_folder": mapping_saved_root_path,
    "start_ts_saved_root_path": start_ts_saved_root_path,
    "result_folder": table_result_folder,
}
test_sequential(file_path_dict, rank, pred_using)


def test_default_policy(file_path_dict, rank, pred_using):
    result_folder_name = file_path_dict["result_folder"]
    if os.path.exists(result_folder_name):
        os.makedirs(result_folder_name, exist_ok=True)
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
    os.makedirs(f"{result_folder_name}/close_long", exist_ok=True)
    os.makedirs(f"{result_folder_name}/open_long", exist_ok=True)
    os.makedirs(f"{result_folder_name}/close_short", exist_ok=True)
    os.makedirs(f"{result_folder_name}/open_short", exist_ok=True)
    close_long_q_table.to_csv(f"{result_folder_name}/close_long/q_table.csv")
    open_long_q_table.to_csv(f"{result_folder_name}/open_long/q_table.csv")
    close_short_q_table.to_csv(f"{result_folder_name}/close_short/q_table.csv")
    open_short_q_table.to_csv(f"{result_folder_name}/open_short/q_table.csv")


# %%
# test_default_policy(file_path_dict, rank, pred_using)
# 跑一个baseline的结果
from zyrl.tools.backtest import Backtest

table_folder_name = table_result_folder
test_folder_name = "./data/delta3/im_20251101_test"
saved_folder_name = "./data/delta3/im_20251127_test_delta3_result"
backtest_config = {
    "q_table_path": {
        "open_long_table_path": f"{table_folder_name}/open_long/q_table.csv",
        "open_short_table_path": f"{table_folder_name}/open_short/q_table.csv",
        "close_long_table_path": f"{table_folder_name}/close_long/q_table.csv",
        "close_short_table_path": f"{table_folder_name}/close_short/q_table.csv",
    },
    "env_config": {
        "training_data_path": test_folder_name,
        "commission_value": 0.1284,
        "index_state_dict_path": f"{mapping_saved_root_path}/index_state_dict.npy",
        "state_index_mapping_path": f"{mapping_saved_root_path}/state_index_mapping.npy",
        "rank": rank,
        "pred_using": pred_using,
        "fixed_commission": True,
    },
    "use_benchmark": True,
}
backtest = Backtest(backtest_config)
reward_list = backtest.run()
cum_reward_list = np.cumsum(reward_list)
import os

if not os.path.exists(saved_folder_name):
    os.makedirs(saved_folder_name)

npy_name = "bechmark_reward_list.npy"
png_name = "cusum_reward_probenv_benchmark_test.png"
benchmark_saved_path = f"{saved_folder_name}/{npy_name}"
np.save(benchmark_saved_path, reward_list)

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


# %%
# 并表处理
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

    controller = Controller(
        task_list=[close_long_task, open_long_task, close_short_task, open_short_task],
        link_list=[open_long_close_long_link, close_short_open_short_link],
        checkpoint_folder=result_folder_name,
    )
    controller.evaluate(result_folder_name, test_folder_path, result_saved_folder_path)


# %%
test_folder_path = test_folder_name
result_saved_folder_path = saved_folder_name
test_sequential_evaluate(
    file_path_dict, rank, pred_using, test_folder_path, result_saved_folder_path
)

# %%
