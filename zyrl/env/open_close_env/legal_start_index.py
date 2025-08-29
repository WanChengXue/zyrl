import os
from typing import Any
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from zyrl.utils.table_utils import load_dataframe


def _get_state(probs: tuple[float, float, float]) -> int:
    def get_prob_region(prob: float) -> tuple[float, float]:
        if prob == 0:
            return (0, 0)
        elif prob <= 0.2:
            return (0, 0.2)
        elif prob <= 0.4:
            return (0.2, 0.4)
        elif prob <= 0.6:
            return (0.4, 0.6)
        elif prob <= 0.8:
            return (0.6, 0.8)
        else:
            return (0.8, 1)

    first_prob, second_prob, third_prob = probs
    first_region = get_prob_region(first_prob)
    second_region = get_prob_region(second_prob)
    third_region = get_prob_region(third_prob)
    return (first_region, second_region, third_region)


def get_single_file_start_index(
    file_name: str,
    state_index_mapping: dict[
        int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ],
):
    data = load_dataframe(file_name)
    long_dict = defaultdict(list)
    short_dict = defaultdict(list)
    for index in range(len(data) - 20):
        row_data = data.iloc[index]
        long_probs = (
            row_data["TradeRate_NF_Long_Delta1"],
            row_data["TradeRate_NF_Long_Delta2"],
            row_data["TradeRate_NF_Long_Delta3"],
        )
        short_probs = (
            row_data["TradeRate_NF_Short_Delta1"],
            row_data["TradeRate_NF_Short_Delta2"],
            row_data["TradeRate_NF_Short_Delta3"],
        )
        long_state = _get_state(long_probs)
        short_state = _get_state(short_probs)
        long_state_index = state_index_mapping[long_state]
        short_state_index = state_index_mapping[short_state]
        long_dict[long_state_index].append(index)
        short_dict[short_state_index].append(index)

    return [long_dict, short_dict]


def get_legal_start_index(config: dict[str, Any]):
    file_name_list = os.listdir(config["data_path"])
    saved_path = config["saved_path"]
    state_index_mapping = config["state_index_mapping"]
    for file_name in tqdm(file_name_list):
        file_path = os.path.join(config["data_path"], file_name)
        long_dict, short_dict = get_single_file_start_index(
            file_path, state_index_mapping
        )
        np.save(f"{saved_path}/{file_name}_long_dict.npy", long_dict)
        np.save(f"{saved_path}/{file_name}_short_dict.npy", short_dict)


if __name__ == "__main__":
    state_table = np.load(
        "./data/split_state_action_data_index/state_index_mapping.npy",
        allow_pickle=True,
    ).item()
    saved_root_path = "./data/split_state_action_data_index"
    config = {
        "data_path": "./data/split_state_action_data",
        "saved_path": saved_root_path,
        "state_index_mapping": state_table,
    }
    get_legal_start_index(config)
