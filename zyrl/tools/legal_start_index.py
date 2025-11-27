import os
from typing import Any
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from zyrl.utils.table_utils import load_dataframe


def _get_state(
    probs: tuple[float, float, float] | tuple[float, float, float, float],
    rank: int = 3,
    pred_value=None,
) -> int:
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

    region_dict = {
        0: (-100, -10),
        1: (-10, -8),
        2: (-8, -6),
        3: (-6, -4),
        4: (-4, -2),
        5: (-2, 0),
        6: (0, 2),
        7: (2, 4),
        8: (4, 6),
        9: (6, 8),
        10: (8, 10),
        11: (10, 100),
    }
    region_interval = None
    if pred_value is not None:
        for region_index, region in region_dict.items():
            left_value, right_value = region
            if pred_value >= left_value and pred_value < right_value:
                region_interval = region
                break

    if rank == 3:
        first_prob, second_prob, third_prob = probs
        first_region = get_prob_region(first_prob)
        second_region = get_prob_region(second_prob)
        third_region = get_prob_region(third_prob)
        if pred_value is not None:
            return (first_region, second_region, third_region, region_interval)
        else:
            return (first_region, second_region, third_region)
    elif rank == 4:
        first_prob, second_prob, third_prob, fourth_prob = probs
        first_region = get_prob_region(first_prob)
        second_region = get_prob_region(second_prob)
        third_region = get_prob_region(third_prob)
        fourth_region = get_prob_region(fourth_prob)
        if pred_value is not None:
            return (
                first_region,
                second_region,
                third_region,
                fourth_region,
                region_interval,
            )
        else:
            return (first_region, second_region, third_region, fourth_region)
    elif rank == 5:
        first_prob, second_prob, third_prob, fourth_prob, fifth_prob = probs
        first_region = get_prob_region(first_prob)
        second_region = get_prob_region(second_prob)
        third_region = get_prob_region(third_prob)
        fourth_region = get_prob_region(fourth_prob)
        fifth_region = get_prob_region(fifth_prob)
        if pred_value is not None:
            return (
                first_region,
                second_region,
                third_region,
                fourth_region,
                fifth_region,
                region_interval,
            )
        else:
            return (
                first_region,
                second_region,
                third_region,
                fourth_region,
                fifth_region,
            )


def get_single_file_start_index(
    file_name: str,
    state_index_mapping: dict[
        int,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        | tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
        | tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ],
    rank: int = 4,
    pred_using: bool = False,
):
    data = load_dataframe(file_name)
    long_dict = defaultdict(list)
    short_dict = defaultdict(list)
    for index in range(len(data) - 20):
        row_data = data.iloc[index]
        if rank == 3:
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
        elif rank == 4:
            long_probs = (
                row_data["TradeRate_NF_Long_Delta1"],
                row_data["TradeRate_NF_Long_Delta2"],
                row_data["TradeRate_NF_Long_Delta3"],
                row_data["TradeRate_NF_Long_Delta4"],
            )
            short_probs = (
                row_data["TradeRate_NF_Short_Delta1"],
                row_data["TradeRate_NF_Short_Delta2"],
                row_data["TradeRate_NF_Short_Delta3"],
                row_data["TradeRate_NF_Short_Delta4"],
            )
        elif rank == 5:
            long_probs = (
                row_data["TradeRate_NF_Long_Delta1"],
                row_data["TradeRate_NF_Long_Delta2"],
                row_data["TradeRate_NF_Long_Delta3"],
                row_data["TradeRate_NF_Long_Delta4"],
                row_data["TradeRate_NF_Long_Delta5"],
            )
            short_probs = (
                row_data["TradeRate_NF_Short_Delta1"],
                row_data["TradeRate_NF_Short_Delta2"],
                row_data["TradeRate_NF_Short_Delta3"],
                row_data["TradeRate_NF_Short_Delta4"],
                row_data["TradeRate_NF_Short_Delta5"],
            )
        pred_value = row_data["Pred"] * row_data["Std"] if pred_using else None
        long_state = _get_state(long_probs, rank, pred_value)
        short_state = _get_state(short_probs, rank, pred_value)
        long_state_index = state_index_mapping[long_state]
        short_state_index = state_index_mapping[short_state]
        long_dict[long_state_index].append(index)
        short_dict[short_state_index].append(index)
        for key in long_dict:
            assert len(long_dict[key]) > 0

        for key in short_dict:
            assert len(short_dict[key]) > 0
    return [long_dict, short_dict]


def get_legal_start_index(config: dict[str, Any]):
    file_name_list = os.listdir(config["data_path"])
    saved_path = config["saved_path"]
    state_index_mapping = config["state_index_mapping"]
    rank = config["rank"]
    for file_name in tqdm(file_name_list):
        file_path = os.path.join(config["data_path"], file_name)
        long_dict, short_dict = get_single_file_start_index(
            file_path, state_index_mapping, rank, config["pred_using"]
        )
        np.save(f"{saved_path}/{file_name}_long_dict.npy", long_dict)
        np.save(f"{saved_path}/{file_name}_short_dict.npy", short_dict)


if __name__ == "__main__":
    data_folder_name = "delta3/im_20250917_old"
    pred_using = False
    index_folder_name = "delta3/state_index_with_delta3"
    rank = 3
    result_start_folder_name = "delta3/im_20250917_old_result"
    state_table = np.load(
        f"./data/{index_folder_name}/state_index_mapping.npy",
        allow_pickle=True,
    ).item()
    saved_root_path = f"./data/{result_start_folder_name}"
    import os

    if not os.path.exists(saved_root_path):
        os.makedirs(saved_root_path, exist_ok=True)
    config = {
        "data_path": f"./data/{data_folder_name}",
        "saved_path": saved_root_path,
        "state_index_mapping": state_table,
        "rank": rank,
        "pred_using": pred_using,
    }
    get_legal_start_index(config)
