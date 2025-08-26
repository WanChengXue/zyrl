import os
import numpy as np
import pandas as pd
from typing import Any


class PreprocessCalculator:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._load_data()

    def _load_data(self) -> None:
        file_list = os.listdir(self._config["data_path"])
        data_list = []
        for file_name in file_list:
            if file_name.endswith(".csv"):
                self._training_data = pd.read_csv(
                    os.path.join(self._config["data_path"], file_name)
                )
                data_list.append(self._training_data["Pred"].values)
        self._data = np.concatenate(data_list)

    def cal_percentile_dict(self) -> None:
        percentile_dict = {}
        state_table = {}
        index_state_dict = {}
        sorted_data = np.sort(self._data)
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
            percentile_value = np.percentile(sorted_data, percentile)
            percentile_dict[f"{100-percentile}%"] = percentile_value
            state_table[percentile_value] = len(percentile_list) - index
            index_state_dict[len(percentile_list) - index] = percentile_value

        np.save(self._config["percentile_dict_path"], percentile_dict)
        np.save(self._config["state_table_path"], state_table)
        np.save(self._config["index_state_dict_path"], index_state_dict)


def init_q_table():
    holding_list = [1, 0, -1]
    state_table_index_list = [i + 1 for i in range(22)]
    action_list = [i for i in range(13)]
    total_state = [
        (holding, state_index)
        for holding in holding_list
        for state_index in state_table_index_list
    ]
    q_table = pd.DataFrame(index=total_state, columns=action_list)
    q_table.fillna(0.0, inplace=True)
    pass


if __name__ == "__main__":
    config = {
        "data_path": "/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/split_state_action_data",
        "percentile_dict_path": "/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/percentile_dict.npy",
        "state_table_path": "/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/state_table.npy",
        "index_state_dict_path": "/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/index_state_dict.npy",
    }
    preprocess_calculator = PreprocessCalculator(config)
    preprocess_calculator.cal_percentile_dict()
