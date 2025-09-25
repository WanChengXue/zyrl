import os
import pandas as pd
import numpy as np
from tqdm import tqdm


class RewriteActionToTable:
    def __init__(self, config: dict):
        self._config = config
        self._rank = config.get("rank", 3)
        self.load_index_state()
        self.load_table()

    def convert_state_to_index(
        self, probs: tuple[float, float, float] | tuple[float, float, float, float]
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

        if self._rank == 3:
            first_prob, second_prob, third_prob = probs
            first_region = get_prob_region(first_prob)
            second_region = get_prob_region(second_prob)
            third_region = get_prob_region(third_prob)
            return self._state_index_mapping[
                (first_region, second_region, third_region)
            ]
        elif self._rank == 4:
            first_prob, second_prob, third_prob, fourth_prob = probs
            first_region = get_prob_region(first_prob)
            second_region = get_prob_region(second_prob)
            third_region = get_prob_region(third_prob)
            fourth_region = get_prob_region(fourth_prob)
            return self._state_index_mapping[
                (first_region, second_region, third_region, fourth_region)
            ]

    def load_index_state(self):
        self._index_state_dict = np.load(
            self._config["index_state_dict_path"], allow_pickle=True
        ).item()
        self._state_index_mapping = np.load(
            self._config["state_index_mapping_path"], allow_pickle=True
        ).item()

    def load_table(self):
        self._open_long_table = pd.read_csv(self._config["open_long_table_path"])
        self._open_short_table = pd.read_csv(self._config["open_short_table_path"])
        self._close_long_table = pd.read_csv(self._config["close_long_table_path"])
        self._close_short_table = pd.read_csv(self._config["close_short_table_path"])

    def rewrite_single_file_data(self, file_name: str):
        data = pd.read_csv(file_name)
        # 添加四列，初始值设为空字符串
        data["open_long"] = ""
        data["open_short"] = ""
        data["close_long"] = ""
        data["close_short"] = ""

        for index, row in tqdm(data.iterrows(), total=len(data)):
            if self._rank == 3:
                long_probs = (
                    row["TradeRate_NF_Long_Delta1"],
                    row["TradeRate_NF_Long_Delta2"],
                    row["TradeRate_NF_Long_Delta3"],
                )
                long_state_index = self.convert_state_to_index(long_probs)
                short_probs = (
                    row["TradeRate_NF_Short_Delta1"],
                    row["TradeRate_NF_Short_Delta2"],
                    row["TradeRate_NF_Short_Delta3"],
                )
                short_state_index = self.convert_state_to_index(short_probs)

            if self._rank == 4:
                long_probs = (
                    row["TradeRate_NF_Long_Delta1"],
                    row["TradeRate_NF_Long_Delta2"],
                    row["TradeRate_NF_Long_Delta3"],
                    row["TradeRate_NF_Long_Delta4"],
                )
                long_state_index = self.convert_state_to_index(long_probs)
                short_probs = (
                    row["TradeRate_NF_Short_Delta1"],
                    row["TradeRate_NF_Short_Delta2"],
                    row["TradeRate_NF_Short_Delta3"],
                    row["TradeRate_NF_Short_Delta4"],
                )
                short_state_index = self.convert_state_to_index(short_probs)

            open_long_table = self._open_long_table.iloc[long_state_index]
            open_short_table = self._open_short_table.iloc[short_state_index]
            close_long_table = self._close_long_table.iloc[short_state_index]
            close_short_table = self._close_short_table.iloc[long_state_index]

            open_long_action = np.argmax(open_long_table.values[1:])
            open_short_action = np.argmax(open_short_table.values[1:])
            close_long_action = np.argmax(close_long_table.values[1:])
            close_short_action = np.argmax(close_short_table.values[1:])

            data.at[index, "open_long"] = (
                open_long_action if open_long_action != 0 else -1
            )
            data.at[index, "open_short"] = (
                open_short_action if open_short_action != 0 else -1
            )
            data.at[index, "close_long"] = (
                close_long_action if close_long_action != 0 else -1
            )
            data.at[index, "close_short"] = (
                close_short_action if close_short_action != 0 else -1
            )

        return data

    def run(self):
        file_list = os.listdir(self._config["data_path"])
        data_list = []
        for file_name in sorted(file_list):
            total_file_path = os.path.join(self._config["data_path"], file_name)
            data = self.rewrite_single_file_data(total_file_path)
            data_list.append(data)

        total_data = pd.concat(data_list)
        self.save_data(total_data)

    def save_data(self, total_data: pd.DataFrame):
        saved_path = self._config["saved_path"]
        total_data.to_csv(saved_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="delta4/im_20250915")
    parser.add_argument(
        "--index_folder", type=str, default="delta4/state_index_with_delta4"
    )
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    config = {
        "data_path": f"./data/{args.data_folder}_test",
        "saved_path": f"./data/{args.data_folder}_test_result/rewrite_action_table.csv",
        "index_state_dict_path": f"./data/{args.index_folder}/index_state_dict.npy",
        "state_index_mapping_path": f"./data/{args.index_folder}/state_index_mapping.npy",
        "open_long_table_path": f"./data/{args.data_folder}_result/open_long_q_table.csv",
        "open_short_table_path": f"./data/{args.data_folder}_result/open_short_q_table.csv",
        "close_long_table_path": f"./data/{args.data_folder}_result/close_long_q_table.csv",
        "close_short_table_path": f"./data/{args.data_folder}_result/close_short_q_table.csv",
        "rank": args.rank,
    }
    rewrite_action_to_table = RewriteActionToTable(config)
    total_data = rewrite_action_to_table.run()
