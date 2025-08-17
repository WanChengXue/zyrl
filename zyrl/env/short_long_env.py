from typing import Any, SupportsFloat
import gymnasium as gym
import pandas as pd
import os
import numpy as np
import random
from zyrl.utils.table_utils import load_dataframe
from zyrl.env.reward import ShortLongReward


class ShortLongEnv(gym.Env):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._training_data_path = config.get("training_data_path", None)
        self._training_market_data_path = config.get("training_market_data_path", None)
        self._test_data_path = config.get("test_data_path", None)
        self._test_market_data_path = config.get("test_market_data_path", None)
        self._current_holding = config.get("init_holding", 0)
        self._commission_value = config.get("commission_value", 0.12)
        self._util_termination = config.get("util_termination", False)
        self._percentile_dict = config["percentile_dict"]
        self._state_table = config["state_table"]
        self._index_state_dict = config["index_state_dict"]
        self.observation_space = gym.spaces.Dict(
            {
                "forward_value_index": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "holding": gym.spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )
        self.action_space = gym.spaces.Discrete(3)
        self._start_index = 0
        self._reward_function = ShortLongReward(self._commission_value)

    def _load_data(
        self, data_path: str, market_data_path: str, file_name: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists(data_path):
            file_list = os.listdir(data_path)
            if file_name is None:
                file_name = random.choice(file_list)
            self._current_file = file_name
            data = load_dataframe(os.path.join(data_path, file_name))
            market_data = load_dataframe(
                os.path.join(market_data_path, file_name.replace(".csv", ".h5"))
            )
            return data, market_data
        else:
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

    def _get_current_state_data(self, index: int) -> dict[str, np.ndarray]:
        predict_value_row = self._training_data.iloc[index]
        predict_value = float(predict_value_row["FW_label"])
        predict_value_index = self._get_state_table_index(predict_value)
        holding = np.array([self._current_holding])
        return {
            "forward_value_index": np.array([predict_value_index]),
            "holding": holding,
        }, predict_value

    def _get_market_data(self, ts_str: str) -> dict[str, np.ndarray]:
        least_ts_str_list = np.where(
            ts_str <= self._training_market_data["localtimeStr"].values
        )
        if len(least_ts_str_list[0]) > 0:
            least_index = int(least_ts_str_list[0][0])
        else:
            least_index = len(self._training_market_data) - 1
        price_info = self._get_price_info(least_index)
        return price_info

    def _get_price_info(self, index: int) -> dict[str, np.ndarray]:
        current_AP = self._training_market_data.iloc[index]["AskPrice1"]
        current_BP = self._training_market_data.iloc[index]["BidPrice1"]
        current_MP = 0.5 * (current_AP + current_BP)
        if index == len(self._training_market_data) - 1:
            next_index = index
        else:
            next_index = index + 1
        next_AP = self._training_market_data.iloc[next_index]["AskPrice1"]
        next_BP = self._training_market_data.iloc[next_index]["BidPrice1"]
        next_MP = 0.5 * (next_AP + next_BP)
        return {
            "current_AP": np.array([current_AP]),
            "current_MP": np.array([current_MP]),
            "current_BP": np.array([current_BP]),
            "next_AP": np.array([next_AP]),
            "next_MP": np.array([next_MP]),
            "next_BP": np.array([next_BP]),
        }

    def reset(self, option: dict | None = None):
        if option is not None:
            file_name = option.get("file_name", None)
            start_index = option.get("start_index", 0)
        else:
            file_name = self._config["file_name"]
            start_index = 0
        if self._training_data_path is not None:
            self._training_data, self._training_market_data = self._load_data(
                self._training_data_path, self._training_market_data_path, file_name
            )
            self._current_index = start_index
            self._total_index = len(self._training_data)
            self._current_holding = self._config.get("init_holding", 0)
            state, predict_value = self._get_current_state_data(self._current_index)
            info = {
                "current_holding": self._current_holding,
                "current_index": self._current_index,
                "predict_value": predict_value,
            }
            info.update(
                {
                    "state_table": self._state_table,
                    "percentile_dict": self._percentile_dict,
                }
            )
            return state, info
        else:
            raise ValueError("No data path provided")

    def _get_current_ts(self, index: int) -> str:
        return self._training_data.iloc[index].name

    def step(
        self, action: np.ndarray | int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action_op = self._convert_action_to_action_op(action)
        current_ts_str = self._get_current_ts(self._current_index)
        price_info = self._get_market_data(current_ts_str)
        reward, next_holding = self._reward_function(
            price_info, action_op, self._current_holding
        )
        done = self._done(next_holding, self._util_termination)
        self._current_holding = next_holding
        self._current_index += 1
        state, predict_value = self._get_current_state_data(self._current_index)
        info = {
            "current_ts_str": current_ts_str,
            "current_index": self._current_index,
            "current_holding": self._current_holding,
            "predict_value": predict_value,
        }
        info.update(price_info)
        return state, reward, done, False, info

    def _done(self, next_holding: int, util_termination: bool = False) -> bool:
        if util_termination:
            if self._current_index >= len(self._training_data) - 2:
                return True
        else:
            if self._current_index >= len(self._training_data) - 2:
                return True
            if next_holding == 0:
                return True
            else:
                return False

    def _convert_action_to_action_op(self, action: np.ndarray | int) -> str:
        if action == -1:
            if self._current_holding == -1:
                return "Keep"
            else:
                return "OpenShort"

        elif action == 0:
            if self._current_holding == 0:
                return "Keep"
            elif self._current_holding == -1:
                return "CloseShort"
            else:
                return "CloseLong"

        else:
            if self._current_holding == 1:
                return "Keep"
            else:
                return "OpenLong"

    def get_index_state_dict(self) -> dict[int, float]:
        return self._state_table

    def get_percentile_dict(self) -> dict[str, float]:
        return self._percentile_dict

    def _get_state_table_index(self, predict_value: float) -> int:
        for key in sorted(self._state_table.keys(), reverse=True):
            if predict_value >= key:
                return self._state_table[key]
        return len(self._state_table) + 1

    def get_state_index(self, state: dict[str, np.ndarray]) -> int:
        return state["forward_value_index"].item()

    @classmethod
    def get_start_index_list(
        cls, file_path: str, state_index: int, index_state_dict: dict[int, float]
    ):
        start_index_list = []
        loaded_data = load_dataframe(file_path)
        for index, value in enumerate(loaded_data["FW_label"].values):
            if state_index == 22:
                if index_state_dict[21] > value:
                    start_index_list.append(index)
            else:
                if (
                    index_state_dict[state_index] <= value
                    and index <= len(loaded_data) - 10
                ):
                    start_index_list.append(index)
        if len(start_index_list) >= 100:
            random_start_index_list = random.sample(start_index_list, 100)
            return random_start_index_list
        else:
            return start_index_list
