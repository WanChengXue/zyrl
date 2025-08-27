"""Multi-action short-long trading environment module.

This module provides a trading environment that supports multiple action types
for short-long trading strategies with gymnasium interface.
"""

from typing import Any, SupportsFloat
import random
import os
import numpy as np
import gymnasium as gym
import pandas as pd
from zyrl.env.open_close_env.reward import MultiActionShortLongReward
from zyrl.utils.table_utils import load_dataframe


class SplitStateActionEnv(gym.Env):
    """Multi-action short-long trading environment class.

    This class provides a trading environment that supports multiple action types
    for short-long trading strategies with gymnasium interface.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._data_path = config["data_path"]
        # open_long, close_long, open_short, close_short
        self._env_type = config["env_type"]
        self._trading_config = {
            "current_holding": config.get("init_holding", 0),
            "commission_value": config.get("commission_value", 0.12),
            "util_termination": config.get("util_termination", False),
        }
        self._load_state_index()
        self.observation_space = gym.spaces.Dict(
            {
                "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._reward_function = MultiActionShortLongReward(
            self._trading_config["commission_value"]
        )
        self._holding = 0
        self._start_index = 0
        self._current_file = None
        self._training_data = None
        self._current_index = 0
        self._total_index = 0
        self._state_table = None
        self._index_state_dict = None
        self._state_index_mapping = None

    def _load_state_index(self):
        self._state_table = np.load(
            self._config["state_table_path"], allow_pickle=True
        ).item()
        self._index_state_dict = np.load(
            self._config["index_state_dict_path"], allow_pickle=True
        ).item()
        self._state_index_mapping = np.load(
            self._config["state_index_mapping_path"], allow_pickle=True
        ).item()

    def _load_data(
        self, data_path: str, file_name: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists(data_path):
            file_list = os.listdir(data_path)
            if file_name is None:
                file_name = random.choice(file_list)
            self._current_file = file_name
            data = load_dataframe(os.path.join(data_path, file_name))
            return data

        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    def _get_current_state_data(self, index: int) -> dict[str, np.ndarray]:
        predict_value_row = self._training_data.iloc[index]
        if self._env_type == "open_long":
            expect_long_delta_1 = float(predict_value_row["TradeRate_NF_Long_Delta1"])
            expect_long_delta_2 = float(predict_value_row["TradeRate_NF_Long_Delta2"])
            expect_long_delta_3 = float(predict_value_row["TradeRate_NF_Long_Delta3"])
            state = {
                "delta1": np.array([expect_long_delta_1]),
                "delta2": np.array([expect_long_delta_2]),
                "delta3": np.array([expect_long_delta_3]),
            }
        elif self._env_type == "close_long":
            expect_short_delta_1 = float(predict_value_row["TradeRate_NF_Short_Delta1"])
            expect_short_delta_2 = float(predict_value_row["TradeRate_NF_Short_Delta2"])
            expect_short_delta_3 = float(predict_value_row["TradeRate_NF_Short_Delta3"])
            state = {
                "delta1": np.array([expect_short_delta_1]),
                "delta2": np.array([expect_short_delta_2]),
                "delta3": np.array([expect_short_delta_3]),
            }
        elif self._env_type == "open_short":
            expect_short_delta_1 = float(predict_value_row["TradeRate_NF_Short_Delta1"])
            expect_short_delta_2 = float(predict_value_row["TradeRate_NF_Short_Delta2"])
            expect_short_delta_3 = float(predict_value_row["TradeRate_NF_Short_Delta3"])
            state = {
                "delta1": np.array([expect_short_delta_1]),
                "delta2": np.array([expect_short_delta_2]),
                "delta3": np.array([expect_short_delta_3]),
            }
        elif self._env_type == "close_short":
            expect_long_delta_1 = float(predict_value_row["TradeRate_NF_Long_Delta1"])
            expect_long_delta_2 = float(predict_value_row["TradeRate_NF_Long_Delta2"])
            expect_long_delta_3 = float(predict_value_row["TradeRate_NF_Long_Delta3"])
            state = {
                "delta1": np.array([expect_long_delta_1]),
                "delta2": np.array([expect_long_delta_2]),
                "delta3": np.array([expect_long_delta_3]),
            }
        else:
            raise ValueError(f"Invalid env type: {self._env_type}")

        return state

    def _get_price_info(
        self,
        current_trade_data: pd.Series,
        next_trade_data: pd.Series,
        action_op: str,
        action_index: int = 0,
    ) -> dict[str, np.ndarray]:
        if action_op == "Keep":
            return {
                "current_MP": np.array([current_trade_data["MidPrice"]]),
                "next_MP": np.array([next_trade_data["MidPrice"]]),
            }

        if action_op == "OpenLong":
            return {
                "current_LP": np.array(
                    [current_trade_data["Price_Long_Delta" + str(action_index)]]
                ),
                "next_MP": np.array([next_trade_data["MidPrice"]]),
            }

        if action_op == "CloseLong":
            return {
                "current_SP": np.array(
                    [current_trade_data["Price_Short_Delta" + str(action_index)]]
                ),
                "current_MP": np.array([current_trade_data["MidPrice"]]),
            }

        if action_op == "OpenShort":
            return {
                "current_SP": np.array(
                    [current_trade_data["Price_Short_Delta" + str(action_index)]]
                ),
                "next_MP": np.array([next_trade_data["MidPrice"]]),
            }

        if action_op == "CloseShort":
            return {
                "current_LP": np.array(
                    [current_trade_data["Price_Long_Delta" + str(action_index)]]
                ),
                "current_MP": np.array([current_trade_data["MidPrice"]]),
            }

    def _random_start_index(
        self, init_state_index: int | None = None, seed: int | None = None
    ) -> tuple[str, int]:
        file_list = os.listdir(self._data_path)
        if seed is not None:
            random.seed(seed)
        file_name = random.choice(file_list)
        if init_state_index is None:
            start_index = 0
        else:
            fw_value_index, current_holding = self._state_index_mapping[
                "index_to_state"
            ][init_state_index]
            start_index_list = self.get_start_index_list(
                os.path.join(self._data_path, file_name),
                fw_value_index,
                self._index_state_dict,
            )
            start_index = random.choice(start_index_list)
            self._trading_config["current_holding"] = current_holding
        return file_name, start_index

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if options is not None and "file_name" in options:
            file_name = options["file_name"]
            start_index = options.get("start_index", 0)
        else:
            init_state_index = (
                options.get("init_state_index", None) if options else None
            )
            file_name, start_index = self._random_start_index(init_state_index, seed)

        self._training_data = self._load_data(self._data_path, file_name)
        self._current_index = start_index
        self._total_index = len(self._training_data)
        state, predict_value_index = self._get_current_state_data(self._current_index)
        info = {
            "current_holding": self._trading_config["current_holding"],
            "current_index": self._current_index,
            "predict_value_index": predict_value_index,
        }
        info.update(
            {
                "state_table": self._state_table,
                "percentile_dict": self._percentile_dict,
            }
        )
        return state, info

    def step(
        self, action: np.ndarray | int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action_op, action_index = self._convert_action_to_action_op(
            action,
            self._trading_config["current_holding"],
            self._training_data.iloc[self._current_index],
        )
        price_info = self._get_price_info(
            self._training_data.iloc[self._current_index],
            self._training_data.iloc[self._current_index + 1],
            action_op,
            action_index,
        )
        reward, next_holding = self._reward_function(
            price_info, action_op, self._trading_config["current_holding"]
        )
        done = self._done(next_holding, self._trading_config["util_termination"])
        self._trading_config["current_holding"] = next_holding
        self._current_index += 1
        state, predict_value = self._get_current_state_data(self._current_index)
        info = {
            "current_ts_str": self._training_data.iloc[self._current_index].name,
            "current_index": self._current_index,
            "current_holding": self._trading_config["current_holding"],
            "predict_value": predict_value,
        }
        info.update(price_info)
        return state, reward, done, False, info

    def _done(self, next_holding: int, util_termination: bool = False) -> bool:
        if util_termination:
            if self._current_index >= len(self._training_data) - 5:
                return True
        else:
            if self._current_index >= len(self._training_data) - 5:
                return True
            if next_holding == 0:
                return True
            else:
                return False

    def _convert_action_to_action_op(
        self, action: np.ndarray | int, current_holding: int, trading_data: pd.Series
    ) -> tuple[str, int]:
        if action == 0:
            return "Keep", 0

        # Define action mappings
        long_actions = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        short_actions = {7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5}

        if action in long_actions:
            trading_prob = trading_data[f"TradeRate_Long_Delta{long_actions[action]}"]
            random_value = random.random()
            if random_value <= trading_prob:
                return (
                    ("OpenLong", long_actions[action])
                    if current_holding == 0
                    else ("CloseShort", long_actions[action])
                )
            return ("Keep", 0)

        if action in short_actions:
            trading_prob = trading_data[f"TradeRate_Short_Delta{short_actions[action]}"]
            random_value = random.random()
            if random_value <= trading_prob:
                return (
                    ("OpenShort", short_actions[action])
                    if current_holding == 0
                    else ("CloseLong", short_actions[action])
                )
            return ("Keep", 0)

        raise ValueError(f"Invalid action: {action}")

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
        state_index = self._get_state_table_index(state["forward_value"].item())
        holding = state["holding"].item()
        return self._state_index_mapping["state_to_index"][(state_index, holding)]

    def render(self):
        pass

    @staticmethod
    def get_start_index_list(
        file_path: str, env_type: str, state_table: list[tuple[float, float]]
    ) -> list:

        def region_check(element, left_value, right_value) -> bool:
            if left_value == right_value == element == 0:
                return True

            if element > left_value and element <= right_value:
                return True

            return False

        start_index_list = []
        loaded_data = load_dataframe(file_path)
        for index in range(len(loaded_data) - 100):
            row_data = loaded_data.iloc[index]
            if env_type == "open_long":
                first_elememt = row_data["TradeRate_NF_Long_Delta1"]
                second_elememt = row_data["TradeRate_NF_Long_Delta2"]
                third_elememt = row_data["TradeRate_NF_Long_Delta3"]

            if env_type == "open_short":
                first_elememt = row_data["TradeRate_NF_Short_Delta1"]
                second_elememt = row_data["TradeRate_NF_Short_Delta2"]
                third_elememt = row_data["TradeRate_NF_Short_Delta3"]

            if env_type == "close_long":
                first_elememt = row_data["TradeRate_NF_Short_Delta1"]
                second_elememt = row_data["TradeRate_NF_Short_Delta2"]
                third_elememt = row_data["TradeRate_NF_Short_Delta3"]

            if env_type == "close_short":
                first_elememt = row_data["TradeRate_NF_Long_Delta1"]
                second_elememt = row_data["TradeRate_NF_Long_Delta2"]
                third_elememt = row_data["TradeRate_NF_Long_Delta3"]

            if (
                region_check(first_elememt, state_table[0][0], state_table[0][1])
                and region_check(second_elememt, state_table[0][1], state_table[0][2])
                and region_check(third_elememt, state_table[0][2], state_table[0][3])
            ):
                start_index_list.append(index)

        return start_index_list
