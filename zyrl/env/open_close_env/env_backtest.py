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


class SplitStateActionEnvBacktest(gym.Env):
    """Multi-action short-long trading environment class.

    This class provides a trading environment that supports multiple action types
    for short-long trading strategies with gymnasium interface.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._data_path = config["training_data_path"]
        self._rank = config.get("rank", 4)
        self._pred_using = config.get("pred_using", False)
        self._holding = 0
        self._start_index = 0
        self._current_file = None
        self._training_data = None
        self._current_index = 0
        self._total_index = 0
        self._index_state_dict = None
        self._state_index_mapping = None

        self._trading_config = {
            "current_holding": self._holding,
            "commission_value": config.get("commission_value", 0.12),
            "util_termination": config.get("util_termination", False),
        }
        self._load_state_index()
        if self._pred_using:
            if self._rank == 3:
                self.observation_space = gym.spaces.Dict(
                    {
                        "long": gym.spaces.Dict(
                            {
                                "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "pred": gym.spaces.Box(low=-20, high=20, shape=(1,)),
                            }
                        ),
                        "short": gym.spaces.Dict(
                            {
                                "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "pred": gym.spaces.Box(low=-20, high=20, shape=(1,)),
                            }
                        ),
                    }
                )
                self.action_space = gym.spaces.Dict(
                    {"long": gym.spaces.Discrete(4), "short": gym.spaces.Discrete(4)}
                )

            if self._rank == 4:
                self.observation_space = gym.spaces.Dict(
                    {
                        "long": gym.spaces.Dict(
                            {
                                "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta4": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "pred": gym.spaces.Box(low=-20, high=20, shape=(1,)),
                            }
                        ),
                        "short": gym.spaces.Dict(
                            {
                                "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "delta4": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                "pred": gym.spaces.Box(low=-20, high=20, shape=(1,)),
                            }
                        ),
                    }
                )
                self.action_space = gym.spaces.Dict(
                    {"long": gym.spaces.Discrete(5), "short": gym.spaces.Discrete(5)}
                )
        else:
            if self._rank == 3:
                self.observation_space = gym.spaces.Dict(
                    {
                        "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                    }
                )
                self.action_space = gym.spaces.Discrete(4)
            if self._rank == 4:
                self.observation_space = gym.spaces.Dict(
                    {
                        "delta1": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "delta2": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "delta3": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "delta4": gym.spaces.Box(low=0, high=1, shape=(1,)),
                    }
                )
                self.action_space = gym.spaces.Discrete(5)
        self._reward_function = MultiActionShortLongReward(
            self._trading_config["commission_value"]
        )

    def _load_state_index(self):
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

    def _get_current_state_data(
        self, predict_value_row: pd.Series, env_type: str
    ) -> dict[str, np.ndarray]:
        if env_type == "open_long":
            expect_long_delta_1 = float(predict_value_row["TradeRate_NF_Long_Delta1"])
            expect_long_delta_2 = float(predict_value_row["TradeRate_NF_Long_Delta2"])
            expect_long_delta_3 = float(predict_value_row["TradeRate_NF_Long_Delta3"])
            state = {
                "delta1": np.array([expect_long_delta_1]),
                "delta2": np.array([expect_long_delta_2]),
                "delta3": np.array([expect_long_delta_3]),
            }
            if self._rank == 4:
                expect_long_delta_4 = float(
                    predict_value_row["TradeRate_NF_Long_Delta4"]
                )
                state.update({"delta4": np.array([expect_long_delta_4])})
            if self._pred_using:
                pred_value = float(predict_value_row["Pred"]) * float(
                    predict_value_row["Std"]
                )
                state.update({"pred": np.array([pred_value])})
        elif env_type == "close_long":
            expect_short_delta_1 = float(predict_value_row["TradeRate_NF_Short_Delta1"])
            expect_short_delta_2 = float(predict_value_row["TradeRate_NF_Short_Delta2"])
            expect_short_delta_3 = float(predict_value_row["TradeRate_NF_Short_Delta3"])
            state = {
                "delta1": np.array([expect_short_delta_1]),
                "delta2": np.array([expect_short_delta_2]),
                "delta3": np.array([expect_short_delta_3]),
            }
            if self._rank == 4:
                expect_short_delta_4 = float(
                    predict_value_row["TradeRate_NF_Short_Delta4"]
                )
                state.update({"delta4": np.array([expect_short_delta_4])})
            if self._pred_using:
                pred_value = float(predict_value_row["Pred"]) * float(
                    predict_value_row["Std"]
                )
                state.update({"pred": np.array([pred_value])})
        elif env_type == "open_short":
            expect_short_delta_1 = float(predict_value_row["TradeRate_NF_Short_Delta1"])
            expect_short_delta_2 = float(predict_value_row["TradeRate_NF_Short_Delta2"])
            expect_short_delta_3 = float(predict_value_row["TradeRate_NF_Short_Delta3"])
            state = {
                "delta1": np.array([expect_short_delta_1]),
                "delta2": np.array([expect_short_delta_2]),
                "delta3": np.array([expect_short_delta_3]),
            }
            if self._rank == 4:
                expect_short_delta_4 = float(
                    predict_value_row["TradeRate_NF_Short_Delta4"]
                )
                state.update({"delta4": np.array([expect_short_delta_4])})
            if self._pred_using:
                pred_value = float(predict_value_row["Pred"]) * float(
                    predict_value_row["Std"]
                )
                state.update({"pred": np.array([pred_value])})
        elif env_type == "close_short":
            expect_long_delta_1 = float(predict_value_row["TradeRate_NF_Long_Delta1"])
            expect_long_delta_2 = float(predict_value_row["TradeRate_NF_Long_Delta2"])
            expect_long_delta_3 = float(predict_value_row["TradeRate_NF_Long_Delta3"])
            state = {
                "delta1": np.array([expect_long_delta_1]),
                "delta2": np.array([expect_long_delta_2]),
                "delta3": np.array([expect_long_delta_3]),
            }
            if self._rank == 4:
                expect_long_delta_4 = float(
                    predict_value_row["TradeRate_NF_Long_Delta4"]
                )
                state.update({"delta4": np.array([expect_long_delta_4])})
            if self._pred_using:
                pred_value = float(predict_value_row["Pred"]) * float(
                    predict_value_row["Std"]
                )
                state.update({"pred": np.array([pred_value])})
        else:
            raise ValueError(f"Invalid env type: {env_type}")

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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] = {}
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        file_name = options["file_name"]
        start_index = options.get("start_index", 0)
        self._training_data = self._load_data(self._data_path, file_name)
        self._current_index = start_index
        self._total_index = len(self._training_data)
        trading_data = self._training_data.iloc[self._current_index]
        long_state = self._get_current_state_data(trading_data, "open_long")
        short_state = self._get_current_state_data(trading_data, "open_short")
        state = {"long": long_state, "short": short_state}
        info = {
            "current_holding": self._trading_config["current_holding"],
            "current_index": self._current_index,
        }
        return state, info

    def _cal_reward(
        self,
        action: int,
        env_type: str,
        trading_data: pd.Series,
        next_trading_data: pd.Series,
    ) -> float:
        action_op, action_index = self._convert_action_to_action_op(
            action,
            env_type,
            self._trading_config["current_holding"],
            trading_data,
        )
        price_info = self._get_price_info(
            trading_data,
            next_trading_data,
            action_op,
            action_index,
        )
        reward, next_holding = self._reward_function(
            price_info, action_op, self._trading_config["current_holding"]
        )
        if action_op == "Keep":
            assert next_holding == self._trading_config["current_holding"]
        return reward, next_holding, price_info, action_op

    def step(
        self, action: tuple[np.ndarray | int, np.ndarray | int]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        long_action, short_action = action
        if self._holding == 0:
            long_reward, next_long_holding, long_price_info, long_action_op = (
                self._cal_reward(
                    long_action,
                    "open_long",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            short_reward, next_short_holding, short_price_info, short_action_op = (
                self._cal_reward(
                    short_action,
                    "open_short",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            reward = long_reward + short_reward
            new_holding = next_long_holding + next_short_holding
            price_info = {
                "long": long_price_info,
                "short": short_price_info,
            }
            action_op = {
                "long": long_action_op,
                "short": short_action_op,
            }
        elif self._holding == 1:
            long_reward, next_long_holding, long_price_info, long_action_op = (
                self._cal_reward(
                    long_action,
                    "close_long",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            reward = long_reward
            new_holding = next_long_holding
            price_info = {"long": long_price_info}
            action_op = {"long": long_action_op}
            if long_action_op == "Keep":
                assert new_holding == 1

        elif self._holding == -1:
            short_reward, next_short_holding, short_price_info, short_action_op = (
                self._cal_reward(
                    short_action,
                    "close_short",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            reward = short_reward
            new_holding = next_short_holding
            price_info = {"short": short_price_info}
            action_op = {"short": short_action_op}
            if short_action_op == "Keep":
                assert new_holding == -1

        self._holding = new_holding
        self._trading_config["current_holding"] = new_holding
        done = self._done(self._current_index)
        self._current_index += 1
        # update state
        assert self._holding in [0, 1, -1], f"Invalid holding: {self._holding}"
        if self._holding == 0:
            long_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "open_long"
            )
            short_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "open_short"
            )
        if self._holding == 1:
            long_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "close_long"
            )
            short_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "open_short"
            )
        if self._holding == -1:
            long_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "open_long"
            )
            short_state = self._get_current_state_data(
                self._training_data.iloc[self._current_index], "close_short"
            )

        state = {"long": long_state, "short": short_state}
        info = {
            "current_ts_str": self._training_data.iloc[self._current_index].name,
            "current_index": self._current_index,
            "current_holding": self._holding,
        }
        info.update(
            {
                "price_info": price_info,
                "action_op": action_op,
            }
        )
        if done:
            force_reward = self.force_close()
            reward += force_reward
        return state, reward, done, False, info

    def _done(self, current_index: int) -> bool:
        if current_index >= len(self._training_data) - 5:
            return True
        else:
            return False

    def force_close(self):
        if self._holding == 0:
            return 0
        if self._holding == 1:
            long_reward, next_long_holding, long_price_info, long_action_op = (
                self._cal_reward(
                    2,
                    "close_long",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            return long_reward
        if self._holding == -1:
            short_reward, next_short_holding, short_price_info, short_action_op = (
                self._cal_reward(
                    2,
                    "close_short",
                    self._training_data.iloc[self._current_index],
                    self._training_data.iloc[self._current_index + 1],
                )
            )
            return short_reward

    def _convert_action_to_action_op(
        self,
        action: np.ndarray | int,
        env_type: str,
        current_holding: int,
        trading_data: pd.Series,
    ) -> tuple[str, int]:
        if action == 0:
            return "Keep", 0

        if env_type in ["open_long", "close_short"]:
            trading_prob = trading_data["TradeRate_Long_Delta" + str(action)]
            random_value = random.random()
            if random_value <= trading_prob:
                return (
                    ("OpenLong", action)
                    if current_holding == 0
                    else ("CloseShort", action)
                )
            return ("Keep", 0)

        if env_type in ["open_short", "close_long"]:
            trading_prob = trading_data["TradeRate_Short_Delta" + str(action)]
            random_value = random.random()
            if random_value <= trading_prob:
                return (
                    ("OpenShort", action)
                    if current_holding == 0
                    else ("CloseLong", action)
                )
            return ("Keep", 0)

        raise ValueError(f"Invalid action: {action}")

    def _get_state_table_index(
        self,
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
        if pred_value is not None:
            for region_index, region in region_dict.items():
                left_value, right_value = region
                if pred_value >= left_value and pred_value < right_value:
                    region_interval = region
                    break
        else:
            region_interval = None

        if rank == 3:
            first_prob, second_prob, third_prob = probs
            first_region = get_prob_region(first_prob)
            second_region = get_prob_region(second_prob)
            third_region = get_prob_region(third_prob)
            if pred_value is not None:
                return self._state_index_mapping[
                    (first_region, second_region, third_region, region_interval)
                ]
            else:
                return self._state_index_mapping[
                    (first_region, second_region, third_region)
                ]
        elif rank == 4:
            first_prob, second_prob, third_prob, fourth_prob = probs
            first_region = get_prob_region(first_prob)
            second_region = get_prob_region(second_prob)
            third_region = get_prob_region(third_prob)
            fourth_region = get_prob_region(fourth_prob)
            if pred_value is not None:
                return self._state_index_mapping[
                    (
                        first_region,
                        second_region,
                        third_region,
                        fourth_region,
                        region_interval,
                    )
                ]
            else:
                return self._state_index_mapping[
                    (first_region, second_region, third_region, fourth_region)
                ]
        else:
            raise ValueError(f"Invalid rank: {rank}")

    def get_state_index(
        self, state: dict[str, dict[str, np.ndarray]], rank: int = 3
    ) -> dict[str, int]:
        if rank == 3:
            long_prob_region = (
                state["long"]["delta1"].item(),
                state["long"]["delta2"].item(),
                state["long"]["delta3"].item(),
            )
            long_pred_value = state["long"]["pred"].item() if self._pred_using else None
            long_state_index = self._get_state_table_index(
                long_prob_region, rank, long_pred_value
            )
            short_prob_region = (
                state["short"]["delta1"].item(),
                state["short"]["delta2"].item(),
                state["short"]["delta3"].item(),
            )
            short_pred_value = (
                state["short"]["pred"].item() if self._pred_using else None
            )
            short_state_index = self._get_state_table_index(
                short_prob_region, rank, short_pred_value
            )
            state_index = {
                "long": long_state_index,
                "short": short_state_index,
            }
            return state_index
        elif rank == 4:
            long_prob_region = (
                state["long"]["delta1"].item(),
                state["long"]["delta2"].item(),
                state["long"]["delta3"].item(),
                state["long"]["delta4"].item(),
            )
            long_pred_value = state["long"]["pred"].item() if self._pred_using else None
            long_state_index = self._get_state_table_index(
                long_prob_region, rank, long_pred_value
            )
            short_prob_region = (
                state["short"]["delta1"].item(),
                state["short"]["delta2"].item(),
                state["short"]["delta3"].item(),
                state["short"]["delta4"].item(),
            )
            short_pred_value = (
                state["short"]["pred"].item() if self._pred_using else None
            )
            short_state_index = self._get_state_table_index(
                short_prob_region, rank, short_pred_value
            )
            state_index = {
                "long": long_state_index,
                "short": short_state_index,
            }
            return state_index

    def render(self):
        pass

    def get_best_action(self, state: dict[str, dict[str, np.ndarray]]):
        long_env_delta2 = state["long"]["delta2"].item()
        short_env_delta2 = state["short"]["delta2"].item()

        if self._holding == 0:
            long_action = 2 if long_env_delta2 > 0 else 0
            short_action = 2 if short_env_delta2 > 0 else 0
        if self._holding == 1:
            long_action = 2 if long_env_delta2 > 0 else 0
            short_action = 0
        if self._holding == -1:
            long_action = 0
            short_action = 2 if short_env_delta2 > 0 else 0
        return (long_action, short_action)
