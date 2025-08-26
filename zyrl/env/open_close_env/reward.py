"""Reward calculation module for short-long trading environment.

This module provides reward calculation functions for various trading actions
including opening/closing long/short positions and keeping positions.
"""

import numpy as np


class MultiActionShortLongReward:
    """Reward calculation class for multi-action short-long trading environment.

    This class provides reward calculation methods for different trading actions
    including opening/closing long/short positions and keeping positions.
    """

    def __init__(self, commission_value) -> None:
        self._commission_value = commission_value

    def open_long_reward(self, price_info: dict, current_holding: int) -> float:
        """Calculate reward for opening long position.

        Args:
            price_info (dict): Price information including current and next market prices.
            current_holding (int): Current holding status (-1 for short, 0 for neutral, 1 for long).

        Returns:
            float: Reward value.

        Raises:
            ValueError: If current holding status is invalid.
        """

        if current_holding == 0:
            # open long
            open_long_value = self._open_long(price_info)
            return open_long_value

        raise ValueError(f"Invalid current holding: {current_holding}")

    def keep_long_reward(self, price_info: dict) -> float:
        """Calculate reward for keeping long position.

        Args:
            price_info (dict): Price information including current and next market prices.

        Returns:
            float: Reward value.
        """
        return self._keep_long(price_info)

    def close_long_reward(self, price_info: dict) -> float:
        """Calculate reward for closing long position.

        Args:
            price_info (dict): Price information including current and next market prices.

        Returns:
            float: Reward value.
        """
        return self._close_long(price_info)

    def open_short_reward(self, price_info: dict, current_holding: int) -> float:
        """Calculate reward for opening short position.

        Args:
            price_info (dict): Price information including current and next market prices.
            current_holding (int): Current holding status (-1 for short, 0 for neutral, 1 for long).

        Returns:
            float: Reward value.
        """
        if current_holding == 0:
            return self._open_short(price_info)

        raise ValueError(f"Invalid current holding: {current_holding}")

    def keep_short_reward(self, price_info: dict) -> float:
        """Calculate reward for keeping short position.

        Args:
            price_info (dict): Price information including current and next market prices.

        Returns:
            float: Reward value.
        """
        return self._keep_short(price_info)

    def close_short_reward(self, price_info: dict) -> float:
        """Calculate reward for closing short position.

        Args:
            price_info (dict): Price information including current and next market prices.

        Returns:
            float: Reward value.
        """
        return self._close_short(price_info)

    def _open_long(self, price_info):
        next_mp = price_info["next_MP"]
        current_lp = price_info["current_LP"]
        return next_mp - current_lp - self._commission_value

    def _keep_long(self, price_info):
        next_mp = price_info["next_MP"]
        current_mp = price_info["current_MP"]
        return next_mp - current_mp

    def _close_long(self, price_info):
        current_sp = price_info["current_SP"]
        current_mp = price_info["current_MP"]
        return current_sp - self._commission_value - current_mp

    def _open_short(self, price_info):
        current_sp = price_info["current_SP"]
        next_mp = price_info["next_MP"]
        return current_sp - self._commission_value - next_mp

    def _keep_short(self, price_info):
        current_mp = price_info["current_MP"]
        next_mp = price_info["next_MP"]
        return current_mp - next_mp

    def _close_short(self, price_info):
        current_mp = price_info["current_MP"]
        current_lp = price_info["current_LP"]
        return current_mp - current_lp - self._commission_value

    def __call__(
        self, price_info: dict, action: str, current_holding: int
    ) -> tuple[np.ndarray, int]:
        action_map = {
            "OpenLong": (lambda: self.open_long_reward(price_info, current_holding), 1),
            "CloseLong": (lambda: self.close_long_reward(price_info), 0),
            "OpenShort": (
                lambda: self.open_short_reward(price_info, current_holding),
                -1,
            ),
            "CloseShort": (lambda: self.close_short_reward(price_info), 0),
        }

        if action in action_map:
            reward_func, new_holding = action_map[action]
            return reward_func(), new_holding

        # Default case for keeping current position
        if current_holding == 0:
            return np.array([0.0]), 0
        if current_holding == -1:
            return self.keep_short_reward(price_info), -1
        return self.keep_long_reward(price_info), 1
