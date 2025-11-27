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

    def __init__(
        self, commission_value, multiplier=200, fixed_commission: bool = False
    ) -> None:
        self._fixed_commission = fixed_commission
        self._commission_value = commission_value
        self._rebate_rate = 0.25 * 0.94
        self._send_order_price = 1 / multiplier
        self._cancel_order_price = 1 / multiplier

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
        if self._fixed_commission:
            commission_fee = self._commission_value
        else:
            commission_fee = (
                self._commission_value * current_lp * (1 - self._rebate_rate)
            )
        # add send order price, and add mulipier * commission value * current_lp * (1-rebate_rate)
        return next_mp - current_lp - commission_fee - self._send_order_price

    def _keep_long(self, price_info):
        next_mp = price_info["next_MP"]
        current_mp = price_info["current_MP"]
        return next_mp - current_mp

    def _close_long(self, price_info):
        current_sp = price_info["current_SP"]
        current_mp = price_info["current_MP"]
        if self._fixed_commission:
            commission_fee = self._commission_value
        else:
            commission_fee = (
                self._commission_value * current_sp * (1 - self._rebate_rate)
            )
        return current_sp - current_mp - commission_fee - self._send_order_price

    def _open_short(self, price_info):
        current_sp = price_info["current_SP"]
        next_mp = price_info["next_MP"]
        if self._fixed_commission:
            commission_fee = self._commission_value
        else:
            commission_fee = (
                self._commission_value * current_sp * (1 - self._rebate_rate)
            )
        return current_sp - next_mp - commission_fee - self._send_order_price

    def _keep_short(self, price_info):
        current_mp = price_info["current_MP"]
        next_mp = price_info["next_MP"]
        return current_mp - next_mp

    def _close_short(self, price_info):
        current_mp = price_info["current_MP"]
        current_lp = price_info["current_LP"]
        if self._fixed_commission:
            commission_fee = self._commission_value
        else:
            commission_fee = (
                self._commission_value * current_lp * (1 - self._rebate_rate)
            )
        return current_mp - current_lp - commission_fee - self._send_order_price

    def __call__(
        self,
        price_info: dict,
        action: str,
        current_holding: int,
        jump_flag: bool = True,
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
            if jump_flag:
                return np.array([0.0]), 0
            else:
                # 说明想要从仓位0进行跳转，失败了，需要扣除发送订单和取消订单的成本
                return np.array([-self._send_order_price - self._cancel_order_price]), 0
        if current_holding == -1:
            reward = self.keep_short_reward(price_info)
            if jump_flag:
                return reward, -1
            else:
                # 说明想要从仓位-1进行跳转，失败，需要扣除发送和取消订单成本
                return reward - self._send_order_price - self._cancel_order_price, -1
        if current_holding == 1:
            reward = self.keep_long_reward(price_info)
            if jump_flag:
                return reward, 1
            else:
                # 说明想要从仓位1进行跳转，失败，需要扣除取发送和取消订单成本
                return reward - self._cancel_order_price - self._send_order_price, 1
        raise ValueError(f"Invalid current holding: {current_holding}")
