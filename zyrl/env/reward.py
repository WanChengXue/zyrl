import numpy as np


class ShortLongReward:
    def __init__(self, commission_value) -> None:
        self._commission_value = commission_value

    def open_long_reward(self, price_info, current_holding):
        if current_holding == -1:
            # close short and open long
            close_short_value = self._close_short(price_info)
            open_long_value = self._open_long(price_info)
            return close_short_value + open_long_value

        elif current_holding == 0:
            # open long
            open_long_value = self._open_long(price_info)
            return open_long_value
        else:
            raise f"Error holding: {current_holding}, By Opening long"

    def keep_long_reward(self, price_info):
        return self._keep_long(price_info)

    def close_long_reward(self, price_info):
        return self._close_long(price_info)

    def open_short_reward(self, price_info, current_holding):
        if current_holding == 1:
            close_long_value = self._close_long(price_info)
            open_short_value = self._open_short(price_info)
            return close_long_value + open_short_value
        elif current_holding == 0:
            return self._open_short(price_info)

        else:
            raise f"Error holding: {current_holding}, By Opening short"

    def keep_short_reward(self, price_info):
        return self._keep_short(price_info)

    def close_short_reward(self, price_info):
        return self._close_short(price_info)

    def _open_long(self, price_info):
        next_mp = price_info["next_MP"]
        current_ap = price_info["current_AP"]
        return next_mp - current_ap - self._commission_value

    def _keep_long(self, price_info):
        next_mp = price_info["next_MP"]
        current_mp = price_info["current_MP"]
        return next_mp - current_mp

    def _close_long(self, price_info):
        current_bp = price_info["current_BP"]
        current_mp = price_info["current_MP"]
        return current_bp - self._commission_value - current_mp

    def _open_short(self, price_info):
        current_bp = price_info["current_BP"]
        next_mp = price_info["next_MP"]
        return current_bp - self._commission_value - next_mp

    def _keep_short(self, price_info):
        current_mp = price_info["current_MP"]
        next_mp = price_info["next_MP"]
        return current_mp - next_mp

    def _close_short(self, price_info):
        current_mp = price_info["current_MP"]
        current_ap = price_info["current_AP"]
        return current_mp - current_ap - self._commission_value

    def __call__(self, price_info, action, current_holding) -> tuple[np.ndarray, int]:
        if action in ["OpenLong", "LONG"]:
            return self.open_long_reward(price_info, current_holding), 1
        elif action in ["CloseLong", "PLONG"]:
            return self.close_long_reward(price_info), 0
        elif action in ["OpenShort", "SHORT"]:
            return self.open_short_reward(price_info, current_holding), -1
        elif action in ["CloseShort", "PSHORT"]:
            return self.close_short_reward(price_info), 0
        else:
            if current_holding == 0:
                return np.array([0.0]), 0
            elif current_holding == -1:
                return self.keep_short_reward(price_info), -1
            else:
                return self.keep_long_reward(price_info), 1
