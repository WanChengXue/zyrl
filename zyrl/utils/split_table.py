"""Table splitting utilities for trading data.

This module provides functions to split trading data tables based on trade periods
and time-based criteria.
"""

from tqdm import tqdm
from zyrl.utils.table_utils import load_dataframe


def split_table(table_path: str, saved_path: str) -> None:
    """Split a table into two parts based on a split ratio.

    Args:
        table_path (str): The path to the table to split.
        saved_path (str): The path to save the two split tables.
    """
    table = load_dataframe(table_path)
    table_length = len(table)
    last_ts = table.iloc[0]["localtimeStr"]
    trade_period = get_trade_period(last_ts)
    start_period_index = 0
    for index in tqdm(range(table_length)):
        current_ts = table.iloc[index]["localtimeStr"]
        if jump_trade_period(last_ts, current_ts):
            table[start_period_index:index].to_csv(
                f"{saved_path}/{trade_period}.csv", index=False
            )
            start_period_index = index
            trade_period = get_trade_period(current_ts)
            last_ts = current_ts
    table[start_period_index:table_length].to_csv(
        f"{saved_path}/{trade_period}.csv", index=False
    )


def get_trade_period(ts: str) -> str:
    """Get the trade period (AM/PM) from a timestamp string.

    Args:
        ts (str): Timestamp string in format 'YYYYMMDD_HHMMSS'.

    Returns:
        str: Trade period string in format 'YYYYMMDD_AM' or 'YYYYMMDD_PM'.
    """
    trade_day = ts[:8]
    trade_hour = ts[9:11]
    if trade_hour in ["09", "10", "11"]:
        trade_period = "AM"
    else:
        trade_period = "PM"
    return f"{trade_day}_{trade_period}"


def jump_trade_period(last_ts: str, current_ts: str) -> bool:
    """Check if the current time is a trade period.

    Args:
        last_ts (str): The last time.
        current_ts (str): The current time.

    Returns:
        bool: True if the current time is a trade period, False otherwise.
    """
    last_trade_day = last_ts[:8]
    current_trade_day = current_ts[:8]

    last_trade_hour = last_ts[9:11]
    current_trade_hour = current_ts[9:11]

    if current_trade_day != last_trade_day:
        return True

    am_trade_hour_list = ["09", "10", "11"]
    pm_trade_hour_list = ["13", "14", "15"]
    am_trade_flag = (
        current_trade_hour in am_trade_hour_list
        and last_trade_hour in am_trade_hour_list
    )
    pm_trade_flag = (
        current_trade_hour in pm_trade_hour_list
        and last_trade_hour in pm_trade_hour_list
    )

    same_period_flag = am_trade_flag or pm_trade_flag
    jump_period_flag = not same_period_flag
    return jump_period_flag


if __name__ == "__main__":
    split_table(
        table_path="/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/split_state_tradeRateTable.feather",
        saved_path="/home/zydl-dev-cl/Desktop/rl_finance/zyrl/data/split_state_action_data",
    )
