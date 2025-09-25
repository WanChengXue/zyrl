import os
import tempfile
import pandas as pd
import numpy as np
from zyrl.env.open_close_env.env import SplitStateActionEnv
from zyrl.env.open_close_env.env_backtest import SplitStateActionEnvBacktest


def create_temp_dataframe():
    data = pd.DataFrame(
        {
            "TradeRate_NF_Long_Delta1": np.random.rand(100),
            "TradeRate_NF_Long_Delta2": np.random.rand(100),
            "TradeRate_NF_Long_Delta3": np.random.rand(100),
            "TradeRate_NF_Short_Delta1": np.random.rand(100),
            "TradeRate_NF_Short_Delta2": np.random.rand(100),
            "TradeRate_NF_Short_Delta3": np.random.rand(100),
            "TradeRate_Long_Delta1": np.random.rand(100),
            "TradeRate_Long_Delta2": np.random.rand(100),
            "TradeRate_Long_Delta3": np.random.rand(100),
            "TradeRate_Short_Delta1": np.random.rand(100),
            "TradeRate_Short_Delta2": np.random.rand(100),
            "TradeRate_Short_Delta3": np.random.rand(100),
            "MidPrice": np.random.rand(100),
            "Price_Long_Delta1": np.random.rand(100),
            "Price_Long_Delta2": np.random.rand(100),
            "Price_Long_Delta3": np.random.rand(100),
            "Price_Short_Delta1": np.random.rand(100),
            "Price_Short_Delta2": np.random.rand(100),
            "Price_Short_Delta3": np.random.rand(100),
        }
    )
    return data


def test_get_price_info():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, "split_state_action_data")
        os.makedirs(data_path, exist_ok=True)
        return data_path


def test_open_close_env():
    config = {
        "data_path": "./data/split_state_action_data",
        "env_type": "open_long",
    }
    env = SplitStateActionEnv(config)
    env.reset()
    env.step(0)
