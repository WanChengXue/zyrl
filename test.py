from zyrl.mc import MCFromStart
from zyrl.env.backtest import Backtest
import matplotlib.pyplot as plt
import numpy as np


def test_mc():

    config = {
        "file_path": "data/predict_data",
        "training_data_path": "data/predict_data",
        "training_market_data_path": "data/market_data",
        "commission_value": 0.12,
        "init_state_index": 1,
        "init_holding": 0,
        "gamma": 0.99,
        "saved_q_table_path": "./q_table.csv",
    }

    mc = MCFromStart(config)

    mc.run()


def test_backtest():
    config = {
        "q_table_path": "./q_table.csv",
        "env_config": {
            "training_data_path": "data/predict_data",
            "training_market_data_path": "data/market_data",
            "commission_value": 0.12,
            "util_termination": True,
            "file_name": "20241101_am.csv",
        },
    }
    backtest = Backtest(config)
    reward_list = backtest.run()
    cum_reward_list = np.cumsum(reward_list)
    plt.plot(cum_reward_list)
    plt.show()
    plt.savefig("./figure/cusum_reward.png")
    plt.close()


test_backtest()
