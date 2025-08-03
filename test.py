from zyrl.mc import MCFromStart


config = {
    "file_path": "data/predict_data",
    "init_state_index": 1,
    "init_holding": 0,
    "gamma": 0.99,
}

mc = MCFromStart(config)
