import numpy as np


def generate_state_table():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])
                index_state_dict[index] = (first_region, second_region, third_region)
                state_index_mapping[(first_region, second_region, third_region)] = index
                index += 1
    return index_state_dict, state_index_mapping


def test_env():
    from zyrl.env.open_close_env.env import SplitStateActionEnv

    config = {
        "data_path": "./data/split_state_action_data",
        "env_type": "close_long",
        "commission_value": 0.12,
        "index_state_dict_path": "./data/split_state_action_data_index/index_state_dict.npy",
        "state_index_mapping_path": "./data/split_state_action_data_index/state_index_mapping.npy",
    }

    env = SplitStateActionEnv(config)

    state = env.reset(options={"init_state_index": 0})
    print(state)

    action = env.action_space.sample()
    env.step(action)


if __name__ == "__main__":
    index_state_dict, state_index_mapping = generate_state_table()
    np.save(
        "./data/split_state_action_data_index/index_state_dict.npy", index_state_dict
    )
    np.save(
        "./data/split_state_action_data_index/state_index_mapping.npy",
        state_index_mapping,
    )
