import os
import pandas as pd
from zyrl.mc import MCFromStart


class Task:
    def __init__(
        self,
        env_cls,
        task_name: str,
        env_config: dict,
        init_q_table: pd.DataFrame | None = None,
        mc_config: dict | None = None,
    ):
        self._env_cls = env_cls
        self._env_config = env_config
        self._init_q_table = init_q_table
        self._mc_config = mc_config
        self._status = "untrained"
        self._task_name = task_name

    def _create_root_folder(self, root_folder: str):
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

    def run(self, root_folder: str):
        checkpoint_folder = f"{root_folder}/{self._task_name}"
        self._create_root_folder(checkpoint_folder)
        if "saved_q_table_path" not in self._mc_config:
            self._mc_config["saved_q_table_path"] = f"{checkpoint_folder}/q_table.csv"
        if "saved_count_table_path" not in self._mc_config:
            self._mc_config["saved_count_table_path"] = (
                f"{checkpoint_folder}/count_table.csv"
            )
        if "mse_loss_plot_path" not in self._mc_config:
            self._mc_config["mse_loss_plot_path"] = (
                f"{checkpoint_folder}/mse_loss_plot.png"
            )
        mc = MCFromStart(
            self._env_cls, self._init_q_table, self._env_config, self._mc_config
        )
        mc.run()

    def get_env(self):
        return self._env_cls(self._env_config)

    def get_task_name(self):
        return self._task_name

    def get_env_config(self):
        return self._env_config

    def get_init_q_table(self):
        return self._init_q_table

    def get_mc_config(self):
        return self._mc_config

    def set_status(self, status: str):
        self._status = status

    def get_status(self):
        return self._status

    def get_env_observation_space(self):
        return self._env_cls(self._env_config).observation_space

    def get_env_action_space(self):
        return self._env_cls(self._env_config).action_space

    def rollout(self, reset_info: dict):
        env = self._env_cls(self._env_config)
        done = False
        current_state, info = env.reset(options=reset_info)
        q_table = pd.read_csv(self._mc_config["saved_q_table_path"])
        sum_reward = 0
        while not done:
            q_table_index = env.get_state_index(current_state)
            q_list = q_table.iloc[q_table_index]
            if "state_index" in q_list.index:
                q_list = q_list.drop("state_index")
            action = int(q_list.idxmax())
            current_state, reward, done, _, info = env.step(action)
            sum_reward += reward
        return sum_reward, info
