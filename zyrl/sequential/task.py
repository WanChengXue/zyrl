import os
import pandas as pd
import torch
import torch.nn as nn
from zyrl.utils.table_utils import load_dataframe
from zyrl.mc import MCFromStart
from zyrl.algo import RLTrainer, SLTrainer
from tqdm import tqdm
from tianshou.data import Batch
import copy


class Task:
    def __init__(
        self,
        env_cls,
        task_name: str,
        env_config: dict,
        init_q_table: pd.DataFrame | None = None,
        mc_config: dict | None = None,
        *,
        train_algo: str | None = "MCFROMSTART",
        rl_config: dict | None = None,
    ):
        self._env_cls = env_cls
        self._env_config = env_config
        self._init_q_table = init_q_table
        self._mc_config = mc_config
        self._train_algo = train_algo
        self._rl_config = rl_config
        self._status = "untrained"
        self._task_name = task_name

    def _create_root_folder(self, root_folder: str):
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

    def _pretrained_sl(
        self, checkpoint_folder: str, rl_trainer_model: nn.Module
    ) -> dict:
        sl_config = self._rl_config["pretrained_config"]
        sl_config["checkpoint_folder"] = f"{checkpoint_folder}/sl"
        sl_config["model"] = rl_trainer_model
        sl_trainer = SLTrainer(learning_config=sl_config)
        sl_trainer.learn()
        return sl_trainer.get_checkpoint_path()

    def run(self, root_folder: str):
        checkpoint_folder = f"{root_folder}/{self._task_name}"
        self._create_root_folder(checkpoint_folder)
        if self._train_algo == "MCFROMSTART":
            if "saved_q_table_path" not in self._mc_config:
                self._mc_config["saved_q_table_path"] = (
                    f"{checkpoint_folder}/q_table.csv"
                )
            if "saved_count_table_path" not in self._mc_config:
                self._mc_config["saved_count_table_path"] = (
                    f"{checkpoint_folder}/count_table.csv"
                )
            if "loss_plot_path" not in self._mc_config:
                self._mc_config["loss_plot_path"] = f"{checkpoint_folder}/loss_plot.png"
            mc = MCFromStart(
                self._env_cls, self._init_q_table, self._env_config, self._mc_config
            )
            mc.run()

        if self._train_algo == "RL":
            self._rl_config["checkpoint_folder"] = checkpoint_folder
            rl_trainer = RLTrainer(rl_config=self._rl_config)
            rl_trainer.save_checkpoint()
            rl_trainer_model = rl_trainer.get_model()
            if self._rl_config["pretrained"] == True:
                sl_trainer_checkpoint = self._pretrained_sl(
                    checkpoint_folder, rl_trainer_model
                )
                rl_trainer.load_checkpoint(sl_trainer_checkpoint)
            rl_trainer.train(self._env_cls, self._env_config)
            self._rl_config["checkpoint_path"] = rl_trainer.get_checkpoint_path()
            del rl_trainer

    def load_checkpoint(self, checkpoint_folder: str):
        if self._train_algo == "MCFROMSTART":
            q_table_path = f"{checkpoint_folder}/{self._task_name}/q_table.csv"
            self._q_table = pd.read_csv(q_table_path, index_col=0)
        if self._train_algo == "RL":
            model_best_path = f"{checkpoint_folder}/{self._task_name}/model_best.pth"
            self._rl_config["checkpoint_path"] = model_best_path
        if self._train_algo == "SL":
            sl_model_path = f"{checkpoint_folder}/{self._task_name}/sl/model.pth"
            self._rl_config["checkpoint_path"] = sl_model_path

        self._status = "trained"

    def evaluate(self, test_folder_path: str):
        if self._train_algo in ["RL", "SL"]:
            return self.evaluate_rl(test_folder_path)
        else:
            return self.evaluate_mc(test_folder_path)

    def evaluate_mc(self, test_folder_path: str):
        file_list = os.listdir(test_folder_path)
        act_env = self._env_cls(self._env_config)
        result_dict = {}
        for file_name in tqdm(sorted(file_list)):
            test_file = os.path.join(test_folder_path, file_name)
            data = load_dataframe(test_file)
            action_list = []
            for index, row in data.iterrows():
                state_index = act_env.get_state_index_from_raw_data(row)
                q_list = self._q_table.iloc[state_index]
                if "state_index" in q_list.index:
                    q_list = q_list.drop("state_index")
                action = int(q_list.idxmax())
                action_list.append(action)
            result_dict[file_name] = action_list
        return result_dict

    def evaluate_rl(self, test_folder_path: str):
        file_list = os.listdir(test_folder_path)
        act_env = self._env_cls(self._env_config)
        rl_trainer = RLTrainer(rl_config=self._rl_config)
        rl_trainer.load_checkpoint(self._rl_config["checkpoint_path"])
        evaluate_device = rl_trainer.get_device()
        model = rl_trainer.get_model().to(evaluate_device, dtype=torch.float32)
        model.device = evaluate_device
        result_dict = {}
        for file_name in tqdm(sorted(file_list)):
            test_file = os.path.join(test_folder_path, file_name)
            data = act_env.get_state_from_file(test_file)
            action_list = model(data)[0]
            argmax_action = torch.argmax(action_list, dim=1).cpu().numpy()
            result_dict[file_name] = argmax_action
        return result_dict

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

    def get_rl_config(self):
        return self._rl_config

    def get_train_algo(self):
        return self._train_algo

    def set_status(self, status: str):
        self._status = status

    def get_status(self):
        return self._status

    def get_env_observation_space(self):
        return self._env_cls(self._env_config).observation_space

    def get_env_action_space(self):
        return self._env_cls(self._env_config).action_space

    def rollout_rl(self, reset_info: dict):
        env = self._env_cls(self._env_config)
        done = False
        current_state, info = env.reset(options=reset_info)
        rollout_rl_config = copy.deepcopy(self._rl_config)
        rollout_rl_config["device"] = "cpu"
        rl_trainer = RLTrainer(rl_config=rollout_rl_config)
        rl_trainer.load_checkpoint(rollout_rl_config["checkpoint_path"])
        model = rl_trainer.get_model()
        sum_reward = 0
        while not done:
            batch_state = Batch(current_state)
            q_list = model(batch_state)[0]
            action = torch.argmax(q_list).item()
            current_state, reward, done, _, info = env.step(action)
            sum_reward += reward
        return sum_reward, info

    def rollout(self, reset_info: dict):
        if self._train_algo == "RL":
            return self.rollout_rl(reset_info)

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
