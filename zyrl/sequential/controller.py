import os
import time
import pandas as pd
from zyrl.sequential.task import Task
from zyrl.sequential.edge import TaskLink
from zyrl.utils.table_utils import load_dataframe


class Controller:
    def __init__(
        self,
        *,
        task_list: list[Task],
        link_list: list[TaskLink],
        checkpoint_folder: str | None = None,
    ):
        self._task_list = task_list
        self._link_list = link_list
        self._checkpoint_folder = checkpoint_folder
        self.create_checkpoint_folder()

    def create_checkpoint_folder(self):
        if self._checkpoint_folder is None:
            self._checkpoint_folder = (
                f"tmp/sequential/checkpoint_{time.strftime('%Y%m%d%H%M%S')}"
            )

        if not os.path.exists(self._checkpoint_folder):
            os.makedirs(self._checkpoint_folder)

    def train(self):
        for _link in self._link_list:
            _link.run(self._checkpoint_folder)

    def load_checkpoint(self, checkpoint_folder: str):
        for _task in self._task_list:
            _task.load_checkpoint(checkpoint_folder)

    def evaluate(
        self, checkpoint_folder: str, test_folder_path: str, saved_folder_path: str
    ):
        self.load_checkpoint(checkpoint_folder)
        result_dict = {}
        for _task in self._task_list:
            res = _task.evaluate(test_folder_path)
            result_dict[_task.get_task_name()] = res
        self.save_result(test_folder_path, result_dict, saved_folder_path)

    def save_result(
        self, test_folder_path: str, result_dict: dict, saved_folder_path: str
    ):
        data_list = []
        for file_name in sorted(os.listdir(test_folder_path)):
            test_file = os.path.join(test_folder_path, file_name)
            data = load_dataframe(test_file)
            for task_name, action_dict in result_dict.items():
                action_list = action_dict[file_name]
                data[task_name] = action_list
                data.to_csv(os.path.join(saved_folder_path, file_name), index=False)
            data_list.append(data)
        concat_data = pd.concat(data_list, axis=0)
        concat_data.to_csv(
            os.path.join(saved_folder_path, "concat_data.csv"), index=False
        )
