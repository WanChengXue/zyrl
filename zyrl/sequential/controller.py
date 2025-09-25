import os
import time
from zyrl.sequential.task import Task
from zyrl.sequential.edge import TaskLink


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
