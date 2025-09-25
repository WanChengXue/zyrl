from zyrl.sequential.task import Task
import gymnasium as gym


class MergedEnv(gym.Env):
    def __init__(self, env_config: dict):
        task_list = env_config["task_list"]
        self._task_list = task_list
        self._main_task = task_list[0]
        self._main_env = self._main_task.get_env()
        self._sub_task_list = self._task_list[1:]
        self.observation_space = self._main_task.get_env_observation_space()
        self.action_space = self._main_task.get_env_action_space()

    def get_main_task(self):
        return self._main_task

    def get_state_index(self, state):
        return self._main_env.get_state_index(state)

    def reset(self, **kwargs):
        return self._main_env.reset(**kwargs)

    def step(self, action):
        next_state, reward, done, _, info = self._main_env.step(action)
        acc_reward = 0
        if done:
            for sub_task in self._sub_task_list:
                sum_reward, info = sub_task.rollout(info)
                acc_reward += sum_reward
        reward += acc_reward
        return next_state, reward, done, _, info


class TaskLink:
    def __init__(self, task_list: list[Task]):
        self._task_list = task_list

    def run(self, checkpoint_folder: str):
        task_list = []
        for task in reversed(self._task_list):
            task_list.append(task)
            if task.get_status() == "untrained" and len(task_list) == 1:
                task.run(checkpoint_folder)
                task.set_status("trained")

            if task.get_status() == "untrained" and len(task_list) > 1:
                constructed_task = self.construct_task(list(reversed(task_list)))
                constructed_task.run(checkpoint_folder)
                main_task = constructed_task.get_env().get_main_task()
                main_task.set_status("trained")

    def construct_task(self, task_list: list[Task]):
        main_task = task_list[0]
        compound_task = Task(
            MergedEnv,
            task_name=main_task.get_task_name(),
            env_config={"task_list": task_list},
            init_q_table=main_task.get_init_q_table(),
            mc_config=main_task.get_mc_config(),
        )
        return compound_task
