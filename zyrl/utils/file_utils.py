import yaml
import os
import importlib.resources


def load_yaml(file_path: str):
    # 优先尝试本地文件
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    # 否则尝试从zyrl.config包内资源读取
    else:
        # 根据file_path动态确定包路径
        if file_path.startswith("config/"):
            # 移除'config/'前缀
            relative_path = file_path[7:]  # len('config/') = 7
            if "/" in relative_path:
                # 有子目录，如 'algo_config/dqn_config.yaml'
                dir_path, fname = os.path.split(relative_path)
                pkg = f"zyrl.config.{dir_path.replace('/', '.')}"
            else:
                # 直接在config目录下，如 'common_config.yaml'
                fname = relative_path
                pkg = "zyrl.config"
        else:
            # 默认处理
            pkg = "zyrl.config"
            fname = os.path.basename(file_path)

        try:
            with importlib.resources.open_text(pkg, fname, encoding="utf-8") as file:
                return yaml.safe_load(file)
        except (FileNotFoundError, ModuleNotFoundError) as e:
            raise FileNotFoundError(
                f"找不到配置文件: {file_path}，也未在包内资源{pkg}下找到{fname}。"
            ) from e


def save_yaml(data: dict, file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
