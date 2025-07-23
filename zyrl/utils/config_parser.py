import yaml
import json
import os
from typing import Any, Dict, Union, Optional, List
from collections import defaultdict

from zyrl.utils.file_utils import load_yaml


class ConfigNode:
    """配置树节点"""

    def __init__(self, name: str = "", value: Any = None, parent=None):
        self.name = name
        self.value = value
        self.parent = parent
        self.children = {}
        self.is_leaf = value is not None

    def add_child(self, name: str, value: Any = None):
        """添加子节点"""
        if name in self.children:
            # 如果节点已存在且是叶子节点，转换为内部节点
            if self.children[name].is_leaf:
                self.children[name].is_leaf = False
                self.children[name].value = None
        else:
            self.children[name] = ConfigNode(name, value, self)
        return self.children[name]

    def get_child(self, name: str):
        """获取子节点"""
        return self.children.get(name)

    def get_value(self):
        """获取节点值（仅叶子节点）"""
        return self.value if self.is_leaf else None

    def set_value(self, value: Any):
        """设置节点值"""
        self.value = value
        self.is_leaf = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if self.is_leaf:
            return self.value

        result = {}
        for child in self.children.values():
            if child.is_leaf:
                result[child.name] = child.value
            else:
                result[child.name] = child.to_dict()
        return result

    def get_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        """获取扁平化字典，key用.连接"""
        result = {}

        if self.is_leaf:
            result[prefix] = self.value
        else:
            for child in self.children.values():
                child_prefix = f"{prefix}.{child.name}" if prefix else child.name
                result.update(child.get_flat_dict(child_prefix))

        return result


class ConfigParser:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self._root = ConfigNode("root")
        self._flat_dict = {}

        if config_path:
            self._load_config()

    def _load_config(self):
        """加载配置文件"""
        config_dict = load_yaml(self.config_path)
        self._build_tree(config_dict)
        self._update_flat_dict()

    def _build_tree(self, config_dict: Dict[str, Any], parent: ConfigNode = None):
        """构建配置树"""
        if parent is None:
            parent = self._root

        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 创建内部节点
                node = parent.add_child(key)
                self._build_tree(value, node)
            else:
                # 创建叶子节点
                parent.add_child(key, value)

    def _update_flat_dict(self):
        """更新扁平化字典"""
        self._flat_dict = self._root.get_flat_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键（用.分隔）"""
        return self._flat_dict.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置（扁平化字典）"""
        return self._flat_dict.copy()

    def get_tree(self) -> Dict[str, Any]:
        """获取树状结构配置"""
        return self._root.to_dict()

    def update(self, key: str, value: Any):
        """更新配置值"""
        keys = key.split(".")
        current = self._root

        # 遍历到倒数第二个节点
        for k in keys[:-1]:
            child = current.get_child(k)
            if child is None:
                child = current.add_child(k)
            current = child

        # 设置最后一个节点的值
        last_key = keys[-1]
        child = current.get_child(last_key)
        if child is None:
            current.add_child(last_key, value)
        else:
            child.set_value(value)

        self._update_flat_dict()

    def merge(
        self,
        other_config: Union[Dict[str, Any], "ConfigParser"],
        overwrite: bool = True,
    ):
        """合并配置"""
        if isinstance(other_config, ConfigParser):
            other_dict = other_config.get_all()
        else:
            other_dict = other_config

        for key, value in other_dict.items():
            if overwrite or key not in self._flat_dict:
                self.update(key, value)

    def merge_file(self, config_path: str, overwrite: bool = True):
        """合并配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path}")

        self.merge(config_dict, overwrite)

    def save(self, file_path: str = None):
        """保存配置到文件"""
        if file_path is None:
            file_path = self.config_path

        if file_path is None:
            raise ValueError("未指定保存路径")

        config_dict = self.get_tree()

        with open(file_path, "w", encoding="utf-8") as f:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
            elif file_path.endswith(".json"):
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path}")

    def get_node(self, key: str) -> Optional[ConfigNode]:
        """获取指定路径的节点"""
        keys = key.split(".")
        current = self._root

        for k in keys:
            child = current.get_child(k)
            if child is None:
                return None
            current = child

        return current

    def has_key(self, key: str) -> bool:
        """检查是否存在指定键"""
        return key in self._flat_dict

    def remove(self, key: str):
        """删除指定键"""
        keys = key.split(".")
        current = self._root

        # 遍历到倒数第二个节点
        for k in keys[:-1]:
            child = current.get_child(k)
            if child is None:
                return  # 键不存在
            current = child

        # 删除最后一个节点
        last_key = keys[-1]
        if last_key in current.children:
            del current.children[last_key]
            self._update_flat_dict()

    def clear(self):
        """清空所有配置"""
        self._root = ConfigNode("root")
        self._flat_dict = {}

    def __str__(self) -> str:
        """字符串表示"""
        return f"ConfigParser(keys={list(self._flat_dict.keys())})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        """返回配置项数量"""
        return len(self._flat_dict)

    def __contains__(self, key: str) -> bool:
        """检查是否包含指定键"""
        return self.has_key(key)

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        if key not in self._flat_dict:
            raise KeyError(f"配置键不存在: {key}")
        return self._flat_dict[key]

    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.update(key, value)

    def update_same_leaf_nodes(
        self,
        other_config: Union[Dict[str, Any], "ConfigParser"],
        overwrite: bool = True,
    ):
        """
        更新当前parser中与other_config相同叶子节点的值

        Args:
            other_config: 要合并的配置（字典或ConfigParser）
            overwrite: 是否覆盖现有值

        Returns:
            int: 更新的节点数量
        """
        if isinstance(other_config, ConfigParser):
            other_dict = other_config.get_all()
        else:
            other_dict = other_config

        updated_count = 0

        for key, value in other_dict.items():
            # 检查当前parser中是否存在相同的叶子节点
            if self.has_key(key):
                if overwrite:
                    self.update(key, value)
                    updated_count += 1
            else:
                # 如果不存在，可以选择是否添加新节点
                # 这里我们选择不添加，只更新已存在的节点
                pass

        return updated_count

    def update_same_leaf_nodes_from_file(
        self, config_path: str, overwrite: bool = True
    ):
        """
        从文件更新当前parser中相同叶子节点的值

        Args:
            config_path: 配置文件路径
            overwrite: 是否覆盖现有值

        Returns:
            int: 更新的节点数量
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path}")

        return self.update_same_leaf_nodes(config_dict, overwrite)

    def get_common_keys(
        self, other_config: Union[Dict[str, Any], "ConfigParser"]
    ) -> List[str]:
        """
        获取与另一个配置相同的键

        Args:
            other_config: 要比较的配置（字典或ConfigParser）

        Returns:
            List[str]: 相同的键列表
        """
        if isinstance(other_config, ConfigParser):
            other_dict = other_config.get_all()
        else:
            other_dict = other_config

        return list(set(self._flat_dict.keys()) & set(other_dict.keys()))

    def get_different_values(
        self, other_config: Union[Dict[str, Any], "ConfigParser"]
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取与另一个配置相同键但值不同的项

        Args:
            other_config: 要比较的配置（字典或ConfigParser）

        Returns:
            Dict[str, Dict[str, Any]]: 格式为 {key: {"current": value, "other": value}}
        """
        if isinstance(other_config, ConfigParser):
            other_dict = other_config.get_all()
        else:
            other_dict = other_config

        differences = {}
        common_keys = self.get_common_keys(other_config)

        for key in common_keys:
            current_value = self._flat_dict[key]
            other_value = other_dict[key]

            if current_value != other_value:
                differences[key] = {"current": current_value, "other": other_value}

        return differences
