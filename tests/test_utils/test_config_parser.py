#!/usr/bin/env python3
"""
测试新的ConfigParser功能
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyrl.utils.config_parser import ConfigParser


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")

    # 创建测试配置
    test_config = """
pipeline:
  mode: "standard"
  auto_setup: true

env:
  num_envs: 4
  vector_type: "dummy"

network:
  type: "mlp"
  hidden_sizes: [256, 256]
  activation: "relu"

policy:
  type: "dqn"
  lr: 0.001
  gamma: 0.99
"""

    # 写入临时配置文件
    with open("test_config.yaml", "w") as f:
        f.write(test_config)

    try:
        # 创建ConfigParser
        config = ConfigParser("test_config.yaml")

        # 测试基本获取
        print(f"pipeline.mode: {config.get('pipeline.mode')}")
        print(f"env.num_envs: {config.get('env.num_envs')}")
        print(f"network.hidden_sizes: {config.get('network.hidden_sizes')}")

        # 测试扁平化字典
        flat_dict = config.get_all()
        print(f"扁平化字典键: {list(flat_dict.keys())}")

        # 测试树状结构
        tree_dict = config.get_tree()
        print(f"树状结构: {tree_dict}")

        print("✓ 基本功能测试通过")

    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

    finally:
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

    return True


def test_update_functionality():
    """测试更新功能"""
    print("\n=== 测试更新功能 ===")

    try:
        # 创建空的ConfigParser
        config = ConfigParser()

        # 添加配置
        config.update("pipeline.mode", "high_level")
        config.update("env.num_envs", 8)
        config.update("network.hidden_sizes", [512, 512])

        # 验证更新
        assert config.get("pipeline.mode") == "high_level"
        assert config.get("env.num_envs") == 8
        assert config.get("network.hidden_sizes") == [512, 512]

        # 测试字典式访问
        assert config["pipeline.mode"] == "high_level"
        config["policy.lr"] = 0.0001
        assert config["policy.lr"] == 0.0001

        print("✓ 更新功能测试通过")

    except Exception as e:
        print(f"✗ 更新功能测试失败: {e}")
        return False

    return True


def test_merge_functionality():
    """测试合并功能"""
    print("\n=== 测试合并功能 ===")

    try:
        # 创建基础配置
        base_config = ConfigParser()
        base_config.update("pipeline.mode", "standard")
        base_config.update("env.num_envs", 4)
        base_config.update("network.type", "mlp")

        # 创建要合并的配置
        merge_dict = {
            "pipeline.mode": "high_level",  # 覆盖
            "policy.lr": 0.001,  # 新增
            "trainer.max_epoch": 100,  # 新增
        }

        # 合并配置
        base_config.merge(merge_dict, overwrite=True)

        # 验证合并结果
        assert base_config.get("pipeline.mode") == "high_level"  # 被覆盖
        assert base_config.get("env.num_envs") == 4  # 保持不变
        assert base_config.get("policy.lr") == 0.001  # 新增
        assert base_config.get("trainer.max_epoch") == 100  # 新增

        print("✓ 合并功能测试通过")

    except Exception as e:
        print(f"✗ 合并功能测试失败: {e}")
        return False

    return True


def test_file_merge():
    """测试文件合并"""
    print("\n=== 测试文件合并 ===")

    # 创建基础配置文件
    base_config = """
pipeline:
  mode: "standard"
  auto_setup: true

env:
  num_envs: 4
  vector_type: "dummy"
"""

    # 创建合并配置文件
    merge_config = """
pipeline:
  mode: "high_level"

policy:
  type: "dqn"
  lr: 0.001
"""

    # 写入临时文件
    with open("base_config.yaml", "w") as f:
        f.write(base_config)

    with open("merge_config.yaml", "w") as f:
        f.write(merge_config)

    try:
        # 加载基础配置
        config = ConfigParser("base_config.yaml")

        # 合并另一个配置文件
        config.merge_file("merge_config.yaml", overwrite=True)

        # 验证合并结果
        assert config.get("pipeline.mode") == "high_level"  # 被覆盖
        assert config.get("env.num_envs") == 4  # 保持不变
        assert config.get("policy.type") == "dqn"  # 新增
        assert config.get("policy.lr") == 0.001  # 新增

        print("✓ 文件合并测试通过")

    except Exception as e:
        print(f"✗ 文件合并测试失败: {e}")
        return False

    finally:
        # 清理临时文件
        for file in ["base_config.yaml", "merge_config.yaml"]:
            if os.path.exists(file):
                os.remove(file)

    return True


def test_advanced_features():
    """测试高级功能"""
    print("\n=== 测试高级功能 ===")

    try:
        # 创建配置
        config = ConfigParser()
        config.update("pipeline.mode", "standard")
        config.update("env.num_envs", 4)
        config.update("network.hidden_sizes", [256, 256])

        # 测试节点获取
        node = config.get_node("pipeline")
        assert node is not None
        assert node.name == "pipeline"

        # 测试键存在检查
        assert config.has_key("pipeline.mode")
        assert "pipeline.mode" in config
        assert not config.has_key("nonexistent.key")

        # 测试删除
        config.remove("env.num_envs")
        assert not config.has_key("env.num_envs")

        # 测试长度
        assert len(config) == 2  # pipeline.mode 和 network.hidden_sizes

        # 测试清空
        config.clear()
        assert len(config) == 0

        print("✓ 高级功能测试通过")

    except Exception as e:
        print(f"✗ 高级功能测试失败: {e}")
        return False

    return True


def test_save_functionality():
    """测试保存功能"""
    print("\n=== 测试保存功能 ===")

    try:
        # 创建配置
        config = ConfigParser()
        config.update("pipeline.mode", "high_level")
        config.update("env.num_envs", 8)
        config.update("network.hidden_sizes", [512, 512])

        # 保存到文件
        config.save("test_save.yaml")

        # 重新加载验证
        loaded_config = ConfigParser("test_save.yaml")
        assert loaded_config.get("pipeline.mode") == "high_level"
        assert loaded_config.get("env.num_envs") == 8
        assert loaded_config.get("network.hidden_sizes") == [512, 512]

        print("✓ 保存功能测试通过")

    except Exception as e:
        print(f"✗ 保存功能测试失败: {e}")
        return False

    finally:
        if os.path.exists("test_save.yaml"):
            os.remove("test_save.yaml")

    return True


def test_update_same_leaf_nodes():
    """测试更新相同叶子节点功能"""
    print("\n=== 测试更新相同叶子节点功能 ===")

    try:
        # 创建基础配置
        base_config = ConfigParser()
        base_config.update("pipeline.mode", "standard")
        base_config.update("env.num_envs", 4)
        base_config.update("network.type", "mlp")
        base_config.update("policy.lr", 0.001)

        # 创建要更新的配置（只包含部分相同键）
        update_dict = {
            "pipeline.mode": "high_level",  # 相同键，会被更新
            "env.num_envs": 8,  # 相同键，会被更新
            "policy.lr": 0.0001,  # 相同键，会被更新
            "new.key": "new_value",  # 新键，不会被更新
            "trainer.max_epoch": 100,  # 新键，不会被更新
        }

        # 更新相同叶子节点
        updated_count = base_config.update_same_leaf_nodes(update_dict, overwrite=True)

        # 验证结果
        assert updated_count == 3  # 只更新了3个相同键
        assert base_config.get("pipeline.mode") == "high_level"  # 被更新
        assert base_config.get("env.num_envs") == 8  # 被更新
        assert base_config.get("policy.lr") == 0.0001  # 被更新
        assert base_config.get("network.type") == "mlp"  # 保持不变
        assert not base_config.has_key("new.key")  # 新键未被添加
        assert not base_config.has_key("trainer.max_epoch")  # 新键未被添加

        print("✓ 更新相同叶子节点功能测试通过")

    except Exception as e:
        print(f"✗ 更新相同叶子节点功能测试失败: {e}")
        return False

    return True


def test_update_same_leaf_nodes_from_file():
    """测试从文件更新相同叶子节点功能"""
    print("\n=== 测试从文件更新相同叶子节点功能 ===")

    # 创建基础配置文件
    base_config = """
pipeline:
  mode: "standard"
  auto_setup: true

env:
  num_envs: 4
  vector_type: "dummy"

network:
  type: "mlp"
  hidden_sizes: [256, 256]
"""

    # 创建更新配置文件（只包含部分相同键）
    update_config = """
pipeline:
  mode: "high_level"

env:
  num_envs: 8

policy:
  type: "dqn"
  lr: 0.001
"""

    # 写入临时文件
    with open("base_config.yaml", "w") as f:
        f.write(base_config)

    with open("update_config.yaml", "w") as f:
        f.write(update_config)

    try:
        # 加载基础配置
        config = ConfigParser("base_config.yaml")

        # 从文件更新相同叶子节点
        updated_count = config.update_same_leaf_nodes_from_file(
            "update_config.yaml", overwrite=True
        )

        # 验证结果
        assert updated_count == 2  # 只更新了2个相同键
        assert config.get("pipeline.mode") == "high_level"  # 被更新
        assert config.get("env.num_envs") == 8  # 被更新
        assert config.get("network.type") == "mlp"  # 保持不变
        assert config.get("network.hidden_sizes") == [256, 256]  # 保持不变
        assert not config.has_key("policy.type")  # 新键未被添加
        assert not config.has_key("policy.lr")  # 新键未被添加

        print("✓ 从文件更新相同叶子节点功能测试通过")

    except Exception as e:
        print(f"✗ 从文件更新相同叶子节点功能测试失败: {e}")
        return False

    finally:
        # 清理临时文件
        for file in ["base_config.yaml", "update_config.yaml"]:
            if os.path.exists(file):
                os.remove(file)

    return True


def test_common_keys_and_differences():
    """测试获取相同键和不同值功能"""
    print("\n=== 测试获取相同键和不同值功能 ===")

    try:
        # 创建第一个配置
        config1 = ConfigParser()
        config1.update("pipeline.mode", "standard")
        config1.update("env.num_envs", 4)
        config1.update("network.type", "mlp")
        config1.update("policy.lr", 0.001)

        # 创建第二个配置
        config2 = ConfigParser()
        config2.update("pipeline.mode", "high_level")  # 值不同
        config2.update("env.num_envs", 4)  # 值相同
        config2.update("network.type", "cnn")  # 值不同
        config2.update("policy.lr", 0.001)  # 值相同
        config2.update("new.key", "new_value")  # 新键

        # 获取相同键
        common_keys = config1.get_common_keys(config2)
        assert set(common_keys) == {
            "pipeline.mode",
            "env.num_envs",
            "network.type",
            "policy.lr",
        }

        # 获取不同值
        differences = config1.get_different_values(config2)
        assert len(differences) == 2  # 只有2个键的值不同
        assert "pipeline.mode" in differences
        assert "network.type" in differences
        assert differences["pipeline.mode"]["current"] == "standard"
        assert differences["pipeline.mode"]["other"] == "high_level"
        assert differences["network.type"]["current"] == "mlp"
        assert differences["network.type"]["other"] == "cnn"

        print("✓ 获取相同键和不同值功能测试通过")

    except Exception as e:
        print(f"✗ 获取相同键和不同值功能测试失败: {e}")
        return False

    return True


def test_no_overwrite_mode():
    """测试不覆盖模式"""
    print("\n=== 测试不覆盖模式 ===")

    try:
        # 创建基础配置
        base_config = ConfigParser()
        base_config.update("pipeline.mode", "standard")
        base_config.update("env.num_envs", 4)
        base_config.update("policy.lr", 0.001)

        # 创建要更新的配置
        update_dict = {
            "pipeline.mode": "high_level",
            "env.num_envs": 8,
            "policy.lr": 0.0001,
        }

        # 不覆盖模式更新
        updated_count = base_config.update_same_leaf_nodes(update_dict, overwrite=False)

        # 验证结果（值应该保持不变）
        assert updated_count == 0  # 没有更新任何值
        assert base_config.get("pipeline.mode") == "standard"  # 保持不变
        assert base_config.get("env.num_envs") == 4  # 保持不变
        assert base_config.get("policy.lr") == 0.001  # 保持不变

        print("✓ 不覆盖模式测试通过")

    except Exception as e:
        print(f"✗ 不覆盖模式测试失败: {e}")
        return False

    return True


def main():
    """运行所有测试"""
    print("开始测试新的ConfigParser功能...\n")

    tests = [
        test_basic_functionality,
        test_update_functionality,
        test_merge_functionality,
        test_file_merge,
        test_advanced_features,
        test_save_functionality,
        test_update_same_leaf_nodes,
        test_update_same_leaf_nodes_from_file,
        test_common_keys_and_differences,
        test_no_overwrite_mode,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！新的ConfigParser功能正常。")
    else:
        print("❌ 部分测试失败，请检查代码。")


if __name__ == "__main__":
    main()
