#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析MCFromStart类的性能开销
"""

import cProfile
import pstats
import time
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from zyrl.profiler import Profiler, quick_profile
from zyrl.mc import MCFromStart


def analyze_mc_methods():
    """分析MCFromStart各个方法的性能"""

    # 配置
    config = {
        "file_path": "./data",
        "saved_q_table_path": "./results/q_table.csv",
        "gamma": 0.99,
        "training_data_path": "./training_data",
        "training_market_data_path": "./market_data",
        "test_data_path": "./test_data",
        "test_market_data_path": "./test_market_data",
        "commission_value": 0.12,
    }

    print("=== MCFromStart性能分析 ===\n")

    # 创建实例
    print("创建MCFromStart实例...")
    mc = MCFromStart(config)

    # 创建性能分析器
    profiler = Profiler("mc_detailed_profile.prof")

    # 1. 分析初始化Q表
    print("\n1. 分析Q表初始化性能...")
    try:
        profiler.profile_function(mc._init_q_table)
    except Exception as e:
        print(f"Q表初始化分析失败: {e}")

    # 2. 分析数据收集（只分析一小部分）
    print("\n2. 分析数据收集性能（限制范围）...")
    try:
        # 临时修改配置，只处理少量数据
        original_file_list = mc._file_list
        mc._file_list = mc._file_list[:2] if len(mc._file_list) > 2 else mc._file_list

        profiler.profile_function(mc._collect_data)

        # 恢复原始文件列表
        mc._file_list = original_file_list
    except Exception as e:
        print(f"数据收集分析失败: {e}")

    # 3. 分析Q表更新
    print("\n3. 分析Q表更新性能...")
    try:
        # 创建测试数据
        test_return_dict = {
            (1, 0, 1): [0.5, 0.3, 0.8],
            (1, 0, 0): [0.2, 0.4],
            (0, 1, -1): [0.1, 0.6, 0.9],
        }
        profiler.profile_function(mc._update_q_table, test_return_dict)
    except Exception as e:
        print(f"Q表更新分析失败: {e}")

    # 4. 分析保存功能
    print("\n4. 分析保存功能性能...")
    try:
        profiler.profile_function(mc._saved_q_table)
    except Exception as e:
        print(f"保存功能分析失败: {e}")

    print("\n=== 性能分析完成 ===")


def analyze_specific_function():
    """分析特定函数的性能"""

    config = {
        "file_path": "./data",
        "saved_q_table_path": "./results/q_table.csv",
        "gamma": 0.99,
        "training_data_path": "./training_data",
        "training_market_data_path": "./market_data",
        "test_data_path": "./test_data",
        "test_market_data_path": "./test_market_data",
        "commission_value": 0.12,
    }

    mc = MCFromStart(config)

    print("=== 特定函数性能分析 ===\n")

    # 分析run方法（只运行一次迭代）
    print("分析run方法性能...")
    try:
        # 临时修改run方法，只运行一次
        original_run = mc.run

        def limited_run():
            before_update_q_table = mc._q_table.copy()
            mc._collect_data()
            mc._end_check(before_update_q_table, mc._q_table)
            mc._saved_q_table()

        mc.run = limited_run

        profiler = Profiler("run_method_profile.prof")
        profiler.profile_function(mc.run)

        # 恢复原始方法
        mc.run = original_run

    except Exception as e:
        print(f"run方法分析失败: {e}")


def quick_analysis():
    """快速性能分析"""

    config = {
        "file_path": "./data",
        "saved_q_table_path": "./results/q_table.csv",
        "gamma": 0.99,
        "training_data_path": "./training_data",
        "training_market_data_path": "./market_data",
        "test_data_path": "./test_data",
        "test_market_data_path": "./test_market_data",
        "commission_value": 0.12,
    }

    print("=== 快速性能分析 ===\n")

    # 分析初始化
    print("分析初始化性能...")
    result, stats = quick_profile(MCFromStart, config)

    print(f"总执行时间: {stats['total_time']:.4f} 秒")
    print(f"总函数调用次数: {stats['total_calls']}")

    print("\n最耗时的函数:")
    for i, func_info in enumerate(stats["top_functions"][:5]):
        print(f"{i+1}. {func_info['function']}")
        print(f"   累计时间: {func_info['cumulative_time']:.4f} 秒")
        print(f"   调用次数: {func_info['calls']}")
        print()


def analyze_ray_worker():
    """分析RayWorker的性能"""

    config = {
        "file_name": "test_file.csv",
        "init_state_index": 1,
        "init_holding": 0,
        "q_table": None,  # 将在下面设置
        "gamma": 0.99,
        "training_data_path": "./training_data",
        "training_market_data_path": "./market_data",
        "test_data_path": "./test_data",
        "test_market_data_path": "./test_market_data",
        "commission_value": 0.12,
        "init_action": 1,
    }

    # 创建MCFromStart实例来获取q_table
    mc_config = config.copy()
    mc_config["file_path"] = "./data"
    mc_config["saved_q_table_path"] = "./results/q_table.csv"

    mc = MCFromStart(mc_config)
    config["q_table"] = mc._q_table

    print("=== RayWorker性能分析 ===\n")

    from zyrl.mc import RayWorker

    # 创建RayWorker实例
    worker = RayWorker(config)

    # 分析run方法
    profiler = Profiler("ray_worker_profile.prof")

    print("分析RayWorker.run方法性能...")
    try:
        profiler.profile_function(worker.run, 1)  # 传入init_action=1
    except Exception as e:
        print(f"RayWorker分析失败: {e}")


if __name__ == "__main__":
    print("选择分析模式:")
    print("1. 分析MCFromStart各个方法")
    print("2. 分析特定函数")
    print("3. 快速分析")
    print("4. 分析RayWorker")
    print("5. 全部分析")

    choice = input("请输入选择 (1-5): ").strip()

    if choice == "1":
        analyze_mc_methods()
    elif choice == "2":
        analyze_specific_function()
    elif choice == "3":
        quick_analysis()
    elif choice == "4":
        analyze_ray_worker()
    elif choice == "5":
        print("执行全部分析...")
        analyze_mc_methods()
        analyze_specific_function()
        quick_analysis()
        analyze_ray_worker()
    else:
        print("无效选择，执行默认分析...")
        analyze_mc_methods()
