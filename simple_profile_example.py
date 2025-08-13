#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的cProfile使用示例
"""

import cProfile
import pstats
import time
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from zyrl.mc import MCFromStart


def simple_profile_example():
    """简单的性能分析示例"""

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

    print("=== 简单性能分析示例 ===\n")

    # 创建分析器
    profiler = cProfile.Profile()

    # 开始分析
    profiler.enable()
    start_time = time.time()

    # 执行要分析的代码
    print("创建MCFromStart实例...")
    mc = MCFromStart(config)

    print("执行一次数据收集...")
    # 限制文件数量以加快分析
    original_files = mc._file_list
    mc._file_list = mc._file_list[:1] if mc._file_list else []

    mc._collect_data()

    # 恢复原始文件列表
    mc._file_list = original_files

    # 结束分析
    end_time = time.time()
    profiler.disable()

    # 计算总时间
    total_time = end_time - start_time
    print(f"\n总执行时间: {total_time:.4f} 秒")

    # 保存分析结果
    profiler.dump_stats("simple_profile.prof")

    # 分析统计信息
    stats = pstats.Stats(profiler)

    print("\n=== 性能分析结果 ===")
    print("按累计时间排序的前10个函数:")
    stats.sort_stats("cumulative")
    stats.print_stats(10)

    print("\n按调用次数排序的前10个函数:")
    stats.sort_stats("calls")
    stats.print_stats(10)

    print("\n按单次调用时间排序的前10个函数:")
    stats.sort_stats("time")
    stats.print_stats(10)

    # 保存详细报告
    with open("simple_profile_report.txt", "w", encoding="utf-8") as f:
        stats.stream = f
        stats.sort_stats("cumulative")
        stats.print_stats()
        stats.print_callers()

    print(f"\n详细报告已保存到: simple_profile_report.txt")


def profile_specific_method():
    """分析特定方法的性能"""

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

    print("=== 分析特定方法性能 ===\n")

    # 分析_update_q_table方法
    profiler = cProfile.Profile()

    profiler.enable()

    # 创建测试数据
    test_return_dict = {
        (1, 0, 1): [0.5, 0.3, 0.8],
        (1, 0, 0): [0.2, 0.4],
        (0, 1, -1): [0.1, 0.6, 0.9],
        (0, 1, 1): [0.7, 0.2],
        (-1, 2, -1): [0.3, 0.5, 0.1],
    }

    mc._update_q_table(test_return_dict)

    profiler.disable()

    # 分析结果
    stats = pstats.Stats(profiler)

    print("_update_q_table方法性能分析:")
    stats.sort_stats("cumulative")
    stats.print_stats(10)

    # 保存结果
    profiler.dump_stats("update_q_table_profile.prof")
    print("分析结果已保存到: update_q_table_profile.prof")


if __name__ == "__main__":
    print("选择分析模式:")
    print("1. 简单性能分析")
    print("2. 分析特定方法")
    print("3. 两种都执行")

    choice = input("请输入选择 (1-3): ").strip()

    if choice == "1":
        simple_profile_example()
    elif choice == "2":
        profile_specific_method()
    elif choice == "3":
        simple_profile_example()
        print("\n" + "=" * 50 + "\n")
        profile_specific_method()
    else:
        print("无效选择，执行简单分析...")
        simple_profile_example()
