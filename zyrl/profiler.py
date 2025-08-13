#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用cProfile进行性能分析的工具
"""

import cProfile
import pstats
import io
import time
import functools
from typing import Callable, Any, Optional
import os


class Profiler:
    """性能分析器类"""

    def __init__(self, output_file: str = "profile_results.prof"):
        self.output_file = output_file
        self.profiler = cProfile.Profile()
        self.stats = None

    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        对单个函数进行性能分析

        Args:
            func: 要分析的函数
            *args, **kwargs: 函数的参数

        Returns:
            函数的返回值
        """
        print(f"开始分析函数: {func.__name__}")

        # 开始分析
        self.profiler.enable()
        start_time = time.time()

        # 执行函数
        result = func(*args, **kwargs)

        # 结束分析
        end_time = time.time()
        self.profiler.disable()

        # 计算总时间
        total_time = end_time - start_time
        print(f"函数 {func.__name__} 总执行时间: {total_time:.4f} 秒")

        # 保存分析结果
        self.profiler.dump_stats(self.output_file)

        # 分析统计信息
        self._analyze_stats()

        return result

    def _analyze_stats(self):
        """分析统计信息"""
        # 创建统计对象
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")

        # 打印前20个最耗时的函数
        print("\n=== 性能分析结果 (按累计时间排序) ===")
        ps.print_stats(20)

        # 打印调用次数最多的函数
        print("\n=== 调用次数最多的函数 ===")
        ps.sort_stats("calls")
        ps.print_stats(10)

        # 打印单次调用时间最长的函数
        print("\n=== 单次调用时间最长的函数 ===")
        ps.sort_stats("time")
        ps.print_stats(10)

        # 保存详细报告到文件
        with open("profile_detailed_report.txt", "w", encoding="utf-8") as f:
            ps.stream = f
            ps.sort_stats("cumulative")
            ps.print_stats()
            ps.print_callers()
            ps.print_callees()

        print(f"\n详细报告已保存到: profile_detailed_report.txt")

    def profile_class_method(self, obj: Any, method_name: str, *args, **kwargs) -> Any:
        """
        对类方法进行性能分析

        Args:
            obj: 对象实例
            method_name: 方法名
            *args, **kwargs: 方法参数

        Returns:
            方法的返回值
        """
        method = getattr(obj, method_name)
        return self.profile_function(method, *args, **kwargs)


def profile_decorator(output_file: Optional[str] = None):
    """
    装饰器：用于自动分析函数性能

    Args:
        output_file: 输出文件名，如果为None则使用函数名
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 确定输出文件名
            if output_file is None:
                filename = f"profile_{func.__name__}.prof"
            else:
                filename = output_file

            # 创建分析器
            profiler = Profiler(filename)

            # 执行分析
            return profiler.profile_function(func, *args, **kwargs)

        return wrapper

    return decorator


def quick_profile(func: Callable, *args, **kwargs) -> tuple[Any, dict]:
    """
    快速性能分析，返回函数结果和性能统计

    Args:
        func: 要分析的函数
        *args, **kwargs: 函数参数

    Returns:
        (函数结果, 性能统计字典)
    """
    profiler = cProfile.Profile()

    # 开始分析
    profiler.enable()
    start_time = time.time()

    # 执行函数
    result = func(*args, **kwargs)

    # 结束分析
    end_time = time.time()
    profiler.disable()

    # 获取统计信息
    stats = pstats.Stats(profiler)

    # 提取关键统计信息
    total_time = end_time - start_time
    total_calls = stats.total_calls

    # 获取最耗时的函数
    stats.sort_stats("cumulative")
    top_functions = []
    for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
        if tt > 0:  # 只包含有执行时间的函数
            top_functions.append(
                {
                    "function": func_name,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "calls": nc,
                }
            )

    # 按累计时间排序
    top_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)

    performance_stats = {
        "total_time": total_time,
        "total_calls": total_calls,
        "top_functions": top_functions[:10],  # 前10个最耗时的函数
    }

    return result, performance_stats


def analyze_mc_performance():
    """分析MCFromStart类的性能"""
    from zyrl.mc import MCFromStart

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

    # 创建实例
    mc = MCFromStart(config)

    # 分析各个方法的性能
    profiler = Profiler("mc_performance.prof")

    print("=== 分析MCFromStart类性能 ===\n")

    # 分析初始化
    print("1. 分析初始化性能...")
    profiler.profile_function(mc._init_q_table)

    # 分析数据收集
    print("\n2. 分析数据收集性能...")
    profiler.profile_function(mc._collect_data)

    # 分析Q表更新
    print("\n3. 分析Q表更新性能...")
    # 创建一个简单的return_dict用于测试
    test_return_dict = {(1, 0, 1): [0.5, 0.3], (1, 0, 0): [0.2]}
    profiler.profile_function(mc._update_q_table, test_return_dict)

    print("\n性能分析完成！")


if __name__ == "__main__":
    # 示例：分析MCFromStart性能
    analyze_mc_performance()
