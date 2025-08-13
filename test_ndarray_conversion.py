#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ndarray转换功能
"""

import numpy as np
import pandas as pd
from zyrl.mc import (
    convert_string_array_to_ndarray,
    convert_q_table_to_ndarray,
    convert_sample_count_to_ndarray,
)


def test_string_array_conversion():
    """测试字符串数组转换为ndarray"""
    print("=== 测试字符串数组转换 ===")

    # 您提供的数组
    string_array = np.array(
        [
            "[-2.24]",
            "[0.]",
            "[1.]",
            "[-0.62]",
            "[1.]",
            "[-1.24]",
            "[0.]",
            "[0.]",
            "[-1.]",
            "[0.]",
            "[-0.5]",
            "[-0.5]",
            "[0.]",
            "[1.]",
            "[-0.5]",
            "[-0.5]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[-0.24]",
            "[-1.24]",
            "[0.]",
            "[0.]",
            "[-1.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[-0.62]",
            "[1.]",
            "[-1.24]",
            "[0.]",
            "[0.]",
            "[-1.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[-0.62]",
            "[1.]",
            "[0.]",
            "[0.]",
            "[1.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[0.]",
            "[-2.24]",
            "[1.]",
            "[0.]",
            "[1.]",
            "[0.]",
            "[0.]",
            "[1.]",
            "[-0.62]",
        ],
        dtype=object,
    )

    print(f"原始数组类型: {type(string_array)}")
    print(f"原始数组形状: {string_array.shape}")
    print(f"原始数组前5个元素: {string_array[:5]}")

    # 转换为ndarray
    ndarray_result = convert_string_array_to_ndarray(string_array)

    print(f"\n转换后数组类型: {type(ndarray_result)}")
    print(f"转换后数组形状: {ndarray_result.shape}")
    print(f"转换后数组前5个元素: {ndarray_result[:5]}")
    print(f"转换后数组数据类型: {ndarray_result.dtype}")

    # 验证转换结果
    expected_first = -2.24
    actual_first = ndarray_result[0]
    print(f"\n验证第一个元素: 期望 {expected_first}, 实际 {actual_first}")
    assert abs(actual_first - expected_first) < 1e-6, "转换结果不正确"

    print("✅ 字符串数组转换测试通过\n")


def test_dataframe_conversion():
    """测试DataFrame转换为ndarray"""
    print("=== 测试DataFrame转换 ===")

    # 创建一个示例DataFrame
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]},
        index=["row1", "row2", "row3", "row4"],
    )

    print("原始DataFrame:")
    print(df)
    print(f"DataFrame形状: {df.shape}")

    # 转换为ndarray
    ndarray_result = convert_q_table_to_ndarray(df)

    print(f"\n转换后ndarray:")
    print(ndarray_result)
    print(f"ndarray形状: {ndarray_result.shape}")
    print(f"ndarray数据类型: {ndarray_result.dtype}")

    # 验证转换结果
    expected_shape = (4, 3)
    actual_shape = ndarray_result.shape
    print(f"\n验证形状: 期望 {expected_shape}, 实际 {actual_shape}")
    assert actual_shape == expected_shape, "转换后的形状不正确"

    print("✅ DataFrame转换测试通过\n")


def test_sample_count_conversion():
    """测试样本计数表转换"""
    print("=== 测试样本计数表转换 ===")

    # 创建一个示例样本计数表
    sample_count_df = pd.DataFrame(
        {"action1": [10, 20, 30], "action2": [15, 25, 35], "action3": [12, 22, 32]},
        index=[("holding1", "state1"), ("holding1", "state2"), ("holding2", "state1")],
    )

    print("原始样本计数表:")
    print(sample_count_df)
    print(f"样本计数表形状: {sample_count_df.shape}")

    # 转换为ndarray
    ndarray_result = convert_sample_count_to_ndarray(sample_count_df)

    print(f"\n转换后ndarray:")
    print(ndarray_result)
    print(f"ndarray形状: {ndarray_result.shape}")
    print(f"ndarray数据类型: {ndarray_result.dtype}")

    # 验证转换结果
    expected_shape = (3, 3)
    actual_shape = ndarray_result.shape
    print(f"\n验证形状: 期望 {expected_shape}, 实际 {actual_shape}")
    assert actual_shape == expected_shape, "转换后的形状不正确"

    print("✅ 样本计数表转换测试通过\n")


if __name__ == "__main__":
    print("开始测试ndarray转换功能...\n")

    try:
        test_string_array_conversion()
        test_dataframe_conversion()
        test_sample_count_conversion()
        print("🎉 所有测试都通过了！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
