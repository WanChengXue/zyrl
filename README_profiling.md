# 性能分析工具使用指南

本指南介绍如何使用cProfile来分析MCFromStart类的性能开销。

## 文件说明

### 1. `zyrl/profiler.py`
- 主要的性能分析工具类
- 提供多种分析方法和装饰器
- 支持详细报告生成

### 2. `analyze_mc_performance.py`
- 专门用于分析MCFromStart类性能的脚本
- 提供多种分析模式选择
- 可以分析不同方法的性能

### 3. `simple_profile_example.py`
- 简单的cProfile使用示例
- 适合初学者使用
- 提供基础的性能分析功能

## 使用方法

### 方法1：使用简单示例脚本

```bash
python simple_profile_example.py
```

这个脚本会：
1. 创建MCFromStart实例
2. 执行一次数据收集
3. 生成性能分析报告
4. 保存详细结果到文件

### 方法2：使用完整分析脚本

```bash
python analyze_mc_performance.py
```

选择分析模式：
- 1: 分析MCFromStart各个方法
- 2: 分析特定函数
- 3: 快速分析
- 4: 分析RayWorker
- 5: 全部分析

### 方法3：在代码中直接使用

```python
from zyrl.profiler import Profiler, quick_profile

# 创建分析器
profiler = Profiler("my_profile.prof")

# 分析函数
result = profiler.profile_function(your_function, *args, **kwargs)

# 或者使用快速分析
result, stats = quick_profile(your_function, *args, **kwargs)
```

### 方法4：使用装饰器

```python
from zyrl.profiler import profile_decorator

@profile_decorator("my_function_profile.prof")
def my_function():
    # 你的代码
    pass
```

## 输出文件说明

### 1. `.prof` 文件
- cProfile的原始数据文件
- 可以用其他工具（如snakeviz）可视化

### 2. `profile_detailed_report.txt`
- 详细的文本报告
- 包含函数调用关系
- 按不同排序方式显示结果

### 3. 控制台输出
- 实时显示分析进度
- 显示最耗时的函数
- 显示总执行时间

## 分析结果解读

### 关键指标

1. **累计时间 (cumulative time)**
   - 函数及其所有子函数的总执行时间
   - 最重要的性能指标

2. **单次时间 (time)**
   - 函数本身的执行时间（不包括子函数）
   - 用于识别函数内部的开销

3. **调用次数 (calls)**
   - 函数被调用的次数
   - 用于识别频繁调用的函数

### 常见性能瓶颈

1. **Ray相关操作**
   - `ray.get()` 和 `ray.remote()` 调用
   - 并行计算的开销

2. **DataFrame操作**
   - pandas的索引和切片操作
   - 大量数据的读写操作

3. **文件I/O**
   - CSV文件的读写
   - 数据文件的加载

## 优化建议

### 1. 减少Ray调用开销
```python
# 批量处理而不是逐个处理
worker_list = [ray_worker.remote(config) for config in configs]
results = ray.get(worker_list)
```

### 2. 优化DataFrame操作
```python
# 使用向量化操作而不是循环
df.loc[condition] = value  # 而不是逐行赋值
```

### 3. 减少文件I/O
```python
# 批量读写文件
# 使用内存缓存
# 考虑使用更高效的文件格式（如parquet）
```

## 可视化分析结果

### 使用snakeviz（推荐）

```bash
# 安装snakeviz
pip install snakeviz

# 可视化.prof文件
snakeviz profile_results.prof
```

### 使用其他工具

```bash
# 使用gprof2dot
pip install gprof2dot
gprof2dot -f pstats profile_results.prof | dot -Tpng -o profile.png
```

## 注意事项

1. **数据量控制**
   - 分析时使用较小的数据集
   - 避免分析整个训练过程

2. **环境一致性**
   - 在相同的硬件环境下进行分析
   - 关闭其他不必要的程序

3. **多次运行**
   - 运行多次取平均值
   - 考虑系统负载的影响

4. **内存使用**
   - 注意内存使用情况
   - 避免内存泄漏

## 示例输出

```
=== 性能分析结果 (按累计时间排序) ===
         123456 function calls in 12.345 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1000    5.234    0.005   10.123    0.010 zyrl/mc.py:125(_collect_data)
     5000    2.345    0.000    4.567    0.001 zyrl/mc.py:196(run)
     2000    1.234    0.001    2.345    0.001 zyrl/mc.py:102(_update_q_table)
```

这个输出显示：
- `_collect_data` 是最耗时的函数（10.123秒）
- `run` 方法调用了5000次
- `_update_q_table` 每次调用平均耗时0.001秒
