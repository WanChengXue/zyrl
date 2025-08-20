# ZYRL - 强化学习中间层框架

ZYRL 是一个基于 Tianshou 的强化学习中间层框架，提供了简洁易用的接口来配置和训练强化学习模型。

## 特性

- 🚀 **统一的Pipeline接口**：一个类支持所有使用模式，通过配置自动选择
- 🔧 **配置驱动设计**：通过 YAML/JSON 配置文件轻松设置训练参数和运行模式
- 🎯 **多种算法支持**：支持 DQN、PPO 等主流强化学习算法
- 📊 **完整的评估工具**：内置基准测试和性能评估功能
- 💾 **检查点管理**：支持模型保存和恢复训练
- 📈 **可视化日志**：集成 TensorBoard 日志记录
- 🔄 **模式切换**：支持运行时动态切换训练模式

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 创建配置文件

```yaml
# config.yaml
pipeline:
  mode: "standard"  # 运行模式: standard, high_level, low_level
  auto_setup: true  # 是否自动设置环境

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

trainer:
  type: "offpolicy"
  max_epoch: 100
  step_per_epoch: 1000
  batch_size: 64
```

### 2. 统一使用方式

```python
from zyrl.pipeline import Pipeline

# 创建pipeline（模式由配置文件决定）
pipeline = Pipeline("CartPole-v1", "config.yaml")

# 查看当前模式
print(f"当前模式: {pipeline.get_mode()}")

# 训练（根据模式自动选择训练方式）
result = pipeline.fit()

# 测试
test_result = pipeline.test(num_episodes=10)
```

### 3. 不同模式的使用

#### 标准模式 (standard)
```python
# 使用标准配置
pipeline = Pipeline("CartPole-v1", "config_standard.yaml")
result = pipeline.fit(max_epoch=100)
```

#### 高级模式 (high_level)
```python
# 使用高级模式配置
pipeline = Pipeline("CartPole-v1", "config_high_level.yaml")
# 高级模式会自动训练和评估
result = pipeline.fit(max_epoch=100, eval_episodes=10)
```

#### 底层模式 (low_level)
```python
# 使用底层模式配置
pipeline = Pipeline("CartPole-v1", "config_low_level.yaml")
# 底层模式提供细粒度控制
result = pipeline.fit(num_envs=2)

# 也可以手动创建组件
env = pipeline.create_environment()
network = pipeline.create_network()
policy = pipeline.create_policy(network)
```

### 4. 运行时模式切换

```python
pipeline = Pipeline("CartPole-v1", "config.yaml")

# 切换到高级模式
pipeline.set_mode("high_level")
result = pipeline.fit(max_epoch=50, eval_episodes=5)

# 切换到底层模式
pipeline.set_mode("low_level")
result = pipeline.fit(num_envs=1)
```

## 配置说明

### Pipeline配置

```yaml
pipeline:
  mode: "standard"        # 运行模式: standard, high_level, low_level
  auto_setup: true        # 是否自动设置环境
```

### 环境配置

```yaml
env:
  num_envs: 4          # 并行环境数量
  vector_type: "dummy" # 向量化类型: dummy 或 subproc
```

### 网络配置

```yaml
network:
  type: "mlp"                    # 网络类型: mlp 或 cnn
  hidden_sizes: [256, 256]       # 隐藏层大小（维度会自动检测）
  activation: "relu"             # 激活函数: relu 或 tanh
```

### 策略配置

```yaml
policy:
  type: "dqn"                    # 策略类型: dqn 或 ppo
  lr: 0.001                      # 学习率
  gamma: 0.99                    # 折扣因子
  n_step: 1                      # n步回报
  target_update_freq: 500        # 目标网络更新频率
```

### 训练器配置

```yaml
trainer:
  type: "offpolicy"              # 训练器类型: offpolicy 或 onpolicy
  max_epoch: 100                 # 最大训练轮数
  step_per_epoch: 1000           # 每轮步数
  step_per_collect: 10           # 每次收集步数
  update_per_step: 0.1           # 每步更新次数
  batch_size: 64                 # 批次大小
```

## 运行模式详解

### 标准模式 (standard)
- **特点**: 平衡的易用性和控制性
- **适用场景**: 大多数训练任务
- **行为**: 使用标准训练流程，需要手动调用测试

### 高级模式 (high_level)
- **特点**: 最大化的易用性
- **适用场景**: 快速原型开发和实验
- **行为**: 自动训练和评估，返回完整结果

### 底层模式 (low_level)
- **特点**: 最大化的控制性
- **适用场景**: 需要精细控制的复杂任务
- **行为**: 手动控制每个组件，提供最大灵活性

## 高级功能

### 配置管理

```python
# 获取当前配置
config = pipeline.get_config()

# 更新配置
pipeline.update_config("trainer.max_epoch", 200)
pipeline.update_config("trainer.batch_size", 128)

# 查看更新后的配置
updated_config = pipeline.get_config()
```

### 检查点管理

```python
# 保存检查点
pipeline.save_checkpoint("model.pth")

# 加载检查点
pipeline.load_checkpoint("model.pth")

# 恢复训练
pipeline.resume_training("model.pth", max_epoch=50)
```

### 基准测试

```python
# 运行基准测试
benchmark_result = pipeline.benchmark(num_runs=5, episodes_per_run=10)
print(f"平均奖励: {benchmark_result['mean_reward']}")
print(f"标准差: {benchmark_result['std_reward']}")
```

### 自定义环境

```python
import gymnasium as gym

class CustomEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))

    def reset(self, seed=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        reward = 1.0 if action == 0 else 0.5
        return self.observation_space.sample(), reward, False, False, {}

# 使用自定义环境
custom_env = CustomEnv()
pipeline = Pipeline(custom_env, "config.yaml")
```

### 渲染环境

```python
# 渲染训练好的模型
pipeline.render(num_episodes=3)
```

## 架构设计

```
┌─────────────────┐
│   Pipeline      │  ← 统一接口：根据配置自动选择模式
│   (Unified)     │
└─────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│ High  │ │ Low   │  ← 不同模式：高级/底层
│ Level │ │ Level │
└───────┘ └───────┘
         │
┌─────────────────┐
│   Tianshou      │  ← 中间层：连接配置和底层训练
│   Interface     │
└─────────────────┘
         │
┌─────────────────┐
│   Tianshou      │  ← 底层：实际的强化学习库
│   Framework     │
└─────────────────┘
```

## 示例

查看 `examples/` 目录中的完整示例：

- `usage_example.py` - 完整的使用示例
- `config_example.yaml` - 标准模式配置
- `config_high_level.yaml` - 高级模式配置
- `config_low_level.yaml` - 底层模式配置

运行示例：

```bash
python examples/usage_example.py
```

## 向后兼容性

为了保持向后兼容性，原有的类名仍然可用：

```python
# 这些类仍然可以正常使用
from zyrl.pipeline import HighLevelPipeline, LowLevelPipeline

hl_pipeline = HighLevelPipeline("CartPole-v1", "config.yaml")
ll_pipeline = LowLevelPipeline("CartPole-v1", "config.yaml")
```

## 依赖

- Python 3.8+
- PyTorch
- Tianshou
- Gymnasium
- PyYAML
- TensorBoard

## 许可证

MIT License
