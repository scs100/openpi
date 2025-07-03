# OpenPI LoRA 训练通用框架

这是一个通用的 OpenPI LoRA 微调训练框架，支持任意 LeRobot 格式数据集的高效训练。

## 功能特性

- 🚀 **高效数据加载**: 预加载数据到内存，大幅提升训练速度
- 🧠 **智能内存管理**: 主动监控和清理内存，避免 OOM 错误
- 🔧 **灵活配置**: 支持多种优化器和训练策略
- 📊 **自动归一化**: 自动计算和应用数据归一化统计
- 🎯 **LoRA 微调**: 专门优化的 LoRA 参数训练
- 📈 **WandB 集成**: 支持在线/离线训练监控

## 文件结构

```
openpi_lora_training/
├── lora_train_config.py      # 训练配置函数
├── fast_dataset.py           # 高效数据集加载器
├── lora_training_script.py   # 主训练脚本
├── compute_norm_stats.py     # 归一化统计计算
├── setup_swap_sudo.sh        # 系统内存优化脚本
└── README.md                 # 本文件
```

## 快速开始

### 1. 准备数据集

确保您的数据集是 LeRobot 格式，目录结构如下：
```
your_dataset/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   └── ...
│   └── ...
├── meta/
│   └── info.json
└── ...
```

### 2. 计算归一化统计

```bash
cd openpi_lora_training
python compute_norm_stats.py
# 按提示输入数据集路径和名称
```

### 3. 配置训练参数

编辑 `lora_training_script.py` 中的配置常量：

```python
# 数据配置
DATASET_PATH = "/path/to/your/dataset"
DATASET_NAME = "your_dataset"
DEFAULT_PROMPT = "your task description"

# 训练配置
BATCH_SIZE = 6
NUM_TRAIN_STEPS = 10000
PEAK_LR = 1e-4
```

### 4. 运行训练

```bash
python lora_training_script.py
```

## 配置选项

### 训练配置函数


### 使用自定义配置

```python
from lora_train_config import create_lora_finetune_config

config = create_lora_finetune_config(
    data_path="/path/to/your/dataset",
    dataset_name="your_dataset",
    prompt="your task description",
    exp_name="my_experiment",
    num_train_steps=15000
)
```

## 内存优化

框架包含多层内存优化机制：

1. **预加载优化**: 数据预加载到内存，避免重复 I/O
2. **Swap 监控**: 实时监控 swap 使用率，主动清理
3. **GPU 内存管理**: 智能 GPU 内存分配和清理
4. **检查点优化**: 内存安全的模型保存机制

### 系统优化脚本

```bash
# 设置 swap 权限（仅需运行一次）
sudo bash setup_swap_sudo.sh
```

## 故障排除

### 常见问题

1. **内存不足 (OOM)**
   - 降低 `BATCH_SIZE`
   - 减少 `preload_episodes`
   - 启用动态批量大小调整

2. **数据加载慢**
   - 确保数据在 SSD 上
   - 增加 `NUM_WORKERS`
   - 使用数据预加载

3. **训练不稳定**
   - 检查学习率设置
   - 验证数据归一化
   - 监控内存使用

### 调试模式

设置环境变量启用详细日志：
```bash
export JAX_TRACEBACK_FILTERING=off
python lora_training_script.py
```

## 性能优化建议

1. **硬件要求**
   - GPU: RTX 4090 24GB (推荐)
   - RAM: 32GB+ (推荐 64GB)
   - 存储: SSD (必需)

2. **批量大小选择**
   - RTX 4090: 批量大小 6-8
   - RTX 4060: 批量大小 2-4
   - 根据 GPU 内存动态调整

3. **训练策略**
   - 先用小数据集验证配置
   - 使用 50 步快速测试
   - 逐步增加训练规模

## 许可证

本项目遵循与 OpenPI 相同的许可证。
