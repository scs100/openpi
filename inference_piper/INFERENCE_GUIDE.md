# OpenPI 推理环境配置指南

## 🎯 概述

本指南将帮助你配置OpenPI推理环境，并使用训练好的模型进行开环推理预测。

## 📋 环境要求

### 硬件要求
- **GPU**: NVIDIA RTX 4090 (24GB) 或更高
  - 推理需要约 17GB GPU 内存
  - RTX 4060 (8GB) 可能无法运行，建议使用更大显存的GPU
- **系统内存**: 建议 32GB 或更多
- **存储**: 至少 10GB 可用空间用于模型和缓存

### 软件要求
- **Python**: 3.11+ (OpenPI要求)
- **CUDA**: 12.x
- **操作系统**: Linux (推荐 Ubuntu 20.04+)

## 🔧 环境配置

### 1. 激活OpenPI环境
```bash
cd /home/testuser/code/opensource/openpi
conda activate openpi
```

### 2. 设置环境变量
```bash
# GPU内存配置 - 防止内存溢出
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 可选：设置缓存目录
export OPENPI_DATA_HOME=/path/to/custom/cache
```

### 3. 验证环境
```bash
# 检查GPU状态
nvidia-smi

# 检查JAX GPU支持
python -c "import jax; print('GPU devices:', jax.devices())"

# 检查PyTorch GPU支持  
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🚀 推理方式

### 方式1: 直接推理测试 (推荐用于快速测试)

```bash
# 运行直接推理测试
python inference_test.py
```

**特点:**
- 直接加载模型进行推理
- 适合快速验证模型功能
- 包含详细的性能统计

### 方式2: 服务器-客户端模式 (推荐用于生产环境)

**启动推理服务器:**
```bash
# 终端1: 启动推理服务器
python serve_trained_model.py
```

**运行客户端测试:**
```bash
# 终端2: 运行客户端测试
python test_trained_model_client.py --steps 10 --obs_type aloha
```

**特点:**
- 服务器-客户端分离架构
- 支持远程推理
- 适合集成到机器人系统中

### 方式3: 使用OpenPI官方工具

**启动官方推理服务器:**
```bash
# 使用你训练的模型启动服务器
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=sgd_swap_manager \
  --policy.dir=checkpoints/sgd_swap_manager/sgd_swap_manager_20k_production/19999
```

**使用官方客户端测试:**
```bash
# 使用官方简单客户端
python examples/simple_client/main.py --env aloha --num_steps 10
```

## 📊 性能预期

基于RTX 4090的性能数据:
- **推理时间**: ~319ms 平均
- **推理频率**: ~3.13 Hz
- **GPU内存使用**: ~17GB (70%)
- **GPU利用率**: 推理时100%，空闲时2%

## 🤖 机器人配置适配

### 你的机器人配置
- **动作维度**: 14维 (左臂6+夹爪1, 右臂6+夹爪1)
- **相机配置**:
  - `exterior_image_1_left` → `base_0_rgb`
  - `wrist_image_left` → `left_wrist_0_rgb`  
  - `wrist_image_right` → `right_wrist_0_rgb`
- **控制频率**: 100fps (但图像实际33.3fps)

### 动作填充
OpenPI要求32维动作，你的14维动作需要填充:
```python
# 你的14维动作
robot_action = np.array([...])  # shape: (14,)

# 填充到32维
openpi_action = np.zeros(32)
openpi_action[:14] = robot_action  # 前14维是你的动作
# 后18维保持为0
```

## 🔍 开环推理测试

### 测试流程
1. **模型加载**: 加载训练好的检查点
2. **观测生成**: 创建虚拟或真实观测数据
3. **推理执行**: 调用模型进行动作预测
4. **结果分析**: 分析动作输出和性能

### 观测数据格式
```python
obs = {
    "state": np.array([...]),  # 14维关节状态
    "images": {
        "cam_high": np.array([...]),      # (3, 224, 224) uint8
        "cam_low": np.array([...]),       # (3, 224, 224) uint8  
        "cam_left_wrist": np.array([...]), # (3, 224, 224) uint8
        "cam_right_wrist": np.array([...]) # (3, 224, 224) uint8
    },
    "prompt": "pick and place purple long eggplant"
}
```

### 动作输出格式
```python
result = {
    "actions": np.array([...]),  # (action_horizon, 32) - 动作序列
    "server_timing": {...},      # 服务器时间统计
    "policy_timing": {...}       # 策略时间统计
}
```

## 🛠️ 故障排除

### 常见问题

**1. GPU内存不足**
```bash
# 解决方案：降低内存分配
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

**2. 检查点加载失败**
```bash
# 检查可用检查点
ls -la checkpoints/sgd_swap_manager/sgd_swap_manager_20k_production/
```

**3. 连接服务器失败**
```bash
# 检查服务器是否运行
netstat -tlnp | grep 8000

# 检查防火墙
sudo ufw status
```

**4. 推理速度慢**
- 确保使用GPU而非CPU
- 检查GPU利用率: `nvidia-smi`
- 考虑使用更小的batch size

### 日志调试
```bash
# 启用详细日志
export PYTHONPATH=$PWD:$PYTHONPATH
python -u your_script.py 2>&1 | tee inference.log
```

## 📈 性能优化建议

1. **内存优化**:
   - 使用适当的GPU内存分配比例
   - 避免预分配内存

2. **推理优化**:
   - 预热模型（前几次推理较慢）
   - 批量处理多个观测

3. **系统优化**:
   - 使用tmux保持会话稳定
   - 监控系统资源使用

## 🎯 下一步

1. **验证推理**: 运行测试脚本验证模型功能
2. **集成机器人**: 将推理集成到你的机器人控制系统
3. **性能调优**: 根据实际需求调整推理参数
4. **部署优化**: 考虑使用Docker或其他部署方案

## 📞 支持

如果遇到问题，请检查:
1. 环境配置是否正确
2. 检查点文件是否完整
3. GPU内存和系统资源是否充足
4. 网络连接是否正常（如使用远程推理）
