# 分离架构启动指南

## 🏗️ 架构概述

新的分离架构将原来的单一推理脚本分为两个独立的服务：

1. **xROCS控制服务** (Python 3.8) - 负责机械臂的顺滑控制
2. **OpenPI推理服务** (Python 3.11) - 负责OpenPI推理和决策

两个服务通过文件接口进行通信，实现了：
- ✅ 顺滑的机械臂控制 (使用xROCS API)
- ✅ OpenPI推理兼容性 (Python 3.11)
- ✅ 解耦的架构设计

## 🚀 完整启动流程

### 前置服务 (保持不变)

**Terminal 1 - ROS核心:**
```bash
roscore
```

**Terminal 2 - 相机服务:**
```bash
roslaunch astra_camera list_devices.launch
roslaunch astra_camera multi_camera.launch
```

**Terminal 3 - Piper机器人服务:**
```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic/
bash can_multi_activate.sh

source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=True
```

**Terminal 4 - OpenPI模型服务:**
```bash
conda activate openpi;
python inference_piper/serve_trained_model.py \
     --checkpoint_dir checkpoints/lora_training/rgb_lora_sgd_lr1e-4/39999 \
     --config_name lora_training \
     --default_prompt "pick then long eggplant and place on the plant" \
     --port 8000 \
     --host 0.0.0.0
```

**Terminal 5 - 数据服务:**
```bash
conda deactivate; deactivate;
cd /home/agilex/code/opensource/openpi/inference_piper && \
python simple_memory_service.py
# python simple_data_service.py


```

### 新的分离架构服务

**Terminal 6 - xROCS控制服务 (新增):**
```bash
conda deactivate; deactivate;
cd /home/agilex/code/opensource/openpi/inference_piper && \
python xrocs_control_service.py
```

**Terminal 7 - OpenPI推理服务 (替换原Terminal 6):**
```bash
conda activate openpi && \
cd /home/agilex/code/opensource/openpi/inference_piper && \
python inference_mem_py311.py
```

## 🔄 启动顺序

1. **先启动前5个服务** (Terminal 1-5) - 按原流程
2. **启动xROCS控制服务** (Terminal 6) - 等待显示"ready"状态
3. **启动OpenPI推理服务** (Terminal 7) - 开始推理

## 📊 监控和调试

### 查看系统状态
```bash
# 查看控制接口文件
ls -la control_interface/

# 查看控制服务状态
cat control_interface/status.json

# 查看命令反馈
cat control_interface/feedback.json

# 查看最新命令
cat control_interface/command.json
```

### 实时监控
```bash
# 实时监控控制接口
watch -n 1 'ls -la control_interface/ && echo && cat control_interface/status.json 2>/dev/null'
```

### 日志输出示例

**xROCS控制服务启动成功:**
```
🤖 xROCS控制服务初始化
✅ 配置加载成功
✅ 机器人站创建成功
✅ 机器人连接成功
✅ 获取机器人句柄: ['right']
✅ 右臂已启用
🔄 开始监控控制命令...
```

**OpenPI推理服务运行:**
```
🤖 xROCS控制客户端初始化
📤 命令1: 关节命令已发送 (ID: 1)
✅ 命令1: 执行成功
🛤️ 使用插值轨迹 (变化0.251弧度)
```

## 🔧 故障排除

### 常见问题

1. **控制服务无法启动**
   - 检查是否在base环境 (Python 3.8)
   - 确认xROCS路径正确
   - 检查ROS和Piper服务是否正常

2. **推理服务连接失败**
   - 确认控制服务已启动
   - 检查control_interface目录权限
   - 查看status.json文件内容

3. **机械臂不动**
   - 检查控制服务日志
   - 确认机械臂已启用
   - 查看feedback.json中的错误信息

### 重启步骤

如果需要重启系统：
1. 停止推理服务 (Terminal 7: Ctrl+C)
2. 停止控制服务 (Terminal 6: Ctrl+C)
3. 清理接口文件: `rm -f control_interface/*.json control_interface/*.lock`
4. 重新启动控制服务和推理服务

## 🎯 优势

相比原来的直接ROS控制方案：

1. **顺滑控制**: 使用xROCS的`reach_target_joint()`实现真正的轨迹规划
2. **更好的兼容性**: 分离Python版本依赖
3. **更强的稳定性**: 解耦的架构，单个服务故障不影响整体
4. **更好的调试**: 独立的日志和状态监控
5. **插值优化**: 固定3步插值，>4度才触发

## 📝 配置文件

- **控制接口**: `control_interface/` 目录
- **xROCS配置**: `/home/agilex/code_repo/xCollector/xCollector/common/config/configuration_agilex2.toml`
- **Home位置**: 右臂 `[-0.038238353639841, -0.0, 0.0, 0.040487524122, 0, -0.096918866038322, 0.02]`
