# OpenPI Inference Guide

本指南介绍如何启动 OpenPI 推理服务端和客户端，并进行完整 episode 的推理与可视化。

---

## 1. 启动推理服务端

首先，确保你已经训练好模型并有对应的 checkpoint。服务端会自动加载最新的 checkpoint。

在项目根目录下运行：

```bash
python inference_piper/serve_trained_model.py
```

- 默认监听 `0.0.0.0:8000`
- 会自动查找 `checkpoints/sgd_swap_manager/sgd_swap_manager_norm/` 下最新的 checkpoint 目录
- 日志会显示模型、端口、checkpoint 路径等信息

如需自定义端口或 checkpoint 路径，请修改 `serve_trained_model.py` 中的 `TrainedModelConfig`。

---

## 2. 启动推理客户端（推理+可视化）

在项目根目录下运行：

```bash
python inference_piper/inference_full_episode_visualization.py --debug
```

### 常用参数说明

- `--host`：推理服务器主机（默认 `localhost`）
- `--port`：推理服务器端口（默认 `8000`）
- `--data_path`：parquet 数据路径（默认 `/home/q/data/pick_and_place_eggplant/openpi`）
- `--episode`：分析第几个 episode（默认 `0`）
- `--step_limit`：推理步数限制（默认 `20`，`0` 表示不限制）
- `--output`：输出图片文件名（默认 `test_inference.png`）
- `--future_steps`：每步预测未来多少步（默认 `30`，只画 t+1）
- `--debug`：打印详细日志

### 典型命令示例

```bash
python inference_piper/inference_full_episode_visualization.py \
    --host localhost \
    --port 8000 \
    --data_path /home/q/data/pick_and_place_eggplant/openpi \
    --episode 0 \
    --step_limit 20 \
    --output my_infer.png \
    --future_steps 30 \
    --debug
```

---

## 3. 输出说明

- 运行结束后，会在当前目录生成可视化图片（如 `test_inference.png`），展示每个关节的全剧集 t+1 预测曲线。
- 控制台会输出每步推理的统计信息。

---

## 4. 常见问题与排查

- **服务端未启动或端口不对**：客户端会提示无法连接。
- **parquet 数据路径不对**：客户端会提示无法加载地面真值数据。
- **图片 key 不匹配**：客户端已自动做 key 映射，无需手动处理。
- **预测维度不一致**：客户端已自动截取前 14 维，无需手动处理。
- **image_mask 字段**：当前服务端已兼容可选 image_mask，客户端无需传递。

---

如有其它问题，请查阅代码注释或联系开发者。 