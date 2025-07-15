#!/usr/bin/env python3
"""
使用OpenPI官方方法计算normalization statistics
支持OpenPI官方推荐的delta actions模式（action - current_state）
"""

import os
import sys
import numpy as np
import pandas as pd
import tqdm
from pathlib import Path

# 添加OpenPI路径
sys.path.insert(0, '/home/testuser/code/opensource/openpi/src')

import openpi.shared.normalize as normalize

def load_dataset(data_path: str):
    """加载数据集"""
    print(f"Loading dataset from {data_path}...")

    # 获取所有parquet文件
    parquet_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    print(f"Found {len(parquet_files)} parquet files")

    all_states = []
    all_actions = []

    for file_path in tqdm.tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pd.read_parquet(file_path)

            # 提取状态和动作 - 修正数据处理方式
            states = np.array([np.array(state) for state in df['observation.state']])
            actions = np.array([np.array(action) for action in df['action']])

            all_states.append(states)
            all_actions.append(actions)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # 合并所有数据
    states = np.vstack(all_states)
    actions = np.vstack(all_actions)

    print(f"Loaded {len(states)} samples")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    return states, actions

def apply_delta_actions(states, actions, joint_mask):
    """应用delta actions转换，使用OpenPI的DeltaActions transform

    OpenPI的DeltaActions计算公式：delta_action = action - current_state
    这表示从当前状态到目标动作的偏移量，而不是时间序列差分
    """
    print(f"应用delta actions转换，joint_mask: {joint_mask[:14]}")
    print("⚠️  注意：使用OpenPI官方逻辑 delta_action = action - current_state")

    delta_actions = actions.copy()
    mask = np.array(joint_mask)
    dims = len(mask)

    # 对每个样本应用delta转换：action[mask] -= state[mask]
    for i in range(len(actions)):
        state = states[i]
        action = actions[i]
        # 只对mask为True的维度应用delta转换
        delta_actions[i, :dims] = np.where(mask, action[:dims] - state[:dims], action[:dims])

    return delta_actions

def compute_norm_stats(states, actions, target_dim=32, use_delta_actions=False, joint_mask=None):
    """使用OpenPI官方方法计算归一化统计信息"""

    # 为了与OpenPI兼容，需要将状态和动作都填充到32维
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    padded_states = np.zeros((states.shape[0], target_dim), dtype=states.dtype)
    padded_states[:, :state_dim] = states

    padded_actions = np.zeros((actions.shape[0], target_dim), dtype=actions.dtype)
    padded_actions[:, :action_dim] = actions

    # 如果启用delta actions模式，应用delta转换
    if use_delta_actions and joint_mask is not None:
        padded_actions = apply_delta_actions(padded_states, padded_actions, joint_mask)
        print("✅ 已应用delta actions转换")

    # 使用OpenPI官方的RunningStats类
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    print("Computing normalization statistics using OpenPI official method...")

    # 分批处理以避免内存问题
    batch_size = 1000
    num_batches = (len(states) + batch_size - 1) // batch_size

    for i in tqdm.tqdm(range(num_batches), desc="Computing stats"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(states))

        batch_states = padded_states[start_idx:end_idx]
        batch_actions = padded_actions[start_idx:end_idx]

        # 更新统计信息 - 使用官方方法
        stats["state"].update(batch_states)
        stats["actions"].update(batch_actions)

    # 获取最终统计信息 - 使用官方方法
    norm_stats = {key: stats_obj.get_statistics() for key, stats_obj in stats.items()}

    return norm_stats

def print_stats_summary(norm_stats):
    """打印统计信息摘要"""
    print("\n" + "="*50)
    print("NORMALIZATION STATISTICS SUMMARY (OFFICIAL METHOD)")
    print("="*50)

    for key, stats in norm_stats.items():
        print(f"\n{key.upper()}:")
        print(f"  Shape: {stats.mean.shape}")
        print(f"  Mean range: [{np.min(stats.mean):.6f}, {np.max(stats.mean):.6f}]")
        print(f"  Std range: [{np.min(stats.std):.6f}, {np.max(stats.std):.6f}]")

        if stats.q01 is not None:
            print(f"  Q01 range: [{np.min(stats.q01):.6f}, {np.max(stats.q01):.6f}]")
        if stats.q99 is not None:
            print(f"  Q99 range: [{np.min(stats.q99):.6f}, {np.max(stats.q99):.6f}]")

        print(f"  First 5 dimensions:")
        for i in range(min(5, len(stats.mean))):
            print(f"    Dim {i}: mean={stats.mean[i]:.6f}, std={stats.std[i]:.6f}")

        # 如果是actions，显示关节维度信息
        if key == "actions":
            print(f"  Joint dimensions (0-13, total 14 joints):")
            for i in range(min(14, len(stats.mean))):
                print(f"    Joint {i}: mean={stats.mean[i]:.6f}, std={stats.std[i]:.6f}")

def save_norm_stats(norm_stats, base_dir, dataset_name):
    """使用OpenPI官方方法保存统计信息"""
    output_dir = Path(base_dir) / dataset_name

    print(f"\nNorm stats saved to: {output_dir}")

    # 使用OpenPI官方的save函数
    normalize.save(output_dir, norm_stats)

    print("Files created:")
    print(f"  - {output_dir}/norm_stats.json")


def main(data_path=None, dataset_name=None, use_delta_actions=True):
    """主函数"""
    # 如果没有提供参数，使用默认配置
    if data_path is None:
        data_path = "/home/testuser/data/pass_drink/openpi"
    if dataset_name is None:
        dataset_name = "pass_drink"

    # 定义关节mask（与lora_train_config.py中保持一致）
    joint_mask = [True] * 6 + [False] + [True] * 6 + [False] + [False] * 18

    try:
        # 加载数据
        print(f"Loading dataset from {data_path}...")
        states, actions = load_dataset(data_path)

        # 使用OpenPI官方方法计算统计信息
        print(f"\nComputing normalization statistics using OpenPI official method...")
        print(f"(delta_actions={use_delta_actions})")

        norm_stats = compute_norm_stats(
            states, actions,
            use_delta_actions=use_delta_actions,
            joint_mask=joint_mask if use_delta_actions else None
        )

        # 打印摘要
        print_stats_summary(norm_stats)

        # 保存统计信息
        suffix = "_del" if use_delta_actions else "_abs"
        save_norm_stats(norm_stats, "./assets", f"{dataset_name}{suffix}")

        print(f"\n✅ Norm stats computation completed successfully using OpenPI official method!")
        mode_desc = "Delta actions (action - current_state)" if use_delta_actions else "Absolute values"
        print(f"Mode: {mode_desc}")

        print(f"\n🎉 Official method norm stats计算完成！")
        print("现在可以使用新的norm stats进行训练了")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
