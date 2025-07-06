#!/usr/bin/env python3
"""
为 LeRobot 格式数据集计算 norm stats
支持任意机器人数据集的归一化统计计算
"""

import numpy as np
import pandas as pd
import tqdm
from pathlib import Path
import openpi.shared.normalize as normalize

def load_dataset(data_path):
    """加载 LeRobot 格式数据集，提取所有维度的状态和动作数据"""
    data_path = Path(data_path)

    # 查找所有parquet文件
    parquet_files = list(data_path.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    print(f"Found {len(parquet_files)} episode files")

    all_states = []
    all_actions = []

    for file_path in tqdm.tqdm(parquet_files, desc="Loading episodes"):
        try:
            df = pd.read_parquet(file_path)

            # 提取状态和动作数据，提取所有维度
            states = np.array([np.array(state) for state in df['observation.state']])
            actions = np.array([np.array(action) for action in df['action']])

            all_states.append(states)
            all_actions.append(actions)

            print(f"Loaded {file_path.name}: {len(states)} steps")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # 合并所有数据
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    print(f"Total data: {len(all_states)} timesteps")
    print(f"State shape: {all_states.shape}")
    print(f"Action shape: {all_actions.shape}")

    return all_states, all_actions

def compute_norm_stats(states, actions, target_dim=32):
    """计算归一化统计信息"""

    # 为了与OpenPI兼容，需要将状态和动作都填充到32维
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    padded_states = np.zeros((states.shape[0], target_dim), dtype=states.dtype)
    padded_states[:, :state_dim] = states

    padded_actions = np.zeros((actions.shape[0], target_dim), dtype=actions.dtype)
    padded_actions[:, :action_dim] = actions
    
    # 创建统计计算器
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    
    print("Computing normalization statistics...")
    
    # 分批处理以避免内存问题
    batch_size = 1000
    num_batches = (len(states) + batch_size - 1) // batch_size
    
    for i in tqdm.tqdm(range(num_batches), desc="Computing stats"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(states))
        
        batch_states = padded_states[start_idx:end_idx]
        batch_actions = padded_actions[start_idx:end_idx]
        
        # 更新统计信息
        state_stats.update(batch_states)
        action_stats.update(batch_actions)
    
    # 获取最终统计信息
    state_norm_stats = state_stats.get_statistics()
    action_norm_stats = action_stats.get_statistics()
    
    return {
        "state": state_norm_stats,
        "actions": action_norm_stats
    }

def print_stats_summary(norm_stats):
    """打印统计信息摘要"""
    print("\n" + "="*50)
    print("NORMALIZATION STATISTICS SUMMARY")
    print("="*50)
    
    for key, stats in norm_stats.items():
        print(f"\n{key.upper()}:")
        print(f"  Shape: {stats.mean.shape}")
        print(f"  Mean range: [{stats.mean.min():.6f}, {stats.mean.max():.6f}]")
        print(f"  Std range: [{stats.std.min():.6f}, {stats.std.max():.6f}]")
        
        if stats.q01 is not None:
            print(f"  Q01 range: [{stats.q01.min():.6f}, {stats.q01.max():.6f}]")
            print(f"  Q99 range: [{stats.q99.min():.6f}, {stats.q99.max():.6f}]")
        
        # 显示前几个维度的详细信息
        print(f"  First 5 dimensions:")
        for i in range(min(5, len(stats.mean))):
            print(f"    Dim {i}: mean={stats.mean[i]:.6f}, std={stats.std[i]:.6f}")
        
        if key == "actions":
            # 自动检测有效的关节维度（非零标准差的维度）
            valid_joints = []
            for i in range(len(stats.std)):
                if stats.std[i] > 1e-6:  # 标准差大于阈值认为是有效维度
                    valid_joints.append(i)

            print(f"  Joint dimensions (0-{len(valid_joints)-1}, total {len(valid_joints)} joints):")
            for i in valid_joints:
                print(f"    Joint {i}: mean={stats.mean[i]:.6f}, std={stats.std[i]:.6f}")

def save_norm_stats(norm_stats, output_dir, dataset_name):
    """保存归一化统计信息"""
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    normalize.save(output_path, norm_stats)
    print(f"\nNorm stats saved to: {output_path}")
    print(f"Files created:")
    print(f"  - {output_path}/norm_stats.json")

def main(data_path=None, dataset_name=None):
    """主函数"""
    if data_path is None:
        data_path = input("请输入数据集路径: ").strip()
    if dataset_name is None:
        dataset_name = input("请输入数据集名称: ").strip()

    try:
        # 加载数据
        print(f"Loading dataset from {data_path}...")
        states, actions = load_dataset(data_path)

        # 计算统计信息
        print("\nComputing normalization statistics...")
        norm_stats = compute_norm_stats(states, actions)

        # 打印摘要
        print_stats_summary(norm_stats)

        # 保存统计信息
        save_norm_stats(norm_stats, "./assets", dataset_name)

        print("\n✅ Norm stats computation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
