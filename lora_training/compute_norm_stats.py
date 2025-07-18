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

# 导入ACTION_HORIZON配置
try:
    from start_lora_training import ACTION_HORIZON
    print(f"✅ 使用配置的ACTION_HORIZON: {ACTION_HORIZON}")
except ImportError:
    ACTION_HORIZON = 15  # 默认值
    print(f"⚠️ 无法导入ACTION_HORIZON，使用默认值: {ACTION_HORIZON}")

def load_dataset_with_sequences(data_path, action_horizon=None):
    """加载数据集，构造action sequences用于计算第N步的norm stats"""
    if action_horizon is None:
        action_horizon = ACTION_HORIZON
    print(f"Loading dataset from {data_path}...")
    print(f"Action horizon: {action_horizon}")

    # 获取所有parquet文件
    parquet_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    print(f"Found {len(parquet_files)} parquet files")

    all_states = []
    all_action_sequences = []

    for file_path in tqdm.tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pd.read_parquet(file_path)

            # 为每个有效的起始点构造action sequence
            for start_idx in range(len(df) - action_horizon):
                # 当前状态
                state = np.array(df.iloc[start_idx]['observation.state'])[:14]

                # 构造action sequence（第1步到第action_horizon步）
                action_sequence = []
                for step in range(action_horizon):
                    target_idx = start_idx + 1 + step
                    if target_idx < len(df):
                        action = np.array(df.iloc[target_idx]['action'])[:14]
                    else:
                        action = np.array(df.iloc[-1]['action'])[:14]  # 边界处理
                    action_sequence.append(action)

                all_states.append(state)
                all_action_sequences.append(action_sequence)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # 转换为numpy数组
    states = np.array(all_states)  # [N, 14]
    action_sequences = np.array(all_action_sequences)  # [N, action_horizon, 14]

    print(f"Loaded {len(states)} samples")
    print(f"State shape: {states.shape}")
    print(f"Action sequences shape: {action_sequences.shape}")

    return states, action_sequences

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

def apply_delta_actions_stepn(states, action_sequences, joint_mask, step=None):
    """应用delta actions转换，使用第N步的action计算delta

    这样计算的norm stats能覆盖最大的delta范围，适用于所有时间步
    """
    if step is None:
        step = ACTION_HORIZON - 1  # 使用最后一步
    print(f"应用delta actions转换，使用第{step+1}步的action")
    print(f"joint_mask: {joint_mask[:14]}")
    print(f"⚠️  注意：使用第{step+1}步的delta范围计算norm stats")

    # 提取第N步的actions
    actions_stepn = action_sequences[:, step, :]  # [N, 14]

    delta_actions = actions_stepn.copy()
    # 只使用前14维的mask（匹配实际数据维度）
    mask_14d = np.array(joint_mask[:14])  # 只取前14维

    print(f"实际数据维度: {actions_stepn.shape[1]}")
    print(f"使用的mask维度: {len(mask_14d)}")

    # 对每个样本应用delta转换：action[step] - current_state
    for i in range(len(actions_stepn)):
        state = states[i]  # [14]
        action = actions_stepn[i]  # [14]
        # 只对mask为True的维度应用delta转换
        delta_actions[i] = np.where(mask_14d, action - state, action)

    return delta_actions

def apply_delta_actions(states, actions, joint_mask):
    """应用delta actions转换，使用OpenPI的DeltaActions transform（原始版本）

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


def compute_norm_stats_stepn(data_path, dataset_name, action_horizon):
    """使用第N步的delta计算norm stats"""
    print(f"🎯 使用第{action_horizon}步的delta范围计算norm stats")

    # 定义关节mask（与lora_train_config.py中保持一致）
    joint_mask = [True] * 6 + [False] + [True] * 6 + [False] + [False] * 18

    try:
        # 加载带有action sequences的数据
        print(f"Loading dataset with action sequences from {data_path}...")
        states, action_sequences = load_dataset_with_sequences(data_path, action_horizon)

        # 使用第30步的delta计算统计信息
        print(f"\nComputing normalization statistics using step {action_horizon} delta...")

        # 填充states到32维
        padded_states = np.zeros((states.shape[0], 32), dtype=states.dtype)
        padded_states[:, :states.shape[1]] = states

        # 应用第N步的delta转换
        delta_actions_stepn = apply_delta_actions_stepn(states, action_sequences, joint_mask, step=action_horizon-1)

        # 填充到32维
        padded_actions = np.zeros((delta_actions_stepn.shape[0], 32), dtype=delta_actions_stepn.dtype)
        padded_actions[:, :delta_actions_stepn.shape[1]] = delta_actions_stepn

        # 使用OpenPI官方的RunningStats类
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        print(f"Computing normalization statistics using step {action_horizon} delta...")

        # 分批处理以避免内存问题
        batch_size = 1000
        num_batches = (len(padded_states) + batch_size - 1) // batch_size

        for i in tqdm.tqdm(range(num_batches), desc="Computing stats"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(padded_states))

            batch_states = padded_states[start_idx:end_idx]
            batch_actions = padded_actions[start_idx:end_idx]

            # 更新统计信息 - 使用官方方法
            stats["state"].update(batch_states)
            stats["actions"].update(batch_actions)

        # 获取最终统计信息 - 使用官方方法
        norm_stats = {key: stats_obj.get_statistics() for key, stats_obj in stats.items()}

        # 打印摘要
        print_stats_summary(norm_stats)

        # 保存统计信息
        suffix = f"_del_step{action_horizon}"
        save_norm_stats(norm_stats, "./assets", f"{dataset_name}{suffix}")

        print(f"\n✅ Step {action_horizon} norm stats computation completed successfully!")
        print(f"Mode: Delta actions using step {action_horizon} range")

        print(f"\n🎉 Step {action_horizon} norm stats计算完成！")
        print("现在可以使用新的norm stats进行训练了")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    # 测试第N步的norm stats计算
    data_path = "/home/testuser/data/pick_and_place_eggplant/openpi"
    dataset_name = "pick_and_place_eggplant"
    action_horizon = 15  # 可以修改为任意步数

    print(f"🎯 计算第{action_horizon}步的norm stats...")
    success = compute_norm_stats_stepn(data_path, dataset_name, action_horizon)

    if success:
        print(f"\n✅ 第{action_horizon}步norm stats计算完成！")
        print(f"文件保存为: assets/pick_and_place_eggplant_del_step{action_horizon}/norm_stats.json")
        print("\n下一步：修改训练配置使用新的norm stats")
    else:
        print("\n❌ 计算失败，请检查错误信息")
