#!/usr/bin/env python3
"""
修复版数据集加载器 - 适配tienkung机器人
- 相机：base_0_rgb (真实) + left_wrist_0_rgb/right_wrist_0_rgb (虚拟黑色图像)
- 状态/动作：完全自动识别维度，读取文件中的所有维度
- 自动填充到32维
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import cv2


class FixedDataset:
    """修复版数据集加载器 - 正确处理动作序列和数据格式"""
    
    def __init__(self, data_path: str, default_prompt: str = "perform the task",
                 preload_episodes: int = None, action_horizon: int = 50):
        self.data_path = Path(data_path)
        self.default_prompt = default_prompt
        self.action_horizon = action_horizon

        # 读取数据集的fps信息（仅用于显示）
        self.data_fps = self._read_dataset_fps()

        # 预加载数据到内存
        self._preload_data(preload_episodes)

        print(f"🚀 修复版数据集初始化完成:")
        print(f"  - 预加载 {len(self.preloaded_episodes)} 个episodes")
        print(f"  - 总样本数: {len(self.file_index)}")
        print(f"  - 数据fps: {self.data_fps}")
        print(f"  - 动作序列长度: {self.action_horizon}")
        print(f"  - 不进行帧率采样，使用原始时间序列")

    def _read_dataset_fps(self) -> float:
        """读取数据集的fps信息（仅用于显示）"""
        info_file = self.data_path / "meta" / "info.json"
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
                fps = info.get('fps', 100)
                print(f"📊 从 {info_file} 读取到数据fps: {fps}")
                return float(fps)
        except Exception as e:
            print(f"⚠️ 无法读取fps信息: {e}，使用默认值100")
            return 100.0

    def _preload_data(self, preload_episodes):
        """预加载数据到内存"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        parquet_files.sort()
        
        if preload_episodes is None:
            files_to_load = parquet_files
        else:
            files_to_load = parquet_files[:preload_episodes]
        
        print(f"🔄 预加载 {len(files_to_load)}/{len(parquet_files)} 个文件到内存...")
        
        self.preloaded_episodes = {}
        self.file_index = []
        action_horizon = self.action_horizon
        
        for i, file_path in enumerate(files_to_load):
            try:
                print(f"📂 加载 {i+1}/{len(files_to_load)}: {file_path.name}")
                
                episode_id = file_path.stem
                df = pd.read_parquet(file_path)
                self.preloaded_episodes[episode_id] = df
                
                # 确保有足够的数据构造动作序列
                num_rows = len(df)
                valid_samples = max(0, num_rows - action_horizon)
                
                for timestep in range(valid_samples):
                    self.file_index.append((episode_id, timestep))
                
                print(f"  ✅ {episode_id}: {num_rows} 行 → {valid_samples} 个有效样本")
                
            except Exception as e:
                print(f"❌ 加载失败 {file_path}: {e}")
        
        print(f"✅ 预加载完成: {len(self.file_index)} 个样本")

    def __len__(self):
        return len(self.file_index)

    def _process_image(self, img_data: Dict) -> np.ndarray:
        """正确处理图像数据"""
        if not isinstance(img_data, dict) or 'bytes' not in img_data:
            return np.zeros((224, 224, 3), dtype=np.float32)

        img_bytes = img_data['bytes']
        if not img_bytes:
            return np.zeros((224, 224, 3), dtype=np.float32)

        try:
            # 解码图像
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            #debug
           
            if img is None:
                return np.zeros((224, 224, 3), dtype=np.float32)
            #pkl2openpi  实现的rgb 不同转换
            # cv2.imdecode总是输出BGR格式，转换为RGB格式
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # cv2.imwrite(f"debugnocvt.jpg", img)
            # import pdb; pdb.set_trace()
            # 调整大小到224x224
            img = cv2.resize(img, (224, 224))

            # 归一化到[-1, 1]
            img_array = img.astype(np.float32) / 255.0
            img_array = img_array * 2.0 - 1.0
            
            return img_array
            
        except Exception as e:
            print(f"⚠️ 图像处理失败: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32)

    def _get_action_sequence(self, episode_df: pd.DataFrame, start_idx: int) -> np.ndarray:
        """构造连续动作序列（不进行帧率采样）"""
        action_horizon = self.action_horizon
        actions_sequence = []

        for step in range(action_horizon):
            # 计算目标帧索引 (从t+1时刻开始，连续N帧)
            target_idx = start_idx + 1 + step

            # 边界处理：如果超出范围，使用最后一帧
            if target_idx >= len(episode_df):
                target_idx = len(episode_df) - 1

            # 严格检查动作数据
            row_data = episode_df.iloc[target_idx]
            if "action" not in row_data:
                raise ValueError(f"❌ 动作数据缺失！episode {episode_df.name if hasattr(episode_df, 'name') else 'unknown'}, 行 {target_idx}, 可用列: {list(row_data.index)}")

            action_data = row_data["action"]
            if not isinstance(action_data, (list, np.ndarray)):
                raise ValueError(f"❌ 动作数据格式错误！期望list/array，得到: {type(action_data)}")

            if len(action_data) == 0:
                raise ValueError(f"❌ 动作数据为空！episode {episode_df.name if hasattr(episode_df, 'name') else 'unknown'}, 行 {target_idx}")

            # 读取文件中的所有action维度
            action_array = np.array(action_data, dtype=np.float32)
            # 取实际维度，最多32维
            action_actual = action_array[:min(len(action_array), 32)]

            if step == 0:  # 只在第一步打印调试信息
                print(f"🔍 DEBUG: 动作维度: {len(action_actual)}, 值范围: [{action_actual.min():.3f}, {action_actual.max():.3f}]")

            # 填充到32维
            action_32d = np.zeros(32, dtype=np.float32)
            action_32d[:len(action_actual)] = action_actual
            actions_sequence.append(action_32d)

        return np.array(actions_sequence, dtype=np.float32)

    def __getitem__(self, idx):
        print(f"🔍 DEBUG: __getitem__ 被调用，idx={idx}")
        if idx >= len(self.file_index):
            raise IndexError(f"索引 {idx} 超出范围 {len(self.file_index)}")

        episode_id, timestep = self.file_index[idx]
        episode_df = self.preloaded_episodes[episode_id]

        # 获取当前时刻的数据
        current_sample = episode_df.iloc[timestep].to_dict()
        print(f"🔍 DEBUG: 获取样本数据完成，episode_id={episode_id}, timestep={timestep}")
        
        # 处理图像 - 提供模型期望的所有相机
        images = {}

        # 处理图像数据 - 严格检查
        print(f"🔍 DEBUG: 开始处理图像，current_sample keys: {list(current_sample.keys())}")

        # 检查直接的图像键（如 'base_0_rgb'）
        base_image_found = False
        for key, value in current_sample.items():
            if 'base' in key.lower() and 'rgb' in key.lower():
                print(f"🔍 DEBUG: 找到直接图像键 {key}")
                print(f"🔍 DEBUG: 图像数据类型: {type(value)}")
                if isinstance(value, dict) and 'bytes' in value:
                    print(f"🔍 DEBUG: 处理字节格式图像")
                    img_array = self._process_image(value)
                    images['base_0_rgb'] = img_array
                    base_image_found = True
                    break
                else:
                    raise ValueError(f"❌ 图像数据格式错误！期望dict with 'bytes'，得到: {type(value)}, 内容: {value}")

        # 备用：检查 observation.images.* 格式
        if not base_image_found:
            for key, value in current_sample.items():
                if key.startswith("observation.images."):
                    img_name = key.replace("observation.images.", "")
                    print(f"🔍 DEBUG: 找到observation格式图像 {img_name}")
                    if 'base' in img_name.lower() or 'exterior' in img_name.lower():
                        print(f"🔍 DEBUG: 处理base相机 {img_name}")
                        img_array = self._process_image(value)
                        images['base_0_rgb'] = img_array
                        base_image_found = True
                        break

        # 如果没有找到base图像，报错
        if not base_image_found:
            raise ValueError(f"❌ 未找到base相机图像！可用的键: {list(current_sample.keys())}")

        # 为模型期望的其他相机提供黑色图像（不覆盖已有的真实相机）
        # 模型期望: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        required_cameras = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        for camera_name in required_cameras:
            if camera_name not in images:
                images[camera_name] = np.zeros((224, 224, 3), dtype=np.float32)

        # 处理状态 - 严格检查并自动识别维度
        print(f"🔍 DEBUG: 开始处理状态数据")
        state_data = None

        # 检查可能的状态键
        possible_state_keys = ["observation.state", "state", "robot_state"]
        for state_key in possible_state_keys:
            if state_key in current_sample:
                state_data = current_sample[state_key]
                print(f"🔍 DEBUG: 找到状态数据，键: {state_key}, 类型: {type(state_data)}")
                break

        if state_data is None:
            raise ValueError(f"❌ 未找到状态数据！期望键: {possible_state_keys}，可用键: {list(current_sample.keys())}")

        if isinstance(state_data, (list, np.ndarray)) and len(state_data) > 0:
            # 读取文件中的所有state维度
            state_array = np.array(state_data, dtype=np.float32)
            # 取实际维度，最多32维
            state_actual = state_array[:min(len(state_array), 32)]
            print(f"🔍 DEBUG: 状态维度: {len(state_actual)}")
        else:
            raise ValueError(f"❌ 状态数据格式错误！期望非空list/array，得到: {type(state_data)}, 长度: {len(state_data) if hasattr(state_data, '__len__') else 'N/A'}")

        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:len(state_actual)] = state_actual
        
        # 构造正确的50步动作序列
        actions_sequence = self._get_action_sequence(episode_df, timestep)
        
        # 创建图像mask - 只有base_0_rgb是真实的，其他是虚拟的
        image_masks = {
            'base_0_rgb': True,           # 真实相机
            'left_wrist_0_rgb': False,    # 虚拟相机（黑色图像）
            'right_wrist_0_rgb': False    # 虚拟相机（黑色图像）
        }
        
        # 验证数据质量
        if idx % 1000 == 0:
            print(f"🎯 样本 {idx}: episode={episode_id}, timestep={timestep}")
            print(f"   动作序列形状: {actions_sequence.shape}")
            print(f"   状态维度: {len(state_actual)} -> 32")
            print(f"   动作范围: [{actions_sequence.min():.3f}, {actions_sequence.max():.3f}]")
            print(f"   状态范围: [{state_32d.min():.3f}, {state_32d.max():.3f}]")
            print(f"   相机: {list(images.keys())}")
        
        return {
            "image": images,
            "image_mask": image_masks,
            "state": state_32d,
            "actions": actions_sequence,  # 正确的50步动作序列
        }

    def get_stats(self):
        """获取数据集统计信息"""
        if len(self.file_index) == 0:
            return {}
        
        # 采样一些数据计算统计
        sample_size = min(100, len(self.file_index))
        indices = np.random.choice(len(self.file_index), sample_size, replace=False)
        
        all_states = []
        all_actions = []
        
        for idx in indices:
            sample = self[idx]
            all_states.append(sample["state"])
            all_actions.append(sample["actions"])
        
        states = np.array(all_states)
        actions = np.array(all_actions)
        
        return {
            "state_mean": np.mean(states, axis=0),
            "state_std": np.std(states, axis=0),
            "action_mean": np.mean(actions, axis=(0, 1)),
            "action_std": np.std(actions, axis=(0, 1)),
            "state_range": [np.min(states), np.max(states)],
            "action_range": [np.min(actions), np.max(actions)],
        }


def create_fixed_dataset(data_path: str, default_prompt: str = "perform the task",
                        preload_episodes: int = None, action_horizon: int = 50):
    """创建修复版数据集"""
    return FixedDataset(data_path, default_prompt, preload_episodes, action_horizon)


if __name__ == "__main__":
    # 测试修复版数据集
    data_path = input("请输入数据集路径: ").strip()
    
    if os.path.exists(data_path):
        print("🧪 测试修复版数据集...")
        dataset = create_fixed_dataset(data_path, preload_episodes=1)
        
        print(f"📊 数据集大小: {len(dataset)}")
        
        # 测试几个样本
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n样本 {i}:")
            print(f"  图像数量: {len(sample['image'])}")
            print(f"  状态形状: {sample['state'].shape}")
            print(f"  动作形状: {sample['actions'].shape}")
            
        # 获取统计信息
        stats = dataset.get_stats()
        if stats:
            print(f"\n📈 数据统计:")
            print(f"  状态范围: {stats['state_range']}")
            print(f"  动作范围: {stats['action_range']}")
        
    else:
        print(f"❌ 数据路径不存在: {data_path}")
