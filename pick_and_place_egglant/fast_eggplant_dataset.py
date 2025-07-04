#!/usr/bin/env python3
"""
快速茄子数据集加载器 - 专门优化训练速度
解决每步146秒的性能问题
"""

import os
import gc
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image


class FastEggplantDataset:
    """快速茄子数据集加载器 - 预加载关键数据，最大化训练速度"""
    
    def __init__(self, data_path: str, default_prompt: str = "pick and place purple long eggplant",
                 training_fps: float = 33.3, preload_episodes: int = None):
        self.data_path = Path(data_path)
        self.default_prompt = default_prompt
        self.training_fps = training_fps

        # 自动检测所有parquet文件数量
        if preload_episodes is None:
            parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
            self.preload_episodes = len(parquet_files)
            print(f"🔍 自动检测到 {self.preload_episodes} 个parquet文件，将全部加载")
        else:
            self.preload_episodes = preload_episodes  # 预加载的episode数量
        
        # 读取数据集的原始fps信息
        self.original_data_fps = self._read_dataset_fps()
        
        # 预加载数据到内存
        self._preload_data()
        
        print(f"🚀 快速数据集初始化完成:")
        print(f"  - 预加载 {len(self.preloaded_episodes)} 个episodes")
        print(f"  - 总样本数: {len(self.file_index)}")
        print(f"  - 训练fps: {self.training_fps}")

    def _read_dataset_fps(self) -> float:
        """读取数据集的原始fps"""
        info_file = self.data_path / "meta" / "info.json"
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
                fps = info.get('fps', 100)
                print(f"📊 从 {info_file} 读取到原始数据fps: {fps}")
                return float(fps)
        except Exception as e:
            print(f"⚠️ 无法读取fps信息: {e}，使用默认值100")
            return 100.0

    def _preload_data(self):
        """预加载数据到内存 - 加载指定数量的episodes"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        parquet_files.sort()

        # 限制预加载的文件数量（如果指定了数量）
        files_to_load = parquet_files[:self.preload_episodes] if self.preload_episodes < len(parquet_files) else parquet_files
        print(f"🔄 预加载 {len(files_to_load)}/{len(parquet_files)} 个文件到内存...")
        
        self.preloaded_episodes = {}  # {episode_id: DataFrame}
        self.file_index = []  # [(episode_id, timestep_in_episode), ...]
        total_samples = 0
        
        for i, file_path in enumerate(files_to_load):
            try:
                print(f"📂 加载 {i+1}/{len(files_to_load)}: {file_path.name}")
                
                # 从文件名提取episode信息
                episode_id = file_path.stem
                
                # 完全加载到内存
                df = pd.read_parquet(file_path)
                self.preloaded_episodes[episode_id] = df
                
                # 创建索引
                action_horizon = 50
                num_rows = len(df)
                valid_samples = max(0, num_rows - action_horizon + 1)
                
                for timestep in range(valid_samples):
                    self.file_index.append((episode_id, timestep))
                    total_samples += 1
                
                print(f"  ✅ {episode_id}: {num_rows} 行 → {valid_samples} 个有效样本")
                
            except Exception as e:
                print(f"❌ 加载失败 {file_path}: {e}")
        
        print(f"✅ 预加载完成: {total_samples} 个样本")

    def __len__(self):
        return len(self.file_index)

    def _get_sample_data(self, episode_id: str, row_idx: int) -> Dict[str, Any]:
        """从预加载的数据中获取样本"""
        try:
            df = self.preloaded_episodes[episode_id]
            return df.iloc[row_idx].to_dict()
        except Exception as e:
            print(f"❌ 获取样本失败 {episode_id}[{row_idx}]: {e}")
            return self._create_dummy_sample()

    def _create_dummy_sample(self) -> Dict[str, Any]:
        """创建虚拟样本"""
        return {
            "action": np.zeros(14, dtype=np.float32).tolist(),
            "observation.state": np.zeros(14, dtype=np.float32).tolist(),
            "observation.images.exterior_image_1_left": {"bytes": b""},
            "observation.images.wrist_image_left": {"bytes": b""},
            "observation.images.wrist_image_right": {"bytes": b""},
            "timestamp": 0.0
        }

    def _process_image_fast(self, img_data: Dict) -> np.ndarray:
        """快速图像处理 - 简化版本"""
        if not isinstance(img_data, dict) or 'bytes' not in img_data:
            return np.zeros((224, 224, 3), dtype=np.float32)
        
        img_bytes = img_data['bytes']
        if not img_bytes:
            return np.zeros((224, 224, 3), dtype=np.float32)
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array * 2.0 - 1.0  # 转换到[-1, 1]
            
            # 确保是3通道
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            
            return img_array
        except Exception as e:
            return np.zeros((224, 224, 3), dtype=np.float32)

    def _find_closest_timestep_by_time(self, episode_data: pd.DataFrame, target_timestamp: float) -> int:
        """根据时间戳找到最接近的时间步"""
        if len(episode_data) == 0:
            return 0
        
        # 简化实现：基于原始fps计算
        original_dt = 1.0 / self.original_data_fps
        estimated_timestep = int(target_timestamp / original_dt)
        return max(0, min(estimated_timestep, len(episode_data) - 1))

    def __getitem__(self, idx):
        if idx >= len(self.file_index):
            raise IndexError(f"索引 {idx} 超出范围 {len(self.file_index)}")

        episode_id, timestep = self.file_index[idx]
        episode_df = self.preloaded_episodes[episode_id]

        # 获取当前时刻的数据
        current_sample = self._get_sample_data(episode_id, timestep)
        
        # 处理图像
        images = {}
        image_name_mapping = {
            'exterior_image_1_left': 'base_0_rgb',
            'wrist_image_left': 'left_wrist_0_rgb',
            'wrist_image_right': 'right_wrist_0_rgb'
        }

        for key, value in current_sample.items():
            if key.startswith("observation.images."):
                img_name = key.replace("observation.images.", "")
                img_array = self._process_image_fast(value)
                mapped_name = image_name_mapping.get(img_name, img_name)
                images[mapped_name] = img_array
        
        # 处理状态 (14维 -> 32维)
        try:
            state_14d = np.array(current_sample.get("observation.state", [0]*14), dtype=np.float32)
        except:
            state_14d = np.zeros(14, dtype=np.float32)

        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:min(14, len(state_14d))] = state_14d[:14]

        # 快速构造动作序列 - 批量处理
        action_horizon = 50
        current_timestamp = float(current_sample.get("timestamp", 0.0))
        training_dt = 1.0 / self.training_fps
        
        # 预计算所有时间步
        future_timesteps = []
        for future_step in range(action_horizon):
            target_timestamp = current_timestamp + (future_step + 1) * training_dt
            future_timestep = self._find_closest_timestep_by_time(episode_df, target_timestamp)
            future_timesteps.append(min(future_timestep, len(episode_df) - 1))
        
        # 批量获取动作数据
        actions_sequence = []
        for future_timestep in future_timesteps:
            try:
                if future_timestep < len(episode_df):
                    action_data = episode_df.iloc[future_timestep].get("action", [0]*14)
                    action_14d = np.array(action_data, dtype=np.float32)
                else:
                    action_14d = np.zeros(14, dtype=np.float32)
            except:
                action_14d = np.zeros(14, dtype=np.float32)
            
            # 零填充到32维
            action_32d = np.zeros(32, dtype=np.float32)
            action_32d[:min(14, len(action_14d))] = action_14d[:14]
            actions_sequence.append(action_32d)

        actions_expanded = np.array(actions_sequence, dtype=np.float32)
        
        # 创建图像mask - OpenPI期望的是(batch_size,)的1维mask，不是(H,W)的2维mask
        image_masks = {}
        for img_name in images:
            image_masks[img_name] = True  # 标量布尔值，表示图像有效

        # 定期显示进度
        if idx % 1000 == 0:
            print(f"🎯 快速加载样本 {idx}: episode={episode_id}, timestep={timestep}")

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state_32d,
            "actions": actions_expanded,
        }

    def get_memory_stats(self):
        """获取内存统计"""
        total_memory = sum(df.memory_usage(deep=True).sum() for df in self.preloaded_episodes.values())
        return {
            "total_samples": len(self.file_index),
            "preloaded_episodes": len(self.preloaded_episodes),
            "memory_usage_mb": total_memory / (1024 * 1024),
        }


def create_fast_dataset(data_path: str, default_prompt: str = "pick and place purple long eggplant",
                       training_fps: float = 33.3, preload_episodes: int = None):
    """创建快速数据集 - 默认加载所有parquet文件"""
    return FastEggplantDataset(data_path, default_prompt, training_fps, preload_episodes)


if __name__ == "__main__":
    # 测试快速数据集
    data_path = "/home/testuser/data/pick_and_place_eggplant/openpi"
    
    if os.path.exists(data_path):
        print("🧪 测试快速数据集...")
        dataset = create_fast_dataset(data_path, preload_episodes=3)
        
        print(f"📊 数据集大小: {len(dataset)}")
        
        # 测试加载速度
        import time
        start_time = time.time()
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"⚡ 平均加载时间: {avg_time:.3f}秒/样本")
        
        # 显示内存统计
        stats = dataset.get_memory_stats()
        print(f"📈 内存统计: {stats}")
        
    else:
        print(f"❌ 数据路径不存在: {data_path}")
