#!/usr/bin/env python3
"""
内存优化的茄子数据集加载器
解决真实数据训练时的内存问题
"""

import os
import gc
import io
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from PIL import Image


class MemoryOptimizedEggplantDataset:
    """内存优化的茄子数据集加载器 - 流式加载，减少内存占用"""
    
    def __init__(self, data_path: str, default_prompt: str = "pick and place purple long eggplant",
                 training_fps: float = 33.3):
        self.data_path = Path(data_path)
        self.default_prompt = default_prompt
        self.training_fps = training_fps  # 训练时使用的fps，默认33.3fps

        # 读取数据集的原始fps信息
        self.original_data_fps = self._read_dataset_fps()
        
        # 不预加载所有数据，只记录文件路径和索引
        self._build_file_index()
        
        # 图像缓存 - 使用普通字典，支持多进程
        self._image_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # 限制缓存大小避免内存泄漏

        # 时间戳缓存 - 避免重复读取parquet文件
        self._timestamp_cache = {}
        self._timestamp_cache_size = 5000  # 时间戳缓存大小

        # 文件缓存 - 缓存已读取的parquet文件，避免重复IO
        self._file_cache = {}
        self._max_file_cache_size = 3  # 最多缓存3个文件，平衡内存和性能

    def _read_dataset_fps(self):
        """从info.json读取数据集的fps"""
        info_json_paths = [
            self.data_path / "info.json",
            self.data_path / "meta" / "info.json",
        ]

        for info_path in info_json_paths:
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        fps = info.get('fps', 100)  # 默认100fps
                        print(f"📊 从 {info_path} 读取到原始数据fps: {fps}")
                        return fps
                except Exception as e:
                    print(f"⚠️ 读取 {info_path} 失败: {e}")

        print(f"⚠️ 未找到info.json，使用默认fps: 100")
        return 100  # 默认值

    def _get_timestamp_from_sample(self, file_path: Path, row_idx: int) -> float:
        """从样本中获取时间戳 - 使用文件缓存优化"""
        try:
            # 使用文件缓存
            file_key = str(file_path)

            if file_key not in self._file_cache:
                df = pd.read_parquet(file_path)
                self._file_cache[file_key] = df

                # 管理缓存大小
                if len(self._file_cache) > self._max_file_cache_size:
                    oldest_key = next(iter(self._file_cache))
                    del self._file_cache[oldest_key]
                    gc.collect()

            df = self._file_cache[file_key]
            timestamp = df.iloc[row_idx]['timestamp']
            return float(timestamp)
        except Exception as e:
            print(f"⚠️ 获取时间戳失败 {file_path}[{row_idx}]: {e}")
            return 0.0

    def _find_closest_timestep_by_time(self, episode_data, target_timestamp: float) -> int:
        """根据时间戳找到最接近的时间步 - 简化版本"""
        # 简化实现：假设时间戳是线性递增的
        # 根据原始数据fps (100fps) 和目标时间戳计算近似位置

        if len(episode_data) == 0:
            return 0

        # 假设时间戳从0开始，以1/original_fps的间隔递增
        original_dt = 1.0 / self.original_data_fps  # 原始数据的时间间隔

        # 计算近似的时间步位置
        estimated_timestep = int(target_timestamp / original_dt)

        # 确保在有效范围内
        estimated_timestep = max(0, min(estimated_timestep, len(episode_data) - 1))

        return estimated_timestep
    
    def _build_file_index(self):
        """构建文件索引，按episode组织数据以支持时序动作序列 - 带进度显示"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        parquet_files.sort()

        print(f"🔍 开始构建数据索引，发现 {len(parquet_files)} 个文件...")

        # 按episode组织数据: {episode_id: [(file_path, row_index), ...]}
        self.episodes = {}
        self.file_index = []  # [(episode_id, timestep_in_episode), ...]
        total_samples = 0

        for i, file_path in enumerate(parquet_files):
            try:
                # 显示进度
                if i % 5 == 0 or i == len(parquet_files) - 1:
                    print(f"📂 处理文件 {i+1}/{len(parquet_files)}: {file_path.name}")

                # 从文件名提取episode信息
                episode_id = file_path.stem  # 例如: "episode_0"

                # 只读取parquet文件的元数据，不加载数据
                df = pd.read_parquet(file_path)
                num_rows = len(df)

                # 存储episode的所有时间步
                episode_data = []
                for row_idx in range(num_rows):
                    episode_data.append((file_path, row_idx))

                self.episodes[episode_id] = episode_data

                # 为每个时间步创建索引，但要确保有足够的未来动作
                action_horizon = 50
                valid_samples_in_file = num_rows - action_horizon + 1
                for timestep in range(max(0, valid_samples_in_file)):  # 确保有足够的未来动作
                    self.file_index.append((episode_id, timestep))
                    total_samples += 1

                # 立即释放DataFrame
                del df
                gc.collect()

                # 显示文件处理结果
                if valid_samples_in_file > 0:
                    print(f"  ✅ {episode_id}: {num_rows} 行 → {valid_samples_in_file} 个有效样本")
                else:
                    print(f"  ⚠️ {episode_id}: {num_rows} 行 → 0 个有效样本 (序列太短)")

            except Exception as e:
                print(f"❌ 跳过文件 {file_path}: {e}")

        print(f"✅ 构建索引完成: {total_samples} 个有效样本，{len(parquet_files)} 个文件")
        print(f"📊 每个样本包含当前状态 + 未来{50}步动作序列")
        print(f"🔄 内存优化: 流式加载，不预加载数据")
    
    def __len__(self):
        return len(self.file_index)
    
    def _load_sample_from_file(self, file_path: Path, row_idx: int) -> Dict[str, Any]:
        """从文件中加载单个样本 - 使用文件缓存优化"""
        try:
            # 使用文件缓存避免重复读取同一个文件
            file_key = str(file_path)

            if file_key not in self._file_cache:
                # 文件不在缓存中，读取并缓存
                df = pd.read_parquet(file_path)
                self._file_cache[file_key] = df

                # 管理缓存大小，避免内存泄漏
                if len(self._file_cache) > self._max_file_cache_size:
                    # 删除最早的缓存项
                    oldest_key = next(iter(self._file_cache))
                    del self._file_cache[oldest_key]
                    gc.collect()

            # 从缓存中获取数据
            df = self._file_cache[file_key]
            sample = df.iloc[row_idx].to_dict()

            return sample

        except Exception as e:
            print(f"❌ 加载样本失败 {file_path}[{row_idx}]: {e}")
            # 返回一个默认样本
            return self._create_dummy_sample()
    
    def _create_dummy_sample(self) -> Dict[str, Any]:
        """创建一个虚拟样本用于错误恢复"""
        return {
            "action": np.zeros(14, dtype=np.float32).tolist(),
            "observation.state": np.zeros(14, dtype=np.float32).tolist(),
            "observation.images.exterior_image_1_left": {"bytes": b""},
            "observation.images.wrist_image_left": {"bytes": b""},
            "observation.images.wrist_image_right": {"bytes": b""},
        }
    
    def _process_image_with_cache(self, img_key: str, img_data: Dict) -> np.ndarray:
        """处理图像数据，使用缓存优化"""
        if not isinstance(img_data, dict) or 'bytes' not in img_data:
            # 返回黑色图像
            return np.zeros((224, 224, 3), dtype=np.float32)
        
        # 检查缓存
        img_bytes = img_data['bytes']
        cache_key = hash(img_bytes) if img_bytes else 0
        
        if cache_key in self._image_cache:
            self._cache_hits += 1
            return self._image_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            if not img_bytes:
                img_array = np.zeros((224, 224, 3), dtype=np.float32)
            else:
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((224, 224))  # 统一尺寸
                img_array = np.array(img, dtype=np.float32) / 255.0
                # 转换到[-1, 1]范围
                img_array = img_array * 2.0 - 1.0
                
                # 确保是3通道
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
            
            # 缓存处理后的图像，管理缓存大小
            if len(self._image_cache) >= self._max_cache_size:
                # 简单的FIFO策略：删除最早的缓存项
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]

            self._image_cache[cache_key] = img_array

            return img_array
            
        except Exception as e:
            print(f"⚠️ 图像处理失败 {img_key}: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32)
    
    def __getitem__(self, idx):
        if idx >= len(self.file_index):
            raise IndexError(f"索引 {idx} 超出范围 {len(self.file_index)}")

        episode_id, timestep = self.file_index[idx]
        episode_data = self.episodes[episode_id]

        # 获取当前时刻的状态和图像
        current_file_path, current_row_idx = episode_data[timestep]
        current_sample = self._load_sample_from_file(current_file_path, current_row_idx)
        
        # 处理图像 - 使用当前时刻的图像
        images = {}
        image_name_mapping = {
            'exterior_image_1_left': 'base_0_rgb',
            'wrist_image_left': 'left_wrist_0_rgb',
            'wrist_image_right': 'right_wrist_0_rgb'
        }

        for key, value in current_sample.items():
            if key.startswith("observation.images."):
                img_name = key.replace("observation.images.", "")
                img_array = self._process_image_with_cache(img_name, value)
                mapped_name = image_name_mapping.get(img_name, img_name)
                images[mapped_name] = img_array
        
        # 处理当前时刻的状态 (14维 -> 32维)
        try:
            state_14d = np.array(current_sample.get("observation.state", [0]*14), dtype=np.float32)
        except:
            state_14d = np.zeros(14, dtype=np.float32)

        # 零填充到32维
        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:min(14, len(state_14d))] = state_14d[:14]

        # 构造未来50步的动作序列
        action_horizon = 50
        actions_sequence = []

        # 获取当前时刻的时间戳
        current_file_path, current_row_idx = episode_data[timestep]
        current_timestamp = self._get_timestamp_from_sample(current_file_path, current_row_idx)

        # 计算训练时的时间间隔
        training_dt = 1.0 / self.training_fps  # 训练fps对应的时间间隔

        # 批量预计算所有需要的时间步，减少重复计算
        future_timesteps = []
        for future_step in range(action_horizon):
            target_timestamp = current_timestamp + (future_step + 1) * training_dt
            future_timestep = self._find_closest_timestep_by_time(episode_data, target_timestamp)
            future_timesteps.append(future_timestep)

        # 按文件分组，减少文件切换
        file_groups = {}
        for i, future_timestep in enumerate(future_timesteps):
            if future_timestep < len(episode_data):
                future_file_path, future_row_idx = episode_data[future_timestep]
                file_key = str(future_file_path)
                if file_key not in file_groups:
                    file_groups[file_key] = []
                file_groups[file_key].append((i, future_row_idx))

        # 批量加载每个文件的数据
        actions_dict = {}
        for file_key, indices in file_groups.items():
            file_path = Path(file_key)
            # 确保文件在缓存中
            if file_key not in self._file_cache:
                df = pd.read_parquet(file_path)
                self._file_cache[file_key] = df

                if len(self._file_cache) > self._max_file_cache_size:
                    oldest_key = next(iter(self._file_cache))
                    del self._file_cache[oldest_key]
                    gc.collect()

            df = self._file_cache[file_key]
            for action_idx, row_idx in indices:
                try:
                    action_14d = np.array(df.iloc[row_idx].get("action", [0]*14), dtype=np.float32)
                except:
                    action_14d = np.zeros(14, dtype=np.float32)
                actions_dict[action_idx] = action_14d

        # 构建最终的动作序列
        for i in range(action_horizon):
            if i in actions_dict:
                action_14d = actions_dict[i]
            else:
                action_14d = np.zeros(14, dtype=np.float32)

            # 零填充到32维
            action_32d = np.zeros(32, dtype=np.float32)
            action_32d[:min(14, len(action_14d))] = action_14d[:14]
            actions_sequence.append(action_32d)

        # 转换为numpy数组 (action_horizon, action_dim)
        actions_expanded = np.array(actions_sequence, dtype=np.float32)  # (50, 32)
        
        # 创建图像mask
        image_masks = {}
        for key in images.keys():
            image_masks[key] = np.array(True, dtype=bool)
        
        # 定期清理内存和显示缓存统计
        if idx % 100 == 0:
            gc.collect()
            if idx % 500 == 0:
                cache_total = self._cache_hits + self._cache_misses
                hit_rate = self._cache_hits / cache_total * 100 if cache_total > 0 else 0
                file_cache_size = len(self._file_cache)
                print(f"📊 样本 {idx}: 图像缓存命中率 {hit_rate:.1f}% ({self._cache_hits}/{cache_total})")
                print(f"📂 文件缓存: {file_cache_size}/{self._max_file_cache_size} 个文件")
                print(f"🎯 时序数据: episode={episode_id}, timestep={timestep}, 动作序列={actions_expanded.shape}")
                print(f"🎬 帧率采样: 原始{self.original_data_fps}fps → 训练{self.training_fps}fps (基于时间戳)")

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state_32d,  # 当前时刻的状态
            "actions": actions_expanded,  # 未来50步的真实动作序列 (50, 32)
        }
    
    def get_memory_stats(self):
        """获取内存统计信息"""
        cache_size = len(self._image_cache)
        cache_total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / cache_total * 100 if cache_total > 0 else 0
        
        return {
            "total_samples": len(self.file_index),
            "cache_size": cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

def create_memory_optimized_dataset(data_path: str, default_prompt: str = "pick and place purple long eggplant"):
    """创建内存优化的数据集"""
    return MemoryOptimizedEggplantDataset(data_path, default_prompt)

if __name__ == "__main__":
    # 测试内存优化数据集
    data_path = "/home/testuser/data/pick_and_place_eggplant/openpi"
    
    if os.path.exists(data_path):
        print("🧪 测试内存优化数据集...")
        dataset = create_memory_optimized_dataset(data_path)
        
        print(f"📊 数据集大小: {len(dataset)}")
        
        # 测试加载几个样本
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"样本 {i}: 图像 {len(sample['image'])}, 动作维度 {sample['actions'].shape}")
        
        # 显示内存统计
        stats = dataset.get_memory_stats()
        print(f"📈 内存统计: {stats}")
        
    else:
        print(f"❌ 数据路径不存在: {data_path}")
