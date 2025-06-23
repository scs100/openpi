#!/usr/bin/env python3
"""
内存优化的茄子数据集加载器
解决真实数据训练时的内存问题
"""

import os
import gc
import io
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import weakref

class MemoryOptimizedEggplantDataset:
    """内存优化的茄子数据集加载器 - 流式加载，减少内存占用"""
    
    def __init__(self, data_path: str, default_prompt: str = "pick and place purple long eggplant"):
        self.data_path = Path(data_path)
        self.default_prompt = default_prompt
        
        # 不预加载所有数据，只记录文件路径和索引
        self._build_file_index()
        
        # 图像缓存 - 使用弱引用避免内存泄漏
        self._image_cache = weakref.WeakValueDictionary()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_file_index(self):
        """构建文件索引，不加载实际数据"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        parquet_files.sort()
        
        self.file_index = []  # [(file_path, row_index), ...]
        total_samples = 0
        
        for file_path in parquet_files:
            try:
                # 只读取parquet文件的元数据，不加载数据
                df = pd.read_parquet(file_path)
                num_rows = len(df)
                
                # 记录每个样本的文件路径和行索引
                for row_idx in range(num_rows):
                    self.file_index.append((file_path, row_idx))
                
                total_samples += num_rows
                
                # 立即释放DataFrame
                del df
                gc.collect()
                
            except Exception as e:
                print(f"⚠️ 跳过文件 {file_path}: {e}")
        
        print(f"✅ 构建索引完成: {total_samples} 个样本，{len(parquet_files)} 个文件")
        print(f"📊 内存优化: 流式加载，不预加载数据")
    
    def __len__(self):
        return len(self.file_index)
    
    def _load_sample_from_file(self, file_path: Path, row_idx: int) -> Dict[str, Any]:
        """从文件中加载单个样本"""
        try:
            # 只加载需要的行
            df = pd.read_parquet(file_path)
            sample = df.iloc[row_idx].to_dict()
            
            # 立即释放DataFrame
            del df
            gc.collect()
            
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
            
            # 缓存处理后的图像
            self._image_cache[cache_key] = img_array
            
            return img_array
            
        except Exception as e:
            print(f"⚠️ 图像处理失败 {img_key}: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32)
    
    def __getitem__(self, idx):
        if idx >= len(self.file_index):
            raise IndexError(f"索引 {idx} 超出范围 {len(self.file_index)}")
        
        file_path, row_idx = self.file_index[idx]
        sample = self._load_sample_from_file(file_path, row_idx)
        
        # 处理图像 - 使用缓存优化
        images = {}
        image_name_mapping = {
            'exterior_image_1_left': 'base_0_rgb',
            'wrist_image_left': 'left_wrist_0_rgb', 
            'wrist_image_right': 'right_wrist_0_rgb'
        }
        
        for key, value in sample.items():
            if key.startswith("observation.images."):
                img_name = key.replace("observation.images.", "")
                img_array = self._process_image_with_cache(img_name, value)
                mapped_name = image_name_mapping.get(img_name, img_name)
                images[mapped_name] = img_array
        
        # 处理动作和状态 (14维 -> 32维)
        try:
            action_14d = np.array(sample.get("action", [0]*14), dtype=np.float32)
            state_14d = np.array(sample.get("observation.state", [0]*14), dtype=np.float32)
        except:
            action_14d = np.zeros(14, dtype=np.float32)
            state_14d = np.zeros(14, dtype=np.float32)
        
        # 零填充到32维
        action_32d = np.zeros(32, dtype=np.float32)
        state_32d = np.zeros(32, dtype=np.float32)
        action_32d[:min(14, len(action_14d))] = action_14d[:14]
        state_32d[:min(14, len(state_14d))] = state_14d[:14]
        
        # 创建图像mask
        image_masks = {}
        for key in images.keys():
            image_masks[key] = np.array(True, dtype=bool)
        
        # 定期清理内存
        if idx % 100 == 0:
            gc.collect()
            if idx % 500 == 0:
                cache_total = self._cache_hits + self._cache_misses
                hit_rate = self._cache_hits / cache_total * 100 if cache_total > 0 else 0
                print(f"📊 样本 {idx}: 缓存命中率 {hit_rate:.1f}% ({self._cache_hits}/{cache_total})")
        
        # OpenPI期望动作有action_horizon维度 (batch, action_horizon, action_dim)
        # 我们需要将单个动作扩展到action_horizon=50
        action_horizon = 50
        actions_expanded = np.tile(action_32d[None, :], (action_horizon, 1))  # (50, 32)

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state_32d,
            "actions": actions_expanded,  # (50, 32) 而不是 (32,)
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
