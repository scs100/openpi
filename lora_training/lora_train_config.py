#!/usr/bin/env python3
"""
OpenPI LoRA 微调通用配置 - 兼容OpenPI训练系统
支持任意 LeRobot 格式数据集的 LoRA 微调
"""

import dataclasses
from pathlib import Path
import numpy as np

from openpi.training import config as _config
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training import optimizer as _optimizer
from openpi.models import pi0
from openpi.shared import nnx_utils
import flax.nnx as nnx

# 从config模块导入DataConfig
DataConfig = _config.DataConfig

class CustomDataset:
    """通用数据集加载器 - 支持 LeRobot 格式数据集"""

    def __init__(self, data_path: str, default_prompt: str = "perform the task"):
        self.data_path = Path(data_path)
        self.default_prompt = default_prompt

        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载parquet数据"""
        import pandas as pd
        import json
        
        # 加载所有parquet文件
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        parquet_files.sort()
        
        self.samples = []
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            self.samples.extend(df.to_dict('records'))
        
        print(f"✅ 加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理图像
        images = {}
        image_name_mapping = {
            'exterior_image_1_left': 'base_0_rgb',
            'wrist_image_left': 'left_wrist_0_rgb', 
            'wrist_image_right': 'right_wrist_0_rgb'
        }
        
        for key, value in sample.items():
            if key.startswith("observation.images."):
                img_name = key.replace("observation.images.", "")
                if isinstance(value, dict) and 'bytes' in value:
                    # 使用cv2解码以正确处理BGR格式
                    import cv2
                    img_bytes = value['bytes']
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    # ⚠️ 关键：OpenCV返回BGR，OpenPI需要RGB，进行转换
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 转换为float32并归一化到[-1, 1]
                    img_array = img.astype(np.float32) / 255.0
                    img_array = img_array * 2.0 - 1.0

                    # 映射键名
                    mapped_name = image_name_mapping.get(img_name, img_name)
                    images[mapped_name] = img_array
        
        # 处理动作 (14维 -> 32维)
        action_14d = np.array(sample["action"], dtype=np.float32)
        action_32d = np.zeros(32, dtype=np.float32)
        action_32d[:14] = action_14d
        
        # 处理状态 (14维 -> 32维)
        state_14d = np.array(sample["observation.state"], dtype=np.float32)
        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:14] = state_14d
        
        # 确保所有数据类型都是JAX兼容的
        # 注意：使用OpenPI期望的字段名

        # 创建image_mask - 所有图像都是有效的
        # 注意：OpenPI期望image_mask只有批量维度，不是空间维度
        image_masks = {}
        for key in images.keys():
            # 创建标量True，表示这个图像是有效的
            image_masks[key] = np.array(True, dtype=bool)  # 标量bool

        return {
            "image": images,                            # OpenPI期望"image"而不是"images"
            "image_mask": image_masks,                  # 图像mask
            "state": state_32d.astype(np.float32),      # 确保float32
            "actions": action_32d.astype(np.float32),   # 确保float32
        }

@dataclasses.dataclass(frozen=True)
class CustomDataConfig(_config.DataConfigFactory):
    """通用数据配置工厂 - 智能适配真实数据或假数据"""
    data_path: str = "/path/to/your/dataset"
    default_prompt: str = "perform the task"
    dataset_name: str = "custom_dataset"  # 数据集名称，用于norm stats路径

    def create(self, assets_dirs, model_config):
        """创建数据配置实例 - 自动检测数据源并加载norm stats"""
        import os
        from openpi.shared import normalize as _normalize
        from pathlib import Path

        print(f"🔍 检查数据路径: {self.data_path}")

        if os.path.exists(self.data_path):
            print(f"✅ 找到LeRobot格式数据集: {self.data_path}")
            print(f"📝 使用提示: {self.default_prompt}")
            print("🎯 启用真实数据训练！")

            # 加载数据集的norm stats
            norm_stats = None
            norm_stats_path = Path(f"assets/{self.dataset_name}")

            try:
                if norm_stats_path.exists():
                    norm_stats = _normalize.load(norm_stats_path)
                    print(f"✅ 成功加载数据集的norm stats: {norm_stats_path}")
                    print(f"   State shape: {norm_stats['state'].mean.shape}")
                    print(f"   Action shape: {norm_stats['actions'].mean.shape}")
                else:
                    print(f"⚠️ 未找到norm stats: {norm_stats_path}")
            except Exception as e:
                print(f"❌ 加载norm stats失败: {e}")
                norm_stats = None

            # 使用真实数据 - 参考LeRobotAlohaDataConfig的实现
            # 创建model transforms，包含默认prompt注入
            model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return _config.DataConfig(
                repo_id="custom_real_data",  # 特殊标识符，我们会在数据加载器中处理
                asset_id=self.dataset_name,  # 指定asset_id用于norm stats
                norm_stats=norm_stats,  # 直接设置norm stats
                model_transforms=model_transforms,  # 使用标准的model transforms
            )
        else:
            print(f"⚠️  数据路径不存在: {self.data_path}")
            print("🔄 使用假数据进行LoRA微调测试")
            return DataConfig(
                repo_id="fake",  # 使用假数据
            )




if __name__ == "__main__":
    print("🤖 OpenPI LoRA 微调通用配置")
    print("=" * 50)

    # 示例配置
    data_path = "/path/to/your/dataset"
    dataset_name = "your_dataset"
    prompt = "perform the task"

 
 

    print(f"\n🚀 使用方法:")
    print(f"# 1. 导入配置模块")
    print(f"from lora_train_config import create_lora_finetune_config")
    print(f"")
    print(f"# 2. 创建配置")
    print(f"config = create_lora_finetune_config(")
    print(f"    data_path='/path/to/your/dataset',")
    print(f"    dataset_name='your_dataset',")
    print(f"    prompt='your task description',")
    print(f"    exp_name='my_experiment'")
    print(f")")
    print(f"")
    print(f"# 3. 运行训练")
    print(f"python scripts/train.py config")

    print(f"\n� 提示:")
    print(f"  - 确保数据路径存在且为 LeRobot 格式")
    print(f"  - 使用 compute_norm_stats.py 预先计算归一化统计")
    print(f"  - 根据GPU内存调整批量大小")
