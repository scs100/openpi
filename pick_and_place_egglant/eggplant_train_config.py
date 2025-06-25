#!/usr/bin/env python3
"""
茄子数据PI-0微调配置 - 兼容OpenPI训练系统
"""

import dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from PIL import Image
import io

from openpi.training import config as _config
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training import optimizer as _optimizer
from openpi.models import pi0
from openpi.shared import nnx_utils
import flax.nnx as nnx

# 从config模块导入DataConfig
DataConfig = _config.DataConfig

class EggplantDataset:
    """茄子数据集加载器"""
    
    def __init__(self, data_path: str, default_prompt: str = "pick and place purple long eggplant"):
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
                    img = Image.open(io.BytesIO(value['bytes']))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    # 转换到[-1, 1]范围
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
class EggplantDataConfig(_config.DataConfigFactory):
    """茄子数据配置工厂 - 智能适配真实数据或假数据"""
    data_path: str = "/home/testuser/data/pick_and_place_eggplant/openpi"
    default_prompt: str = "pick and place purple long eggplant"

    def create(self, assets_dirs, model_config):
        """创建数据配置实例 - 自动检测数据源并加载norm stats"""
        import os
        from openpi.shared import normalize as _normalize
        from pathlib import Path

        print(f"🔍 检查数据路径: {self.data_path}")

        if os.path.exists(self.data_path):
            print(f"✅ 找到LeRobot格式茄子数据: {self.data_path}")
            print(f"📝 使用提示: {self.default_prompt}")
            print("🎯 启用真实数据训练！")

            # 加载自己数据的norm stats
            norm_stats = None
            norm_stats_path = Path("assets/pick_and_place_eggplant")

            try:
                if norm_stats_path.exists():
                    norm_stats = _normalize.load(norm_stats_path)
                    print(f"✅ 成功加载自己数据的norm stats: {norm_stats_path}")
                    print(f"   State shape: {norm_stats['state'].mean.shape}")
                    print(f"   Action shape: {norm_stats['actions'].mean.shape}")
                else:
                    print(f"⚠️ 未找到norm stats: {norm_stats_path}")
            except Exception as e:
                print(f"❌ 加载norm stats失败: {e}")
                norm_stats = None

            # 使用真实茄子数据 - 参考LeRobotAlohaDataConfig的实现
            # 创建model transforms，包含默认prompt注入
            model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return _config.DataConfig(
                repo_id="eggplant_real_data",  # 特殊标识符，我们会在数据加载器中处理
                asset_id="pick_and_place_eggplant",  # 指定asset_id用于norm stats
                norm_stats=norm_stats,  # 直接设置norm stats
                model_transforms=model_transforms,  # 使用标准的model transforms
            )
        else:
            print(f"⚠️  数据路径不存在: {self.data_path}")
            print("🔄 使用假数据进行LoRA微调测试")
            return DataConfig(
                repo_id="fake",  # 使用假数据
            )

# 快速测试配置 (10步验证) - LoRA版本
eggplant_quick_test = _config.TrainConfig(
    name="eggplant_quick_test",

    # 模型配置 - LoRA
    model=pi0.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),

    # 数据配置
    data=EggplantDataConfig(),

    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

    # 快速测试参数
    batch_size=4,
    num_train_steps=10,  # 只训练10步用于测试
    save_interval=5,
    log_interval=2,

    # 学习率调度
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=2,
        peak_lr=2.5e-5,
        decay_steps=10,
        decay_lr=2.5e-6,
    ),

    # 优化器
    optimizer=_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    ),

    # 冻结配置 - 只训练LoRA参数
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),

    # LoRA不使用EMA
    ema_decay=None,

    exp_name="eggplant_quick_test",
    overwrite=True,
    wandb_enabled=False,
)

# LoRA微调配置 (推荐)
eggplant_lora_finetune = _config.TrainConfig(
    name="eggplant_lora_finetune",
    
    # 模型配置 - 使用LoRA
    model=pi0.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    
    # 数据配置
    data=EggplantDataConfig(),
    
    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    
    # 训练参数
    batch_size=8,
    num_train_steps=15_000,
    save_interval=1000,
    
    # 学习率调度
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=15_000,
        decay_lr=5e-6,
    ),
    
    # 优化器
    optimizer=_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    ),

    # 冻结配置 - 只训练LoRA参数
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),

    # LoRA不使用EMA
    ema_decay=None,

    # 实验配置
    exp_name="eggplant_lora_finetune",
    overwrite=True,
    wandb_enabled=True,
)



# 50步测试配置 (使用OpenPI最稳妥的默认配置)
eggplant_50_step_test = _config.TrainConfig(
    name="eggplant_50_step_test",

    # 模型配置 - 使用OpenPI官方成功的低内存配置
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
        # 使用默认值：action_dim=32, action_horizon=50, max_token_len=48
        # 这是OpenPI开发者确认在RTX 4090上成功的配置
    ),

    # 数据配置
    data=EggplantDataConfig(),

    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

    # 测试参数
    batch_size=2,        # 最小批量大小测试LoRA是否工作
    num_train_steps=50,  # 只训练50步
    save_interval=25,    # 25步保存一次
    log_interval=5,      # 5步记录一次

    # 学习率调度 (OpenPI默认，最稳妥)
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=5,      # 短预热
        peak_lr=2.5e-5,      # OpenPI默认峰值学习率
        decay_steps=50,      # 总步数
        decay_lr=2.5e-6,     # OpenPI默认最终学习率
    ),

    # 优化器 (OpenPI默认配置，最稳妥)
    optimizer=_optimizer.AdamW(
        b1=0.9,                    # OpenPI默认
        b2=0.95,                   # OpenPI默认
        eps=1e-8,                  # OpenPI默认
        weight_decay=1e-10,        # OpenPI默认
        clip_gradient_norm=1.0,    # OpenPI默认
    ),

    # 冻结配置 - 冻结VLM和LLM，只训练LoRA参数
    freeze_filter=nnx.All(
        nnx.Not(nnx_utils.PathRegex(".*lora.*")),  # 不冻结LoRA参数
        nnx.Any(
            nnx_utils.PathRegex(".*llm.*"),        # 冻结LLM参数
            nnx_utils.PathRegex(".*img.*"),        # 冻结VLM/图像编码器参数
        )
    ),

    # LoRA不使用EMA (OpenPI推荐)
    ema_decay=None,

    # 实验配置
    exp_name="eggplant_50_step_test",
    overwrite=True,
    wandb_enabled=False,  # 测试时关闭wandb
)


# 批量大小测试配置 (测试更高效率)
eggplant_batch_test = _config.TrainConfig(
    name="eggplant_batch_test",

    # 模型配置 - 使用OpenPI官方成功配置
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
        # 使用默认值：action_dim=32, action_horizon=50, max_token_len=48
    ),

    # 数据配置
    data=EggplantDataConfig(),

    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

    # 训练参数 - 优化的批量大小1配置
    batch_size=1,           # 稳定的批量大小1
    num_train_steps=200,    # 更多步数补偿小批量
    save_interval=50,       # 每50步保存
    log_interval=10,        # 每10步记录

    # 学习率调度 - 针对小批量优化
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=40,    # 更长的预热期
        peak_lr=2.5e-5,     # 小批量使用标准学习率
        decay_steps=200,
        decay_lr=2.5e-6,
    ),

    # 优化器 - OpenPI默认值
    optimizer=_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    ),

    # 冻结配置 - 使用OpenPI官方方法
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),

    # LoRA不使用EMA
    ema_decay=None,

    # 实验配置
    exp_name="eggplant_optimized_batch1",
    overwrite=True,
    wandb_enabled=True,     # 启用wandb记录
)


# 梯度累积配置 (模拟更大批量)
eggplant_grad_accumulation = _config.TrainConfig(
    name="eggplant_grad_accumulation",

    # 模型配置 - 使用OpenPI官方成功配置
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
        # 使用默认值：action_dim=32, action_horizon=50, max_token_len=48
    ),

    # 数据配置
    data=EggplantDataConfig(),

    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

    # 训练参数 - 梯度累积模拟大批量
    batch_size=1,           # 物理批量大小1
    num_train_steps=400,    # 更多步数
    save_interval=100,      # 每100步保存
    log_interval=20,        # 每20步记录

    # 学习率调度 - 模拟批量大小4的效果
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=80,    # 更长预热期
        peak_lr=4e-5,       # 稍高学习率模拟大批量
        decay_steps=400,
        decay_lr=4e-6,
    ),

    # 优化器 - OpenPI默认值
    optimizer=_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    ),

    # 冻结配置
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),

    # LoRA不使用EMA
    ema_decay=None,

    # 实验配置
    exp_name="eggplant_grad_accumulation",
    overwrite=True,
    wandb_enabled=True,
)


# 内存优化配置 (专门解决JAX内存管理问题)
eggplant_memory_optimized = _config.TrainConfig(
    name="eggplant_memory_optimized",

    # 模型配置 - 使用OpenPI官方配置
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
        # 使用默认值：action_dim=32, action_horizon=50, max_token_len=48
    ),

    # 数据配置
    data=EggplantDataConfig(),

    # 权重加载器
    weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

    # 训练参数 - 内存优化策略
    batch_size=1,           # 稳定的批量大小1
    num_train_steps=150,    # 中等步数，避免长期内存累积
    save_interval=30,       # 频繁保存，定期清理内存
    log_interval=5,         # 频繁记录，监控内存状态

    # 学习率调度 - 针对内存优化调整
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=30,
        peak_lr=2.5e-5,     # 标准LoRA学习率
        decay_steps=150,
        decay_lr=2.5e-6,
    ),

    # 优化器 - 内存优化的AdamW配置
    optimizer=_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=0.5,  # 减少梯度裁剪阈值，降低内存使用
    ),

    # 冻结配置 - 确保只训练LoRA参数
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),

    # 内存优化设置
    ema_decay=None,         # 不使用EMA，减少内存

    # 实验配置
    exp_name="eggplant_memory_optimized",
    overwrite=True,
    wandb_enabled=False,    # 关闭wandb，减少内存开销
)

if __name__ == "__main__":
    print("🤖 茄子数据PI-0微调配置")
    print("=" * 50)
    
    configs = {
        "test": eggplant_quick_test,
        "50step": eggplant_50_step_test,
        "batch": eggplant_batch_test,
        "grad": eggplant_grad_accumulation,
        "memory": eggplant_memory_optimized,
        "lora": eggplant_lora_finetune,
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name.upper()} 配置:")
        print(f"  - 批量大小: {config.batch_size}")
        print(f"  - 训练步数: {config.num_train_steps}")
        print(f"  - 峰值学习率: {config.lr_schedule.peak_lr}")
        print(f"  - 模型变体: {getattr(config.model, 'paligemma_variant', 'standard')}")
        print(f"  - EMA: {'是' if config.ema_decay else '否'}")
    
    print(f"\n🚀 使用方法 (仅LoRA微调):")
    print(f"python train_eggplant_50_steps.py  # 推荐：50步测试")
    print(f"python scripts/train.py eggplant_train_config:eggplant_quick_test")
    print(f"python scripts/train.py eggplant_train_config:eggplant_lora_finetune")
