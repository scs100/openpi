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
import openpi.transforms as _transforms
import flax.nnx as nnx

# 从config模块导入DataConfig
DataConfig = _config.DataConfig

@dataclasses.dataclass(frozen=True)
class CustomDataConfig(_config.DataConfigFactory):
    """通用数据配置工厂 - 智能适配真实数据或假数据，支持OpenPI官方推荐的delta actions"""
    data_path: str = "/path/to/your/dataset"
    default_prompt: str = "perform the task"
    dataset_name: str = "custom_dataset"  # 数据集名称，用于norm stats路径
    use_delta_joint_actions: bool = True  # 是否使用相对关节角（delta actions）

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

            # 加载数据集的norm stats（使用第30步的delta范围）
            norm_stats = None
            if self.use_delta_joint_actions:
                # 使用第30步的norm stats，覆盖最大delta范围
                suffix = "_del_step30"
                print("🎯 使用第30步的norm stats，适用于action_horizon=30的训练")
            else:
                suffix = "_abs"
            norm_stats_path = Path(f"assets/{self.dataset_name}{suffix}")

            try:
                if norm_stats_path.exists():
                    norm_stats = _normalize.load(norm_stats_path)
                    print(f"✅ 成功加载数据集的norm stats: {norm_stats_path}")
                    print(f"   State shape: {norm_stats['state'].mean.shape}")
                    print(f"   Action shape: {norm_stats['actions'].mean.shape}")
                    mode_desc = "Delta actions (action - current_state)" if self.use_delta_joint_actions else "Absolute actions"
                    print(f"   模式: {mode_desc}")
                else:
                    print(f"⚠️ 未找到norm stats: {norm_stats_path}")
                    print(f"   请运行 compute_norm_stats.py 生成对应的norm stats")
            except Exception as e:
                print(f"❌ 加载norm stats失败: {e}")
                norm_stats = None

            # 使用真实数据 - 参考LeRobotAlohaDataConfig的实现
            # 创建data transforms，支持OpenPI官方推荐的delta actions
            data_transforms = _transforms.Group()

            # 创建mask：前12维是关节（6+6），第7和14维是夹爪（保持绝对值）
            # 对于14维动作：[joint0-5, gripper0, joint6-11, gripper1]
            # mask = [True]*6 + [False] + [True]*6 + [False] + [False]*18 (填充到32维)
            joint_mask = [True] * 6 + [False] + [True] * 6 + [False] + [False] * 18

            # 如果启用相对关节角模式，添加delta actions transforms
            if self.use_delta_joint_actions:
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(joint_mask)],
                    outputs=[_transforms.AbsoluteActions(joint_mask)],
                )
                print(f"✅ 启用OpenPI官方delta actions模式 (action - current_state)")
                print(f"   关节mask: {joint_mask[:14]}")

            # 创建model transforms，包含默认prompt注入
            model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return _config.DataConfig(
                repo_id="custom_real_data",  # 特殊标识符，我们会在数据加载器中处理
                asset_id=self.dataset_name,  # 指定asset_id用于norm stats
                norm_stats=norm_stats,  # 直接设置norm stats
                data_transforms=data_transforms,  # 添加data transforms
                model_transforms=model_transforms,  # 使用标准的model transforms
            )
        else:
            print(f"❌ 数据路径不存在: {self.data_path}")
            raise FileNotFoundError(f"指定的数据集路径不存在: {self.data_path}")



