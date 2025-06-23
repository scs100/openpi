#!/usr/bin/env python3
"""
SGD优化器主动Swap管理训练脚本
专门解决真实数据训练时swap内存溢出问题 - 主动管理版本

配置修改说明：
- 修改训练参数：调整 BATCH_SIZE, NUM_TRAIN_STEPS 等常量
- 修改学习率：调整 PEAK_LR, DECAY_LR, WARMUP_STEPS 等常量
- 修改内存设置：调整 GPU_MEM_FRACTION, SWAP_THRESHOLD 等常量
- 修改数据路径：调整 EGGPLANT_DATA_PATH 常量
- 修改实验名称：调整 EXPERIMENT_NAME, WANDB_PROJECT 等常量

所有配置都使用常量定义，便于统一管理和修改
"""

import sys
import os
import logging
import gc
import psutil
import subprocess
import threading
import time
import signal
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.abspath('.'))

# 配置常量定义
# 数据配置
EGGPLANT_DATA_PATH = "/home/testuser/data/pick_and_place_eggplant/openpi"
DEFAULT_PROMPT = "pick and place purple long eggplant"

# 训练配置
BATCH_SIZE = 7              # 批量大小
NUM_TRAIN_STEPS = 20000     # 训练步数 (增加到20k，持续训练)
SAVE_INTERVAL = 2000        # 保存间隔 (每2k步保存)
LOG_INTERVAL = 100          # 日志间隔 (更频繁记录)
NUM_WORKERS = 0             # 工作进程数

# 学习率配置
WARMUP_STEPS = 10           # 预热步数
PEAK_LR = 1e-4              # 峰值学习率
DECAY_LR = 1e-5             # 最终学习率
SGD_MOMENTUM = 0.9          # SGD动量
SGD_NESTEROV = True         # 是否使用Nesterov

# 模型配置
ACTION_DIM = 32             # 动作维度（必须保持32，与预训练模型匹配）
ACTION_HORIZON = 50         # 动作序列长度
MAX_TOKEN_LEN = 48          # 最大token长度
PALIGEMMA_VARIANT = "gemma_2b_lora"
ACTION_EXPERT_VARIANT = "gemma_300m_lora"

# 内存管理配置
GPU_MEM_FRACTION = '0.70'   # GPU内存分配比例
SWAP_THRESHOLD = 70         # Swap监控阈值(%)
MONITOR_INTERVAL = 10        # 监控间隔(秒)

# 实验配置
EXPERIMENT_NAME = "sgd_swap_manager_20k_production"
WANDB_PROJECT = "openpi_eggplant_production"
CHECKPOINT_PATH = "s3://openpi-assets/checkpoints/pi0_base/params"

# 主动Swap管理配置
class SwapManager:
    """主动Swap内存管理器"""

    def __init__(self, logger, max_swap_percent=70, check_interval=5):
        self.logger = logger
        self.max_swap_percent = max_swap_percent
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.emergency_cleanup_count = 0
        self.training_paused = False

    def initial_swap_cleanup(self):
        """训练开始前的初始swap清理"""
        self.logger.info("🧹 执行训练前swap内存清理...")

        # 记录清理前状态
        swap_before = psutil.swap_memory()
        memory_before = psutil.virtual_memory()

        self.logger.info(f"清理前 - Swap: {swap_before.used / (1024**3):.1f}GB ({swap_before.percent:.1f}%)")
        self.logger.info(f"清理前 - 内存: {memory_before.used / (1024**3):.1f}GB ({memory_before.percent:.1f}%)")

        # 1. Python垃圾回收
        collected = gc.collect()
        self.logger.info(f"  - Python GC回收: {collected} 对象")

        # 2. 强制释放系统缓存
        try:
            subprocess.run(['sync'], timeout=10)
            self.logger.info("  - 文件系统同步完成")

            # 清理页面缓存、目录项缓存和inode缓存
            result = subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'],
                                  timeout=10, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("  - 系统缓存清理完成 (drop_caches=3)")
            else:
                self.logger.warning(f"  - 系统缓存清理失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.warning("  - 系统缓存清理超时")
        except Exception as e:
            self.logger.warning(f"  - 系统缓存清理错误: {e}")

        # 3. 尝试压缩swap (如果支持)
        try:
            result = subprocess.run(['sudo', 'swapoff', '-a'], timeout=30, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("  - Swap已关闭")
                time.sleep(2)  # 等待系统稳定

                result = subprocess.run(['sudo', 'swapon', '-a'], timeout=30, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info("  - Swap已重新启用")
                else:
                    self.logger.warning(f"  - Swap重新启用失败: {result.stderr}")
            else:
                self.logger.info("  - 跳过swap重置 (可能正在使用中)")
        except subprocess.TimeoutExpired:
            self.logger.warning("  - Swap操作超时")
        except Exception as e:
            self.logger.warning(f"  - Swap操作错误: {e}")

        # 4. 最终垃圾回收
        time.sleep(1)
        collected2 = gc.collect()
        self.logger.info(f"  - 最终GC回收: {collected2} 对象")

        # 记录清理后状态
        swap_after = psutil.swap_memory()
        memory_after = psutil.virtual_memory()

        swap_freed = (swap_before.used - swap_after.used) / (1024**3)
        memory_freed = (memory_before.used - memory_after.used) / (1024**3)

        self.logger.info(f"清理后 - Swap: {swap_after.used / (1024**3):.1f}GB ({swap_after.percent:.1f}%)")
        self.logger.info(f"清理后 - 内存: {memory_after.used / (1024**3):.1f}GB ({memory_after.percent:.1f}%)")
        self.logger.info(f"✅ 清理效果 - Swap释放: {swap_freed:.1f}GB, 内存释放: {memory_freed:.1f}GB")

        return swap_freed, memory_freed
        
    def start_monitoring(self):
        """开始监控swap使用"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"🔍 开始主动Swap监控 (阈值: {self.max_swap_percent}%, 间隔: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("🛑 Swap监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                swap = psutil.swap_memory()
                memory = psutil.virtual_memory()
                
                # 检查swap使用率
                if swap.percent > self.max_swap_percent:
                    self.logger.warning(f"🚨 Swap使用率过高: {swap.percent:.1f}% > {self.max_swap_percent}%")
                    self._emergency_cleanup()
                
                # 检查系统内存
                if memory.percent > 90:
                    self.logger.warning(f"⚠️ 系统内存使用率过高: {memory.percent:.1f}%")
                    self._force_memory_cleanup()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Swap监控错误: {e}")
                time.sleep(self.check_interval)
    
    def _emergency_cleanup(self):
        """紧急内存清理"""
        self.emergency_cleanup_count += 1
        self.logger.warning(f"🚨 执行紧急内存清理 (第{self.emergency_cleanup_count}次)")
        
        # 1. Python垃圾回收
        collected = gc.collect()
        self.logger.info(f"  - Python GC回收: {collected} 对象")
        
        # 2. 强制释放系统内存
        try:
            subprocess.run(['sync'], timeout=5)
            # 尝试释放页面缓存 (需要权限)
            try:
                subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=1'], timeout=5, check=False)
            except:
                pass
        except:
            pass
        
        # 3. JAX内存清理
        try:
            import jax
            for device in jax.devices():
                device.memory_stats()
        except:
            pass
        
        # 4. 如果清理次数过多，建议暂停训练
        if self.emergency_cleanup_count > 5:
            self.logger.error("🚨 紧急清理次数过多，建议检查内存配置或暂停训练")
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        gc.collect()
        try:
            subprocess.run(['sync'], timeout=5)
        except:
            pass

def optimize_system_memory():
    """优化系统内存设置 - 使用配置常量"""
    # GPU内存分配配置
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = GPU_MEM_FRACTION
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # JAX内存管理优化
    os.environ['JAX_ENABLE_X64'] = 'false'
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

    # 禁用编译缓存
    os.environ['JAX_COMPILATION_CACHE_DIR'] = ''
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '999999999'

    # XLA优化
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

    # 系统内存优化
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'

    # WandB配置 - 在线模式
    # 移除离线模式设置，使用在线模式上传数据
    if 'WANDB_MODE' in os.environ:
        del os.environ['WANDB_MODE']
    os.environ['WANDB_SILENT'] = 'false'  # 显示WandB输出

    print("✅ 主动Swap管理内存优化设置完成")
    print("✅ WandB设置为在线模式")

# 应用内存优化
optimize_system_memory()

from scripts.train import main

# 应用内存优化数据加载器补丁
def patch_memory_optimized_data_loader():
    """补丁OpenPI数据加载器使用内存优化版本"""
    from openpi.training import data_loader as _data_loader
    from openpi.training import config as _config
    from openpi.models import model as _model
    from memory_optimized_dataset import MemoryOptimizedEggplantDataset
    
    # 保存原始函数
    original_create_dataset = _data_loader.create_dataset
    
    def memory_optimized_create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig):
        """内存优化版本的create_dataset"""
        repo_id = data_config.repo_id
        
        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")
        
        if repo_id == "fake":
            return _data_loader.FakeDataset(model_config, num_samples=1024)
        
        # 使用内存优化的茄子数据集
        if repo_id == "eggplant_real_data":
            print("🚀 使用内存优化茄子数据集!")
            return MemoryOptimizedEggplantDataset(
                data_path=EGGPLANT_DATA_PATH,
                default_prompt=DEFAULT_PROMPT
            )
        
        # 其他情况使用原始函数
        return original_create_dataset(data_config, model_config)
    
    # 替换函数
    _data_loader.create_dataset = memory_optimized_create_dataset
    print("✅ 已应用内存优化数据加载器补丁")

patch_memory_optimized_data_loader()

def setup_logging():
    """简单有效的日志设置"""
    from log_utils import setup_test_logger

    # 创建日志文件和日志器
    logger, log_file = setup_test_logger("sgd_swap_manager_100steps", "logs")

    return logger, log_file

def log_system_memory(logger, step_name=""):
    """记录系统内存和GPU内存信息"""
    try:
        # 系统内存
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        logger.info(f"💾 系统内存 {step_name}:")
        logger.info(f"  - 总内存: {memory.total / (1024**3):.1f}GB")
        logger.info(f"  - 已用内存: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        logger.info(f"  - 可用内存: {memory.available / (1024**3):.1f}GB")
        logger.info(f"  - Swap总量: {swap.total / (1024**3):.1f}GB")
        logger.info(f"  - Swap已用: {swap.used / (1024**3):.1f}GB ({swap.percent:.1f}%)")

        # GPU内存
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, total, _ = result.stdout.strip().split(', ')
            usage_pct = float(used) / float(total) * 100
            logger.info(f"🖥️ GPU内存 {step_name}: {used}MB / {total}MB ({usage_pct:.1f}%)")

        # 警告检查
        if memory.percent > 85:
            logger.warning(f"⚠️ 系统内存使用率过高: {memory.percent:.1f}%")
        if swap.percent > 50:
            logger.warning(f"⚠️ Swap使用率过高: {swap.percent:.1f}%")

    except Exception as e:
        logger.warning(f"内存信息获取失败: {e}")



def create_swap_managed_config():
    """创建主动Swap管理的SGD配置"""
    import openpi.training.config as _config
    from openpi.models import pi0
    from openpi.training import optimizer as _optimizer
    from eggplant_train_config import EggplantDataConfig, CheckpointWeightLoader
    
    config = _config.TrainConfig(
        name="sgd_swap_manager",
        project_name=WANDB_PROJECT,

        # 模型配置 - 使用常量定义
        model=pi0.Pi0Config(
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            max_token_len=MAX_TOKEN_LEN,
            paligemma_variant=PALIGEMMA_VARIANT,
            action_expert_variant=ACTION_EXPERT_VARIANT
        ),

        # 数据配置 - 使用常量定义
        data=EggplantDataConfig(
            data_path=EGGPLANT_DATA_PATH,
            default_prompt=DEFAULT_PROMPT
        ),

        # 权重加载器 - 使用常量定义
        weight_loader=CheckpointWeightLoader(CHECKPOINT_PATH),

        # 训练参数 - 使用常量定义
        batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        log_interval=LOG_INTERVAL,
        num_workers=NUM_WORKERS,

        # 学习率调度 - 使用常量定义
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=WARMUP_STEPS,
            peak_lr=PEAK_LR,
            decay_steps=NUM_TRAIN_STEPS,  # 衰减步数等于总训练步数
            decay_lr=DECAY_LR,
        ),

        # SGD优化器 - 使用常量定义
        optimizer=_optimizer.SGD(
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        ),

        # 冻结配置 - 使用常量定义
        freeze_filter=pi0.Pi0Config(
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            max_token_len=MAX_TOKEN_LEN,
            paligemma_variant=PALIGEMMA_VARIANT,
            action_expert_variant=ACTION_EXPERT_VARIANT
        ).get_freeze_filter(),

        # 内存优化设置
        ema_decay=None,

        # 实验配置 - 使用常量定义
        exp_name=EXPERIMENT_NAME,
        overwrite=True,
        wandb_enabled=True,
    )
    
    return config

if __name__ == "__main__":
    # 设置基础日志
    logger, log_file = setup_logging()

    print("🚀 SGD主动Swap管理训练 - 20K步正式训练")
    print("=" * 60)
    print("🎯 目标: 20000步持续训练，每2000步保存权重")
    print("💡 策略: 实时监控 + 主动清理 + WandB在线记录 + Terminal输出捕获")
    print(f"📝 日志文件: {log_file}")
    print("=" * 60)

    # 使用TerminalOutputCapture捕获所有输出
    from log_utils import TerminalOutputCapture

    with TerminalOutputCapture(log_file, "sgd_20k_training") as capture_logger:
        capture_logger.info("🔍 开始捕获所有terminal输出到日志文件")

        # 创建Swap管理器
        swap_manager = SwapManager(capture_logger, max_swap_percent=SWAP_THRESHOLD, check_interval=MONITOR_INTERVAL)
        start_time = time.time()

        try:
            # 记录初始内存状态
            log_system_memory(capture_logger, "初始状态")

            # 执行训练前的swap清理
            swap_freed, memory_freed = swap_manager.initial_swap_cleanup()

            # 创建配置
            config = create_swap_managed_config()

            capture_logger.info("🔧 训练配置 (使用常量定义):")
            capture_logger.info(f"  - 批量大小: {BATCH_SIZE}")
            capture_logger.info(f"  - 训练步数: {NUM_TRAIN_STEPS}")
            capture_logger.info(f"  - 保存间隔: {SAVE_INTERVAL}")
            capture_logger.info(f"  - 学习率: {PEAK_LR:.2e} -> {DECAY_LR:.2e}")
            capture_logger.info(f"  - GPU内存分配: {GPU_MEM_FRACTION}")
            capture_logger.info(f"  - Swap阈值: {SWAP_THRESHOLD}%")

            # 学习率图表功能已移除

            # 启动Swap监控
            swap_manager.start_monitoring()
            log_system_memory(capture_logger, "训练开始前")

            # 跳过归一化统计
            import openpi.training.data_loader as _data_loader
            original_create_data_loader = _data_loader.create_data_loader

            def skip_norm_create_data_loader(config, **kwargs):
                kwargs['skip_norm_stats'] = True
                return original_create_data_loader(config, **kwargs)

            _data_loader.create_data_loader = skip_norm_create_data_loader
            capture_logger.info("✅ 已跳过归一化统计计算")

            # 运行训练
            import openpi.shared.array_typing as at
            with at.disable_typechecking():
                capture_logger.info("🎯 开始训练...")
                main(config)

            capture_logger.info("✅ 训练成功完成!")
            capture_logger.info(f"🎉 紧急清理次数: {swap_manager.emergency_cleanup_count}")

            # 记录训练总结
            from log_utils import log_test_summary
            training_duration = time.time() - start_time
            log_test_summary(capture_logger, True, training_duration, {
                "训练步数": NUM_TRAIN_STEPS,
                "批量大小": BATCH_SIZE,
                "学习率范围": f"{PEAK_LR:.2e} -> {DECAY_LR:.2e}",
                "GPU内存分配": GPU_MEM_FRACTION,
                "紧急清理次数": swap_manager.emergency_cleanup_count,
                "实验名称": EXPERIMENT_NAME
            })

        except Exception as e:
            capture_logger.error(f"❌ 训练失败: {e}")
            log_system_memory(capture_logger, "失败时")
            raise

        finally:
            # 停止Swap监控
            swap_manager.stop_monitoring()
            log_system_memory(capture_logger, "最终状态")

    print(f"\n🎉 训练完成！完整日志已保存到: {log_file}")
