#!/usr/bin/env python3
"""
OpenPI LoRA 训练脚本 - 主动内存管理版本
支持任意 LeRobot 格式数据集的 LoRA 微调训练

配置修改说明：
- 修改训练参数：调整 BATCH_SIZE, NUM_TRAIN_STEPS 等常量
- 修改学习率：调整 PEAK_LR, DECAY_LR, WARMUP_STEPS 等常量
- 修改内存设置：调整 GPU_MEM_FRACTION, SWAP_THRESHOLD 等常量
- 修改数据路径：调整 DATASET_PATH 常量
- 修改实验名称：调整 EXPERIMENT_NAME, WANDB_PROJECT 等常量
- 修改训练帧率：在数据预处理阶段完成，不在训练时采样

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
DATASET_PATH = "/home/testuser/data/pick_and_place_eggplant/openpi_33fps"  # 修改为您的数据集路径
DATASET_NAME = "pick_and_place_eggplant_33fps"  # 修改为您的数据集名称
DEFAULT_PROMPT = "pick then long eggplant and place on the plant" # 修改为您的任务描述
# 训练帧率配置 - 已废弃，现在使用数据集原始时间序列
TRAINING_FPS = 33.3  # ⚠️ 此参数已不再使用，帧率在数据预处理时确定
# 实验配置
EXPERIMENT_NAME = "lora_training"  # 修改为您的实验名称
WANDB_PROJECT = "lora_eggplant"  # 修改为您的WandB项目名
# WandB配置
FORCE_WANDB_OFFLINE = False  # 设置为True强制使用离线模式，False为智能模式
# 恢复训练配置
RESUME_TRAINING = False           # 启用恢复训练
OVERWRITE_CHECKPOINT = True   #

# 优化器选择配置
USE_ADAMW = False           # True: 使用AdamW, False: 使用SGD

# 批量大小配置 - 根据优化器动态调整
SGD_BATCH_SIZE = 6          # SGD批量大小 (已验证在RTX 4090上稳定)
ADAMW_BATCH_SIZE = 4        # AdamW批量大小 (为额外内存需求预留空间)
BATCH_SIZE = ADAMW_BATCH_SIZE if USE_ADAMW else SGD_BATCH_SIZE

NUM_WORKERS = 0            # 预加载使用单进程即可
SAVE_INTERVAL = 1000       # 保存间隔 (每1000步保存，大幅减少内存压力)
NUM_TRAIN_STEPS = 40000     # 训练步数 (增加到20k，持续训练)
LOG_INTERVAL = 100          # 日志间隔 (更频繁记录)
KEEP_PERIOD = 2000          # 检查点保留周期 (每1000步的检查点永久保留)

# 学习率配置 - 动态预热步数
WARMUP_RATIO = 0.02         # 预热比例 (2% of total steps)

# SGD配置
SGD_PEAK_LR = 1e-4          # SGD峰值学习率
SGD_DECAY_LR = 1e-5         # SGD最终学习率
SGD_MOMENTUM = 0.9          # SGD动量
SGD_NESTEROV = True         # 是否使用Nesterov

# AdamW配置
ADAMW_PEAK_LR = 5e-5        # AdamW峰值学习率 (比SGD低)
ADAMW_DECAY_LR = 5e-6       # AdamW最终学习率
ADAMW_B1 = 0.9              # AdamW一阶矩衰减率
ADAMW_B2 = 0.95             # AdamW二阶矩衰减率
ADAMW_EPS = 1e-8            # AdamW数值稳定性
ADAMW_WEIGHT_DECAY = 1e-10  # AdamW权重衰减
ADAMW_CLIP_NORM = 1.0       # AdamW梯度裁剪

# 动态配置 - 根据优化器选择
PEAK_LR = ADAMW_PEAK_LR if USE_ADAMW else SGD_PEAK_LR
DECAY_LR = ADAMW_DECAY_LR if USE_ADAMW else SGD_DECAY_LR


# 模型配置
ACTION_DIM = 32             # 动作维度（必须保持32，与预训练模型匹配）
ACTION_HORIZON = 50         # 动作序列长度
MAX_TOKEN_LEN = 48          # 最大token长度
PALIGEMMA_VARIANT = "gemma_2b_lora"
ACTION_EXPERT_VARIANT = "gemma_300m_lora"

# 内存管理配置 - 根据优化器动态调整
SGD_GPU_MEM_FRACTION = '0.70'    # SGD GPU内存分配比例
ADAMW_GPU_MEM_FRACTION = '0.65'  # AdamW GPU内存分配比例 (更保守)
GPU_MEM_FRACTION = ADAMW_GPU_MEM_FRACTION if USE_ADAMW else SGD_GPU_MEM_FRACTION
# 分阶段Swap管理阈值 - 优化版
SWAP_WARNING_THRESHOLD = 30    # 警告阈值 - 开始轻度清理
SWAP_ACTION_THRESHOLD = 50     # 行动阈值 - 中度清理
SWAP_EMERGENCY_THRESHOLD = 70  # 紧急阈值 - 重度清理
MONITOR_INTERVAL = 10          # 监控间隔(秒) - 平衡监控和性能


CHECKPOINT_PATH = "s3://openpi-assets/checkpoints/pi0_base/params"



# 计算动态预热步数 (在RESUME_TRAINING定义后)
WARMUP_STEPS = int(NUM_TRAIN_STEPS * WARMUP_RATIO) if not RESUME_TRAINING else 50  # 恢复训练时短预热

# 恢复训练专用内存配置
RESUME_GPU_MEM_FRACTION = GPU_MEM_FRACTION  # 恢复训练时使用与当前优化器匹配的内存配置
RESUME_BATCH_SIZE = BATCH_SIZE    # 恢复训练时使用已验证的批量大小
DYNAMIC_BATCH_ADJUSTMENT = False  # 已找到最优批量大小，禁用动态调整
MIN_BATCH_SIZE = 1                # 最小批量大小

# 稳定性保护配置
MEMORY_MONITOR_INTERVAL = 60      # 内存监控间隔(秒) - 降低频率减少系统负担
MAX_GPU_MEMORY_PERCENT = 75       # GPU内存使用率警告阈值
MAX_SYSTEM_MEMORY_PERCENT = 85    # 系统内存使用率警告阈值 (调整到85%)
AUTO_SAVE_INTERVAL = 500          # 自动保存间隔(步数) - 与SAVE_INTERVAL一致

# 动态批量大小调整配置
ENABLE_DYNAMIC_BATCH_SCALING = True   # 启用训练中动态批量大小调整
TARGET_GPU_UTILIZATION = 73          # 目标GPU显存使用率 (73%)
SAFE_GPU_UTILIZATION = 78            # 安全上限GPU显存使用率 (78%)
BATCH_ADJUSTMENT_INTERVAL = 100       # 每100步检查一次是否可以调整批量大小
MIN_STABLE_STEPS = 50                 # 批量大小稳定运行50步后才考虑增加
MAX_BATCH_SIZE = 10                   # 最大批量大小限制

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
        if RESUME_TRAINING:
            self.logger.info("🔄 执行恢复训练前的深度内存清理...")
            return self._resume_training_cleanup()
        else:
            self.logger.info("🧹 执行训练前swap内存清理...")
            return self._normal_training_cleanup()

    def _resume_training_cleanup(self):
        """恢复训练专用的深度内存清理"""
        # 记录清理前状态
        swap_before = psutil.swap_memory()
        memory_before = psutil.virtual_memory()

        self.logger.info(f"清理前 - Swap: {swap_before.used / (1024**3):.1f}GB ({swap_before.percent:.1f}%)")
        self.logger.info(f"清理前 - 内存: {memory_before.used / (1024**3):.1f}GB ({memory_before.percent:.1f}%)")

        # 1. 强制GPU内存清理（恢复训练特有）
        self.logger.info("  🔄 执行GPU内存深度清理...")
        try:
            # 重置GPU
            subprocess.run(['nvidia-smi', '--gpu-reset'], timeout=10, capture_output=True)
            time.sleep(3)
            self.logger.info("  - GPU重置完成")
        except Exception as e:
            self.logger.warning(f"  - GPU重置失败: {e}")

        # 2. 多轮Python垃圾回收
        total_collected = 0
        for i in range(3):
            collected = gc.collect()
            total_collected += collected
            time.sleep(1)
        self.logger.info(f"  - Python深度GC回收: {total_collected} 对象")

        # 3. 强制释放系统缓存
        try:
            subprocess.run(['sync'], timeout=15)
            self.logger.info("  - 文件系统同步完成")

            # 清理所有缓存
            result = subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'],
                                  timeout=15, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("  - 系统缓存清理完成 (drop_caches=3)")
            else:
                self.logger.warning(f"  - 系统缓存清理失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.warning("  - 系统缓存清理超时")
        except Exception as e:
            self.logger.warning(f"  - 系统缓存清理错误: {e}")

        # 4. 强制swap重置（恢复训练时更重要）
        try:
            result = subprocess.run(['sudo', 'swapoff', '-a'], timeout=45, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("  - Swap已关闭")
                time.sleep(5)  # 恢复训练时等待更长时间

                result = subprocess.run(['sudo', 'swapon', '-a'], timeout=45, capture_output=True, text=True)
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

        # 5. 最终深度清理
        time.sleep(2)
        final_collected = gc.collect()
        self.logger.info(f"  - 最终深度GC回收: {final_collected} 对象")

        # 记录清理后状态
        swap_after = psutil.swap_memory()
        memory_after = psutil.virtual_memory()

        swap_freed = (swap_before.used - swap_after.used) / (1024**3)
        memory_freed = (memory_before.used - memory_after.used) / (1024**3)

        self.logger.info(f"清理后 - Swap: {swap_after.used / (1024**3):.1f}GB ({swap_after.percent:.1f}%)")
        self.logger.info(f"清理后 - 内存: {memory_after.used / (1024**3):.1f}GB ({memory_after.percent:.1f}%)")
        self.logger.info(f"✅ 恢复训练深度清理效果 - Swap释放: {swap_freed:.1f}GB, 内存释放: {memory_freed:.1f}GB")

        return swap_freed, memory_freed

    def _normal_training_cleanup(self):
        """正常训练的标准内存清理"""
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
        self.logger.info(f"🔍 开始增强内存监控 (Swap阈值: {self.max_swap_percent}%, 间隔: {self.check_interval}s)")
        self.logger.info(f"🛡️ 稳定性保护 - GPU阈值: {MAX_GPU_MEMORY_PERCENT}%, 系统内存阈值: {MAX_SYSTEM_MEMORY_PERCENT}%")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("🛑 Swap监控已停止")
    
    def _monitor_loop(self):
        """增强监控循环 - 包含稳定性保护"""
        while self.monitoring:
            try:
                swap = psutil.swap_memory()
                memory = psutil.virtual_memory()

                # 分阶段Swap管理
                if swap.percent > SWAP_EMERGENCY_THRESHOLD:
                    self.logger.warning(f"🚨 Swap紧急阈值: {swap.percent:.1f}% > {SWAP_EMERGENCY_THRESHOLD}%")
                    self._emergency_cleanup()
                elif swap.percent > SWAP_ACTION_THRESHOLD:
                    self.logger.warning(f"⚠️ Swap行动阈值: {swap.percent:.1f}% > {SWAP_ACTION_THRESHOLD}%")
                    self._moderate_cleanup()
                elif swap.percent > SWAP_WARNING_THRESHOLD:
                    self.logger.info(f"💡 Swap警告阈值: {swap.percent:.1f}% > {SWAP_WARNING_THRESHOLD}%")
                    self._light_cleanup()

                # 检查系统内存 - 使用配置的阈值
                if memory.percent > MAX_SYSTEM_MEMORY_PERCENT:
                    self.logger.warning(f"⚠️ 系统内存使用率过高: {memory.percent:.1f}% > {MAX_SYSTEM_MEMORY_PERCENT}%")
                    self._force_memory_cleanup()

                # 激进内存保护 - 当内存使用率超过95%时
                if memory.percent > 95:
                    self.logger.error(f"🚨 内存使用率危险 {memory.percent:.1f}%，启动激进保护")
                    self.aggressive_memory_protection()

                # 检查GPU内存
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                           '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        used, total = result.stdout.strip().split(', ')
                        gpu_percent = float(used) / float(total) * 100

                        if gpu_percent > MAX_GPU_MEMORY_PERCENT:
                            self.logger.warning(f"🖥️ GPU内存使用率过高: {gpu_percent:.1f}% > {MAX_GPU_MEMORY_PERCENT}%")
                            self._gpu_memory_cleanup()
                except Exception as e:
                    self.logger.debug(f"GPU内存检查失败: {e}")

                # 定期记录内存状态（每5分钟）
                if hasattr(self, '_last_status_log'):
                    if time.time() - self._last_status_log > 300:  # 5分钟
                        self._log_memory_status(swap, memory)
                        self._last_status_log = time.time()
                else:
                    self._last_status_log = time.time()

                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"内存监控错误: {e}")
                time.sleep(self.check_interval)

    def _gpu_memory_cleanup(self):
        """GPU内存清理"""
        self.logger.info("🖥️ 执行GPU内存清理...")
        try:
            import jax
            jax.clear_caches()
            gc.collect()
            self.logger.info("  - GPU缓存清理完成")
        except Exception as e:
            self.logger.warning(f"  - GPU清理失败: {e}")

    def _log_memory_status(self, swap, memory):
        """记录内存状态"""
        self.logger.info(f"📊 内存状态报告:")
        self.logger.info(f"  - 系统内存: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        self.logger.info(f"  - Swap: {swap.used / (1024**3):.1f}GB / {swap.total / (1024**3):.1f}GB ({swap.percent:.1f}%)")

        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, total = result.stdout.strip().split(', ')
                gpu_percent = float(used) / float(total) * 100
                self.logger.info(f"  - GPU内存: {used}MB / {total}MB ({gpu_percent:.1f}%)")
        except:
            pass
    
    def _emergency_cleanup(self):
        """紧急内存清理"""
        self.emergency_cleanup_count += 1
        self.logger.warning(f"🚨 执行紧急内存清理 (第{self.emergency_cleanup_count}次)")
        
        # 1. Python垃圾回收
        collected = gc.collect()
        self.logger.info(f"  - Python GC回收: {collected} 对象")
        
        # 2. 强制释放系统内存 - 增强版
        try:
            subprocess.run(['sync'], timeout=5)
            # 尝试释放页面缓存 (需要权限) - 更激进的清理
            try:
                subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'], timeout=5, check=False)
                self.logger.info("  - 系统缓存清理: drop_caches=3")
            except:
                try:
                    subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=1'], timeout=5, check=False)
                    self.logger.info("  - 系统缓存清理: drop_caches=1")
                except:
                    pass

            # 温和的swap压缩 (不重置，避免数据重新加载)
            try:
                # 只在swap使用率超过80%时才考虑重置
                swap = psutil.swap_memory()
                if swap.percent > 80:
                    subprocess.run(['sudo', 'swapoff', '-a'], timeout=10, check=False)
                    subprocess.run(['sudo', 'swapon', '-a'], timeout=10, check=False)
                    self.logger.info("  - 紧急Swap重置完成 (>80%)")
                else:
                    self.logger.info("  - Swap使用率可控，跳过重置")
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
    
    def _light_cleanup(self):
        """轻度清理 - 25%阈值触发"""
        self.logger.info("🧹 执行轻度内存清理...")
        gc.collect()

    def _moderate_cleanup(self):
        """中度清理 - 40%阈值触发"""
        self.logger.info("🧽 执行中度内存清理...")
        gc.collect()
        try:
            subprocess.run(['sync'], timeout=5)
            subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=1'], timeout=5, check=False)
            self.logger.info("  - 页面缓存清理完成")
        except:
            pass

    def _force_memory_cleanup(self):
        """强制内存清理"""
        gc.collect()
        try:
            subprocess.run(['sync'], timeout=5)
        except:
            pass

    def pre_save_cleanup(self):
        """保存权重前的预清理 - 确保有足够内存"""
        self.logger.info("💾 执行保存前内存预清理...")

        # 记录清理前状态
        swap_before = psutil.swap_memory()
        memory_before = psutil.virtual_memory()

        # 1. 强制Python垃圾回收
        collected = gc.collect()
        self.logger.info(f"  - Python GC回收: {collected} 对象")

        # 2. 清理系统缓存
        try:
            subprocess.run(['sync'], timeout=5)
            subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=1'], timeout=5, check=False)
            self.logger.info("  - 系统缓存清理完成")
        except:
            pass

        # 3. JAX缓存清理
        try:
            import jax
            jax.clear_caches()
            self.logger.info("  - JAX缓存清理完成")
        except:
            pass

        # 记录清理后状态
        swap_after = psutil.swap_memory()
        memory_after = psutil.virtual_memory()

        self.logger.info(f"  - 内存释放: {(memory_before.used - memory_after.used) / (1024**3):.1f}GB")
        self.logger.info(f"  - Swap释放: {(swap_before.used - swap_after.used) / (1024**3):.1f}GB")
        self.logger.info(f"  - 当前可用内存: {memory_after.available / (1024**3):.1f}GB")

        return memory_after.available / (1024**3)  # 返回可用内存GB数

    def aggressive_memory_protection(self):
        """激进内存保护 - 清理无用进程保全训练"""
        self.logger.info("🛡️ 执行激进内存保护...")

        try:
            import psutil
            current_pid = os.getpid()

            # 1. 找出内存占用大的非关键进程
            memory_hogs = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cmdline']):
                try:
                    if proc.info['pid'] == current_pid:
                        continue  # 跳过当前训练进程

                    # 跳过系统关键进程
                    if proc.info['name'] in ['systemd', 'kernel', 'kthreadd', 'init', 'ssh', 'sshd']:
                        continue

                    # 跳过GPU相关进程
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'nvidia' in cmdline.lower() or 'cuda' in cmdline.lower():
                        continue

                    # 找出内存占用超过1%的进程
                    if proc.info['memory_percent'] > 1.0:
                        memory_hogs.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_percent': proc.info['memory_percent'],
                            'cmdline': cmdline[:100]  # 限制长度
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 2. 按内存占用排序
            memory_hogs.sort(key=lambda x: x['memory_percent'], reverse=True)

            # 3. 清理内存占用大的非关键进程
            killed_count = 0
            freed_memory = 0

            for proc in memory_hogs[:5]:  # 最多清理5个进程
                try:
                    # 跳过一些重要的进程
                    if any(keyword in proc['name'].lower() for keyword in ['python', 'wandb', 'tmux', 'screen']):
                        if 'openpi' not in proc['cmdline']:  # 如果不是OpenPI相关的Python进程
                            continue

                    self.logger.info(f"  🎯 清理进程: {proc['name']} (PID: {proc['pid']}, 内存: {proc['memory_percent']:.1f}%)")

                    # 尝试优雅终止
                    process = psutil.Process(proc['pid'])
                    process.terminate()

                    # 等待2秒
                    try:
                        process.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        # 强制杀死
                        process.kill()
                        self.logger.info(f"    💀 强制杀死进程 {proc['pid']}")

                    freed_memory += proc['memory_percent']
                    killed_count += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    continue

            # 4. 清理系统缓存
            try:
                subprocess.run(['sync'], timeout=5)
                subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'], timeout=5, check=False)
                self.logger.info("  🧹 系统缓存清理完成")
            except:
                pass

            self.logger.info(f"✅ 激进清理完成: 清理了{killed_count}个进程，释放约{freed_memory:.1f}%内存")

        except Exception as e:
            self.logger.error(f"❌ 激进内存保护失败: {e}")

def force_gpu_memory_reset():
    """强制GPU内存重置 - 专门用于恢复训练前清理"""
    if not RESUME_TRAINING:
        return

    print("🔄 执行恢复训练前的强制GPU内存重置...")

    try:
        # 1. 尝试重置GPU
        result = subprocess.run(['nvidia-smi', '--gpu-reset'],
                              timeout=15, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ GPU硬件重置成功")
            time.sleep(5)
        else:
            print(f"  ⚠️ GPU硬件重置失败: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️ GPU重置异常: {e}")

    # 2. 强制清理所有Python GPU引用
    try:
        import gc
        # 多轮垃圾回收
        for i in range(5):
            collected = gc.collect()
            print(f"  🗑️ 深度GC第{i+1}轮: {collected} 对象")
            time.sleep(0.5)
    except Exception as e:
        print(f"  ⚠️ GC清理失败: {e}")

    # 3. 清理JAX相关缓存
    try:
        import jax
        jax.clear_caches()
        print("  ✅ JAX缓存清理完成")
    except Exception as e:
        print(f"  ⚠️ JAX清理失败: {e}")

    print("✅ 强制GPU内存重置完成")

def optimize_system_memory():
    """优化系统内存设置 - 使用配置常量，针对恢复训练优化"""
    # GPU内存分配配置 - 恢复训练时更保守的内存分配
    if RESUME_TRAINING:
        # 恢复训练时使用更保守的内存分配，避免碎片化
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = RESUME_GPU_MEM_FRACTION
        print(f"🔄 恢复训练模式：GPU内存分配降低到{RESUME_GPU_MEM_FRACTION}以避免碎片化")
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = GPU_MEM_FRACTION

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # JAX内存管理优化
    os.environ['JAX_ENABLE_X64'] = 'false'
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

    # 恢复训练时强制清理编译缓存
    if RESUME_TRAINING:
        os.environ['JAX_COMPILATION_CACHE_DIR'] = ''
        os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '999999999'
        # 强制重新编译，避免缓存导致的内存问题
        os.environ['JAX_DISABLE_JIT'] = 'false'
        print("🧹 恢复训练模式：强制清理JAX编译缓存")
    else:
        # 禁用编译缓存
        os.environ['JAX_COMPILATION_CACHE_DIR'] = ''
        os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '999999999'

    # XLA优化 - 恢复训练时更保守
    if RESUME_TRAINING:
        # 使用有效的XLA内存优化设置
        os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
        print("🔧 恢复训练模式：使用保守的XLA设置")
    else:
        os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

    # 系统内存优化
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'

    # WandB配置 - 智能模式（在线优先，离线备用）
    # 检测网络连接状态
    def check_wandb_connectivity():
        """检查WandB连接状态"""
        try:
            import requests
            response = requests.get("https://api.wandb.ai/", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    if FORCE_WANDB_OFFLINE:
        # 强制离线模式
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_SILENT'] = 'true'
        print("🔒 强制使用WandB离线模式")
        print("💡 训练完成后可使用 'wandb sync' 手动同步数据")
        wandb_mode = "offline"
    else:
        # 智能模式：检测网络连接
        wandb_online = check_wandb_connectivity()
        if wandb_online:
            # 在线模式
            if 'WANDB_MODE' in os.environ:
                del os.environ['WANDB_MODE']
            os.environ['WANDB_SILENT'] = 'false'
            print("✅ WandB网络连接正常，使用在线模式")
            wandb_mode = "online"
        else:
            # 离线模式备用
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['WANDB_SILENT'] = 'true'
            print("⚠️ WandB网络连接失败，使用离线模式")
            print("💡 训练完成后可使用 'wandb sync' 手动同步数据")
            wandb_mode = "offline"

    print("✅ 主动Swap管理内存优化设置完成")
    print(f"✅ WandB设置为{wandb_mode}模式")

# 应用内存优化
force_gpu_memory_reset()  # 恢复训练前强制重置GPU
optimize_system_memory()

from scripts.train import main

# 创建安全的训练函数，替换原始的WandB初始化
def safe_main(config):
    """安全的训练主函数，使用容错的WandB初始化"""
    # 导入必要的模块
    import scripts.train as train_module

    # 保存原始的init_wandb函数
    original_init_wandb = train_module.init_wandb

    # 替换为我们的安全版本
    def safe_init_wandb_wrapper(config, *, resuming=False, log_code=False, enabled=True):
        success, mode = safe_wandb_init(config, resuming=resuming)
        if success:
            print(f"✅ WandB初始化成功 (模式: {mode})")
        else:
            print("⚠️ WandB初始化失败，但训练将继续")
        return success

    try:
        # 临时替换init_wandb函数
        train_module.init_wandb = safe_init_wandb_wrapper

        # 调用原始的main函数
        return main(config)

    finally:
        # 恢复原始函数
        train_module.init_wandb = original_init_wandb

# 应用内存优化数据加载器补丁
def patch_memory_optimized_data_loader():
    """补丁OpenPI数据加载器使用修复版本"""
    from openpi.training import data_loader as _data_loader
    from openpi.training import config as _config
    from openpi.models import model as _model
    from fixed_dataset import FixedDataset

    # 保存原始函数
    original_create_dataset = _data_loader.create_dataset

    def fixed_create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig):
        """修复版本的create_dataset"""
        repo_id = data_config.repo_id

        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")

        if repo_id == "fake":
            return _data_loader.FakeDataset(model_config, num_samples=1024)

        # 使用修复版数据集
        if repo_id == "custom_real_data":
            print("🚀 使用修复版数据集!")
            print(f"🎬 训练配置: 使用原始时间序列，不进行帧率采样")

            # 计算时间跨度信息（基于数据集原始fps）
            print(f"⏰ 50步动作序列为连续50帧")

            return FixedDataset(
                data_path=DATASET_PATH,
                default_prompt=DEFAULT_PROMPT,
                preload_episodes=None  # 自动加载所有parquet文件
            )

        # 其他情况使用原始函数
        return original_create_dataset(data_config, model_config)

    # 替换函数
    _data_loader.create_dataset = fixed_create_dataset
    print("✅ 已应用修复版数据加载器补丁")

patch_memory_optimized_data_loader()

# 创建全局swap管理器实例
global_swap_manager = None

def patch_checkpoint_save_with_memory_cleanup():
    """补丁保存检查点函数，跳过重复的norm stats保存"""
    from openpi.training import checkpoints as _checkpoints
    import glob
    import shutil

    # 保存原始函数
    original_save_state = _checkpoints.save_state

    def memory_safe_save_state(checkpoint_manager, state, data_loader, step):
        """内存安全的保存状态函数 - 增强版，包含临时目录清理"""
        global global_swap_manager

        # 清理可能存在的临时目录
        try:
            if hasattr(checkpoint_manager, '_directory'):
                checkpoint_dir = str(checkpoint_manager._directory)
            elif hasattr(checkpoint_manager, 'directory'):
                checkpoint_dir = str(checkpoint_manager.directory)
            else:
                checkpoint_dir = None

            if checkpoint_dir:
                temp_pattern = f"{checkpoint_dir}/{step}.orbax-checkpoint-tmp-*"
                temp_dirs = glob.glob(temp_pattern)
                for temp_dir in temp_dirs:
                    try:
                        shutil.rmtree(temp_dir)
                        if global_swap_manager:
                            global_swap_manager.logger.info(f"🧹 清理残留临时目录: {temp_dir}")
                    except Exception as e:
                        if global_swap_manager:
                            global_swap_manager.logger.warning(f"⚠️ 清理临时目录失败: {e}")
        except Exception as e:
            if global_swap_manager:
                global_swap_manager.logger.warning(f"⚠️ 临时目录清理异常: {e}")

        if global_swap_manager is not None:
            # 检查当前内存状态
            memory = psutil.virtual_memory()
            current_usage = memory.percent

            global_swap_manager.logger.info(f"💾 准备保存步骤 {step}，当前内存使用: {current_usage:.1f}%")

            # 如果内存使用率超过90%，执行强制清理
            if current_usage > 90:
                global_swap_manager.logger.warning(f"🚨 内存使用率过高 {current_usage:.1f}%，执行强制清理")
                global_swap_manager._emergency_cleanup()

                # 重新检查内存
                memory = psutil.virtual_memory()
                current_usage = memory.percent
                global_swap_manager.logger.info(f"🧹 清理后内存使用: {current_usage:.1f}%")

            # 保存前预清理
            available_memory = global_swap_manager.pre_save_cleanup()

            # 严格的内存检查 - 需要至少8GB可用内存
            if available_memory < 8.0:
                error_msg = f"❌ 内存不足，无法安全保存权重！可用内存: {available_memory:.1f}GB < 8.0GB"
                global_swap_manager.logger.error(error_msg)
                global_swap_manager.logger.error("🛑 跳过此次保存，避免系统崩溃")

                # 智能处理：记录跳过次数并采取渐进式措施
                if not hasattr(global_swap_manager, 'save_skip_count'):
                    global_swap_manager.save_skip_count = 0
                global_swap_manager.save_skip_count += 1

                global_swap_manager.logger.warning(f"⚠️ 连续跳过保存次数: {global_swap_manager.save_skip_count}")

                # 渐进式内存管理策略
                if global_swap_manager.save_skip_count >= 2:
                    global_swap_manager.logger.error("🚨 连续跳过保存2次，执行深度内存清理")
                    global_swap_manager._deep_memory_cleanup()

                if global_swap_manager.save_skip_count >= 3:
                    global_swap_manager.logger.error("🚨 连续跳过保存3次，建议降低批量大小或重启训练")
                    # 可以在这里添加自动降低批量大小的逻辑

                return  # 直接返回，不执行保存
            else:
                # 保存成功，重置跳过计数
                if hasattr(global_swap_manager, 'save_skip_count'):
                    global_swap_manager.save_skip_count = 0
                global_swap_manager.logger.info(f"✅ 内存充足 {available_memory:.1f}GB，开始保存权重")

        # 执行修改后的保存操作 - 跳过重复的norm stats保存
        try:
            # 创建一个不保存norm stats的save_assets函数
            def skip_norm_save_assets(directory):
                # 跳过norm stats保存，因为我们已经有了
                if global_swap_manager is not None:
                    global_swap_manager.logger.info(f"⏭️ 跳过重复的norm stats保存")
                pass

            # 分离参数用于推理
            from openpi.shared import array_typing as at
            with at.disable_typechecking():
                train_state_for_save, params = _checkpoints._split_params(state)

            # 只保存必要的项目，跳过assets
            items = {
                "train_state": train_state_for_save,
                "params": {"params": params},
            }

            checkpoint_manager.save(step, items)

            if global_swap_manager is not None:
                global_swap_manager.logger.info(f"✅ 步骤 {step} 权重保存成功 (跳过norm stats)")
        except Exception as e:
            if global_swap_manager is not None:
                global_swap_manager.logger.error(f"❌ 步骤 {step} 权重保存失败: {e}")
                # 保存失败后立即清理内存
                global_swap_manager._emergency_cleanup()
            raise

    # 替换函数
    _checkpoints.save_state = memory_safe_save_state
    print("✅ 已应用内存安全的权重保存补丁")

patch_checkpoint_save_with_memory_cleanup()

def setup_logging():
    """简单有效的日志设置"""
    from log_utils import setup_test_logger

    # 创建日志文件和日志器
    optimizer_name = "adamw" if USE_ADAMW else "sgd"
    logger, log_file = setup_test_logger(f"{optimizer_name}_swap_manager", "logs")

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



def safe_wandb_init(config, resuming=False):
    """安全的WandB初始化，支持网络故障容错"""
    try:
        import wandb
        import dataclasses

        if not config.wandb_enabled:
            wandb.init(mode="disabled")
            return True, "disabled"

        # 检测网络连接
        def check_wandb_connectivity():
            try:
                import requests
                response = requests.get("https://api.wandb.ai/", timeout=10)
                return response.status_code == 200
            except Exception:
                return False

        # 检查是否强制离线模式
        if FORCE_WANDB_OFFLINE:
            print("🔒 强制使用WandB离线模式...")
        # 尝试在线模式
        elif check_wandb_connectivity():
            try:
                if resuming:
                    ckpt_dir = config.checkpoint_dir
                    if ckpt_dir.exists() and (ckpt_dir / "wandb_id.txt").exists():
                        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
                        wandb.init(id=run_id, resume="must", project=config.project_name)
                    else:
                        # 恢复训练但没有wandb_id，创建新的run
                        wandb.init(
                            name=config.exp_name,
                            config=dataclasses.asdict(config),
                            project=config.project_name,
                        )
                        if ckpt_dir.exists():
                            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
                else:
                    wandb.init(
                        name=config.exp_name,
                        config=dataclasses.asdict(config),
                        project=config.project_name,
                    )
                    ckpt_dir = config.checkpoint_dir
                    if ckpt_dir.exists():
                        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

                return True, "online"
            except Exception as e:
                print(f"⚠️ WandB在线模式初始化失败: {e}")
                # 降级到离线模式
                pass

        # 离线模式备用
        print("🔄 使用WandB离线模式...")
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        return True, "offline"

    except Exception as e:
        print(f"❌ WandB初始化完全失败: {e}")
        print("🚫 禁用WandB，继续训练...")
        try:
            import wandb
            wandb.init(mode="disabled")
        except:
            pass
        return False, "failed"

def create_dynamic_training_config(initial_batch_size):
    """创建动态训练配置，支持OOM时自动降低批量大小"""
    import openpi.training.config as _config
    from openpi.models import pi0
    from openpi.training import optimizer as _optimizer
    from openpi.training.weight_loaders import CheckpointWeightLoader
    from lora_train_config import CustomDataConfig

    config = _config.TrainConfig(
        name="lora_training",
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
        data=CustomDataConfig(
            data_path=DATASET_PATH,
            default_prompt=DEFAULT_PROMPT,
            dataset_name=DATASET_NAME
        ),

        # 权重加载器 - 使用常量定义
        weight_loader=CheckpointWeightLoader(CHECKPOINT_PATH),

        # 训练参数 - 使用动态批量大小
        batch_size=initial_batch_size,
        num_train_steps=NUM_TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        log_interval=LOG_INTERVAL,
        keep_period=KEEP_PERIOD,  # 保留重要检查点
        num_workers=NUM_WORKERS,

        # 学习率调度 - 使用常量定义
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=WARMUP_STEPS,
            peak_lr=PEAK_LR,
            decay_steps=NUM_TRAIN_STEPS,  # 衰减步数等于总训练步数
            decay_lr=DECAY_LR,
        ),

        # 优化器 - 根据配置选择SGD或AdamW
        optimizer=_optimizer.AdamW(
            b1=ADAMW_B1,
            b2=ADAMW_B2,
            eps=ADAMW_EPS,
            weight_decay=ADAMW_WEIGHT_DECAY,
            clip_gradient_norm=ADAMW_CLIP_NORM,
        ) if USE_ADAMW else _optimizer.SGD(
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
        overwrite=OVERWRITE_CHECKPOINT,
        resume=RESUME_TRAINING,
        wandb_enabled=True,
    )

    return config

def train_with_dynamic_batch_size(capture_logger, swap_manager):
    """使用动态批量大小进行训练"""
    current_batch_size = RESUME_BATCH_SIZE if RESUME_TRAINING else BATCH_SIZE

    # 获取原始的数据加载器函数（避免重复包装）
    import openpi.training.data_loader as _data_loader
    if not hasattr(_data_loader, '_original_create_data_loader'):
        _data_loader._original_create_data_loader = _data_loader.create_data_loader

    def setup_data_loader_patch():
        """设置数据加载器补丁 - 使用norm stats"""
        def use_norm_create_data_loader(config, **kwargs):
            # 启用norm stats，测试单进程是否避免竞争
            kwargs['skip_norm_stats'] = False
            return _data_loader._original_create_data_loader(config, **kwargs)

        _data_loader.create_data_loader = use_norm_create_data_loader

    while current_batch_size >= MIN_BATCH_SIZE:
        try:
            capture_logger.info(f"🎯 尝试批量大小: {current_batch_size}")

            # 创建配置
            config = create_dynamic_training_config(current_batch_size)

            capture_logger.info(f"🔧 训练配置:")
            capture_logger.info(f"  - 批量大小: {current_batch_size}")
            capture_logger.info(f"  - 训练步数: {NUM_TRAIN_STEPS}")
            capture_logger.info(f"  - GPU内存分配: {RESUME_GPU_MEM_FRACTION if RESUME_TRAINING else GPU_MEM_FRACTION}")
            capture_logger.info(f"  - 恢复训练: {'是' if RESUME_TRAINING else '否'}")

            # 设置数据加载器补丁
            setup_data_loader_patch()
            capture_logger.info("✅ 已启用norm stats (单进程测试)")

            # 运行训练
            import openpi.shared.array_typing as at
            with at.disable_typechecking():
                capture_logger.info("🎯 开始训练...")
                safe_main(config)

            capture_logger.info("✅ 训练成功完成!")
            return True, current_batch_size

        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "Out of memory" in error_msg or "RecursionError" in error_msg:
                capture_logger.warning(f"❌ 批量大小 {current_batch_size} 失败: {type(e).__name__}")
                current_batch_size -= 1

                if current_batch_size >= MIN_BATCH_SIZE:
                    capture_logger.info(f"🔄 降低批量大小到 {current_batch_size}，重新尝试...")
                    # 强制内存清理和重置
                    import gc
                    gc.collect()

                    # 重置数据加载器函数
                    _data_loader.create_data_loader = _data_loader._original_create_data_loader
                    time.sleep(2)
                else:
                    capture_logger.error(f"❌ 已达到最小批量大小 {MIN_BATCH_SIZE}，仍然失败")
                    raise
            else:
                capture_logger.error(f"❌ 非内存相关错误: {e}")
                raise

    capture_logger.error(f"❌ 所有批量大小都失败，最小批量大小 {MIN_BATCH_SIZE} 仍然失败")
    return False, MIN_BATCH_SIZE

def create_swap_managed_config():
    """创建主动Swap管理的SGD配置"""
    import openpi.training.config as _config
    from openpi.models import pi0
    from openpi.training import optimizer as _optimizer
    from openpi.training.weight_loaders import CheckpointWeightLoader
    from lora_train_config import CustomDataConfig

    config = _config.TrainConfig(
        name="lora_training",
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
        data=CustomDataConfig(
            data_path=DATASET_PATH,
            default_prompt=DEFAULT_PROMPT,
            dataset_name=DATASET_NAME
        ),

        # 权重加载器 - 使用常量定义
        weight_loader=CheckpointWeightLoader(CHECKPOINT_PATH),

        # 训练参数 - 使用常量定义，恢复训练时调整批量大小
        batch_size=RESUME_BATCH_SIZE if RESUME_TRAINING else BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        log_interval=LOG_INTERVAL,
        keep_period=KEEP_PERIOD,  # 保留重要检查点
        num_workers=NUM_WORKERS,

        # 学习率调度 - 使用常量定义
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=WARMUP_STEPS,
            peak_lr=PEAK_LR,
            decay_steps=NUM_TRAIN_STEPS,  # 衰减步数等于总训练步数
            decay_lr=DECAY_LR,
        ),

        # 优化器 - 根据配置选择SGD或AdamW
        optimizer=_optimizer.AdamW(
            b1=ADAMW_B1,
            b2=ADAMW_B2,
            eps=ADAMW_EPS,
            weight_decay=ADAMW_WEIGHT_DECAY,
            clip_gradient_norm=ADAMW_CLIP_NORM,
        ) if USE_ADAMW else _optimizer.SGD(
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
        overwrite=OVERWRITE_CHECKPOINT,
        resume=RESUME_TRAINING,
        wandb_enabled=True,
    )

    return config

if __name__ == "__main__":
    # 设置基础日志
    logger, log_file = setup_logging()

    optimizer_name = "AdamW" if USE_ADAMW else "SGD"
    print(f"🚀 {optimizer_name}主动Swap管理训练 - 20K步正式训练")
    print("=" * 60)
    print("🎯 目标: 20000步持续训练，每2000步保存权重")
    print("💡 策略: 实时监控 + 主动清理 + WandB在线记录 + Terminal输出捕获")
    print(f"⚙️ 优化器: {optimizer_name} (批量大小: {BATCH_SIZE}, GPU内存: {GPU_MEM_FRACTION})")
    print(f"📝 日志文件: {log_file}")
    print(f"🔄 恢复训练: {'启用' if RESUME_TRAINING else '禁用'}")
    print(f"📁 覆盖检查点: {'是' if OVERWRITE_CHECKPOINT else '否'}")
    print("=" * 60)

    # 使用TerminalOutputCapture捕获所有输出
    from log_utils import TerminalOutputCapture

    with TerminalOutputCapture(log_file, f"{optimizer_name}_20k_training") as capture_logger:
        capture_logger.info("🔍 开始捕获所有terminal输出到日志文件")

        # 创建Swap管理器
        swap_manager = SwapManager(capture_logger, max_swap_percent=SWAP_EMERGENCY_THRESHOLD, check_interval=MONITOR_INTERVAL)

        # 设置全局swap管理器，用于权重保存时的内存管理
        global_swap_manager = swap_manager

        start_time = time.time()

        try:
            # 记录初始内存状态
            log_system_memory(capture_logger, "初始状态")

            # 执行训练前的swap清理
            swap_freed, memory_freed = swap_manager.initial_swap_cleanup()

            # 启动Swap监控
            swap_manager.start_monitoring()
            log_system_memory(capture_logger, "训练开始前")

            # 使用动态批量大小训练
            if DYNAMIC_BATCH_ADJUSTMENT and RESUME_TRAINING:
                capture_logger.info("� 启用动态批量大小调整...")
                success, final_batch_size = train_with_dynamic_batch_size(capture_logger, swap_manager)

                if success:
                    capture_logger.info(f"✅ 动态训练成功完成! 最终批量大小: {final_batch_size}")
                else:
                    capture_logger.error("❌ 动态训练失败")
                    raise RuntimeError("动态批量大小调整失败")
            else:
                # 传统训练方式
                config = create_swap_managed_config()

                capture_logger.info("🔧 训练配置:")
                capture_logger.info(f"  - 批量大小: {config.batch_size}")
                capture_logger.info(f"  - 训练步数: {NUM_TRAIN_STEPS}")
                capture_logger.info(f"  - 预热步数: {WARMUP_STEPS} ({WARMUP_STEPS/NUM_TRAIN_STEPS*100:.1f}%)")
                capture_logger.info(f"  - 保存间隔: {SAVE_INTERVAL}")
                capture_logger.info(f"  - 学习率: {PEAK_LR:.2e} -> {DECAY_LR:.2e}")
                capture_logger.info(f"  - GPU内存分配: {RESUME_GPU_MEM_FRACTION if RESUME_TRAINING else GPU_MEM_FRACTION}")
                capture_logger.info(f"  - Swap阈值: 警告{SWAP_WARNING_THRESHOLD}% / 行动{SWAP_ACTION_THRESHOLD}% / 紧急{SWAP_EMERGENCY_THRESHOLD}%")

                # 使用归一化统计（避免重复包装）
                import openpi.training.data_loader as _data_loader
                if not hasattr(_data_loader, '_original_create_data_loader'):
                    _data_loader._original_create_data_loader = _data_loader.create_data_loader

                def use_norm_create_data_loader(config, **kwargs):
                    # 启用norm stats，测试单进程是否避免竞争
                    kwargs['skip_norm_stats'] = False
                    return _data_loader._original_create_data_loader(config, **kwargs)

                _data_loader.create_data_loader = use_norm_create_data_loader
                capture_logger.info("✅ 已启用norm stats (单进程测试)")

                # 运行训练
                import openpi.shared.array_typing as at
                with at.disable_typechecking():
                    capture_logger.info("🎯 开始训练...")
                    safe_main(config)

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
