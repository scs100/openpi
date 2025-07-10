#!/usr/bin/env python3
"""
OpenPI推理脚本 - Python 3.11版本
使用保存的OpenPI兼容数据进行真正的OpenPI推理
集成ACT风格时间聚合功能
"""

import time
import json
import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import threading
import queue

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入控制接口
try:
    from control_interface import InferenceControlClient
    CONTROL_INTERFACE_AVAILABLE = True
    print("✅ 控制接口可用")
except ImportError:
    print("❌ 控制接口不可用")
    CONTROL_INTERFACE_AVAILABLE = False

# 导入OpenPI推理引擎
try:
    from inference_agilexv2_openpi import PolicyInference
    OPENPI_AVAILABLE = True
    print("✅ OpenPI推理引擎可用")
except ImportError as e:
    print(f"❌ OpenPI推理引擎不可用: {e}")
    OPENPI_AVAILABLE = False


class PolynomialTrajectoryPlanner:
    """五次多项式轨迹规划器 - 提供平滑的关节运动轨迹"""

    def __init__(self, dof=7, interpolation_freq=200):
        """
        初始化轨迹规划器

        Args:
            dof: 自由度数量（关节数）
            interpolation_freq: 插值频率 (Hz)
        """
        self.dof = dof
        self.freq = interpolation_freq
        self.dt = 1.0 / interpolation_freq

        # 轨迹状态
        self.current_position = np.zeros(dof)
        self.current_velocity = np.zeros(dof)
        self.current_acceleration = np.zeros(dof)

        # 目标状态
        self.target_position = np.zeros(dof)
        self.target_velocity = np.zeros(dof)
        self.target_acceleration = np.zeros(dof)

        # 轨迹参数
        self.trajectory_time = 0.5  # 默认轨迹时间500ms
        self.is_moving = False
        self.trajectory_start_time = 0

        print(f"🎯 多项式轨迹规划器初始化: {dof}DOF, {interpolation_freq}Hz")

    def set_trajectory_time(self, time_seconds):
        """设置轨迹执行时间"""
        self.trajectory_time = max(0.1, time_seconds)  # 最小100ms
        print(f"⏱️  轨迹时间设置为: {self.trajectory_time:.3f}s")

    def quintic_polynomial(self, t, T, p0, pf, v0=0, vf=0, a0=0, af=0):
        """
        五次多项式轨迹计算

        Args:
            t: 当前时间
            T: 总时间
            p0: 起始位置
            pf: 终止位置
            v0: 起始速度
            vf: 终止速度
            a0: 起始加速度
            af: 终止加速度

        Returns:
            (position, velocity, acceleration)
        """
        if T <= 0:
            return pf, np.zeros_like(pf), np.zeros_like(pf)

        # 归一化时间
        tau = np.clip(t / T, 0, 1)

        # 五次多项式系数
        a0_coeff = p0
        a1_coeff = v0
        a2_coeff = a0 / 2
        a3_coeff = (20 * (pf - p0) - (8 * vf + 12 * v0) * T - (3 * af - a0) * T**2) / (2 * T**3)
        a4_coeff = (30 * (p0 - pf) + (14 * vf + 16 * v0) * T + (3 * af - 2 * a0) * T**2) / (2 * T**4)
        a5_coeff = (12 * (pf - p0) - (6 * vf + 6 * v0) * T - (af - a0) * T**2) / (2 * T**5)

        # 计算位置、速度、加速度
        position = (a0_coeff + a1_coeff * tau + a2_coeff * tau**2 +
                   a3_coeff * tau**3 + a4_coeff * tau**4 + a5_coeff * tau**5)

        velocity = (a1_coeff + 2 * a2_coeff * tau + 3 * a3_coeff * tau**2 +
                   4 * a4_coeff * tau**3 + 5 * a5_coeff * tau**4) / T

        acceleration = (2 * a2_coeff + 6 * a3_coeff * tau + 12 * a4_coeff * tau**2 +
                       20 * a5_coeff * tau**3) / (T**2)

        return position, velocity, acceleration

    def start_trajectory(self, target_joints, current_joints=None):
        """
        开始新的轨迹规划

        Args:
            target_joints: 目标关节角度
            current_joints: 当前关节角度（如果为None，使用上次的目标位置）
        """
        if current_joints is not None:
            self.current_position = np.array(current_joints)
        else:
            # 使用上次的目标位置作为当前位置
            self.current_position = self.target_position.copy()

        self.target_position = np.array(target_joints)

        # 重置速度和加速度（平滑启动）
        self.current_velocity = np.zeros(self.dof)
        self.current_acceleration = np.zeros(self.dof)
        self.target_velocity = np.zeros(self.dof)
        self.target_acceleration = np.zeros(self.dof)

        self.is_moving = True
        self.trajectory_start_time = time.time()

        # 计算最大关节变化，动态调整轨迹时间
        max_joint_change = np.max(np.abs(self.target_position - self.current_position))
        if max_joint_change > 0.5:  # 大幅度运动
            self.trajectory_time = min(1.0, max_joint_change * 0.8)
        else:  # 小幅度运动
            self.trajectory_time = max(0.2, max_joint_change * 0.6)

        print(f"🚀 开始轨迹规划: 最大变化{max_joint_change:.3f}rad, 时间{self.trajectory_time:.3f}s")

    def get_current_trajectory_point(self):
        """
        获取当前轨迹点

        Returns:
            (position, velocity, acceleration, is_finished)
        """
        if not self.is_moving:
            return self.target_position, np.zeros(self.dof), np.zeros(self.dof), True

        current_time = time.time()
        elapsed_time = current_time - self.trajectory_start_time

        if elapsed_time >= self.trajectory_time:
            # 轨迹完成
            self.is_moving = False
            self.current_position = self.target_position.copy()
            self.current_velocity = np.zeros(self.dof)
            self.current_acceleration = np.zeros(self.dof)
            return self.target_position, np.zeros(self.dof), np.zeros(self.dof), True

        # 计算当前轨迹点
        position, velocity, acceleration = self.quintic_polynomial(
            elapsed_time, self.trajectory_time,
            self.current_position, self.target_position,
            self.current_velocity, self.target_velocity,
            self.current_acceleration, self.target_acceleration
        )

        return position, velocity, acceleration, False


class InferenceCache:
    """推理结果缓存，用于ACT风格的时间聚合"""

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = []  # 存储推理结果

    def add_inference_result(self, step_id, timestamp, actions, inference_time):
        """添加推理结果到缓存"""
        result = {
            'step_id': step_id,
            'timestamp': timestamp,
            'actions': actions,  # shape: (action_horizon, action_dim)
            'inference_time': inference_time
        }

        self.cache.append(result)

        # 保持缓存大小限制
        if len(self.cache) > self.max_size:
            self.cache.pop(0)

    def get_recent_results(self, count=5):
        """获取最近的N个推理结果"""
        return self.cache[-count:] if len(self.cache) >= count else self.cache

    def clear(self):
        """清空缓存"""
        self.cache.clear()


class TemporalEnsemble:
    """ACT风格的时间聚合器 - 基于真实时间对齐"""

    def __init__(self, ensemble_size=5, action_horizon=50, action_dt=0.03):
        self.ensemble_size = ensemble_size
        self.action_horizon = action_horizon
        self.action_dt = action_dt  # 每个action之间的时间间隔（秒）

    def temporal_ensemble_actions(self, recent_results, current_time, target_time_offset=0.0):
        """
        基于真实时间的ACT风格时间聚合函数

        Args:
            recent_results: 最近的推理结果列表
            current_time: 当前时间戳
            target_time_offset: 目标时间偏移（秒，0表示当前时间的动作）

        Returns:
            聚合后的动作向量
        """
        if not recent_results or len(recent_results) < 2:
            return None

        valid_actions = []
        weights = []
        debug_info = []

        target_time = current_time + target_time_offset

        for i, result in enumerate(recent_results):
            # 计算这个推理结果的基准时间（推理完成时间）
            inference_time = result['timestamp']

            # 计算目标时间相对于推理时间的偏移
            time_offset = target_time - inference_time

            # 将时间偏移转换为动作序列索引 - 使用floor而不是round，确保不会超前选择
            action_index = int(np.floor(time_offset / self.action_dt))

            # 检查动作索引是否在有效范围内
            if 0 <= action_index < result['actions'].shape[0]:
                action = result['actions'][action_index]

                # 计算时间权重 - 越新的推理权重越高
                time_weight = np.exp(-0.1 * (len(recent_results) - i - 1))

                # 计算时间对齐权重 - 时间偏移越小权重越高
                time_alignment_weight = np.exp(-0.05 * abs(time_offset))

                # 计算动作索引权重 - 索引越小（越接近当前）权重越高
                index_weight = np.exp(-0.02 * action_index)

                combined_weight = time_weight * time_alignment_weight * index_weight

                valid_actions.append(action)
                weights.append(combined_weight)

                # 调试信息
                debug_info.append({
                    'inference_time': inference_time,
                    'time_offset': time_offset,
                    'action_index': action_index,
                    'weight': combined_weight
                })

        if not valid_actions:
            return None

        # 加权平均聚合
        valid_actions = np.array(valid_actions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 归一化权重

        aggregated_action = np.average(valid_actions, axis=0, weights=weights)

        return aggregated_action, len(valid_actions), weights, debug_info


class AsyncInferenceController:
    """异步推理控制器 - 真正的异步实现，基于真实时间对齐，带动作滤波"""

    def __init__(self, data_provider, inference_engine, command_sender,
                 history_size=8, target_time_offset=0.6, action_dt=0.03,
                 filter_alpha=0.3, execution_protection_time=0.4, debug_delay=0.0):
        self.data_provider = data_provider
        self.inference_engine = inference_engine
        self.command_sender = command_sender

        # 时间聚合组件 - 基于真实时间
        self.inference_cache = InferenceCache(max_size=100)
        self.temporal_ensemble = TemporalEnsemble(ensemble_size=history_size, action_horizon=50, action_dt=action_dt)
        self.history_size = history_size  # 使用历史多少次推理数据
        self.target_time_offset = target_time_offset  # 目标时间偏移（秒）
        self.action_dt = action_dt  # 每个action之间的时间间隔（秒）

        # 动作滤波器
        self.filter_alpha = filter_alpha  # 滤波系数 (0-1, 越小越平滑)
        self.last_filtered_action = None  # 上次滤波后的动作

        # 执行完整性保护
        self.execution_protection_time = execution_protection_time  # 执行保护时间（秒）
        self.last_command_time = 0.0  # 上次发送命令的时间
        self.command_target_time = 0.0  # 当前命令的目标完成时间

        # 调试延迟配置 (秒) - 用于手动控制推理频率
        self.debug_delay = debug_delay

        # 线程间通信
        self.latest_obs = None
        self.obs_lock = threading.Lock()

        # 最新命令存储 - 只保存最新的命令，不使用队列
        self.latest_command = None
        self.command_lock = threading.Lock()
        self.command_available = threading.Event()  # 用于通知有新命令

        # 控制标志
        self.running = False
        self.threads = []

        # 统计信息
        self.inference_count = 0
        self.command_count = 0
        self.start_time = None
        self.stats_lock = threading.Lock()

    def start(self):
        """启动异步控制"""
        print("🚀 启动异步推理控制系统...")
        self.running = True
        self.start_time = time.time()

        # 启动数据更新线程
        data_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        data_thread.start()
        self.threads.append(data_thread)

        # 启动推理线程
        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()
        self.threads.append(inference_thread)

        # 启动命令执行线程
        command_thread = threading.Thread(target=self._command_loop, daemon=True)
        command_thread.start()
        self.threads.append(command_thread)

        # 启动统计线程
        stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        stats_thread.start()
        self.threads.append(stats_thread)

        print("✅ 异步系统启动完成")
        print("   📡 数据更新线程: 30Hz实时获取最新观测数据")
        print("   🧠 推理线程: 持续进行模型推理和时间聚合")
        print("   🤖 命令线程: 异步执行机械臂控制命令")
        print("   📊 统计线程: 实时显示性能统计")

    def stop(self):
        """停止异步控制"""
        print("\n⏹️  停止异步推理控制系统...")
        self.running = False

        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=1.0)

        print("✅ 异步系统已停止")

    def _data_update_loop(self):
        """数据更新线程 - 持续获取最新观测数据"""
        print("📡 数据更新线程启动")

        data_read_count = 0
        successful_reads = 0
        total_data_time = 0.0
        last_data_time = time.time()
        no_data_count = 0

        while self.running:
            try:
                # 获取最新观测数据并计时
                data_start = time.perf_counter()
                obs = self.data_provider.get_observation()
                data_time = (time.perf_counter() - data_start) * 1000  # 毫秒
                data_read_count += 1

                if obs and 'arm_joints' in obs and 'images' in obs:
                    # 成功获取数据
                    successful_reads += 1
                    no_data_count = 0  # 重置无数据计数
                    last_data_time = time.time()

                    # 添加数据获取时间到观测数据
                    obs['data_acquisition_time'] = data_time

                    with self.obs_lock:
                        self.latest_obs = obs

                    # 统计数据获取性能
                    total_data_time += data_time

                    # 每200次成功读取显示一次平均性能
                    if successful_reads % 200 == 0:
                        avg_data_time = total_data_time / successful_reads
                        success_rate = (successful_reads / data_read_count) * 100
                        print(f"📊 数据获取性能: 平均{avg_data_time:.2f}ms/次, 成功率{success_rate:.1f}% ({successful_reads}/{data_read_count})")
                else:
                    # 无数据情况
                    no_data_count += 1

                    # 每1000次无数据时显示警告
                    if no_data_count % 1000 == 0:
                        time_since_last = time.time() - last_data_time
                        print(f"⚠️  持续无数据 {no_data_count} 次，距离上次数据 {time_since_last:.1f}秒")
                        print("   请检查 simple_memory_service.py 是否正常运行")

                # 高频率更新 - 30Hz
                time.sleep(0.033)

            except Exception as e:
                print(f"❌ 数据更新错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        print("📡 数据更新线程结束")

    def _inference_loop(self):
        """推理线程 - 持续进行推理和时间聚合"""
        print("🧠 推理线程启动")

        no_data_wait_count = 0
        last_inference_time = time.time()

        while self.running:
            try:
                # 获取最新观测数据
                data_get_start = time.perf_counter()
                with self.obs_lock:
                    current_obs = self.latest_obs
                data_get_time = (time.perf_counter() - data_get_start) * 1000

                if current_obs is None:
                    no_data_wait_count += 1

                    # 每5000次等待显示一次状态
                    if no_data_wait_count % 5000 == 0:
                        wait_time = time.time() - last_inference_time
                        print(f"🧠 推理线程等待数据: {no_data_wait_count} 次等待, {wait_time:.1f}秒无推理")

                    time.sleep(0.01)
                    continue

                # 有数据时重置等待计数
                if no_data_wait_count > 0:
                    print(f"🧠 推理线程恢复: 等待了 {no_data_wait_count} 次")
                    no_data_wait_count = 0

                # 提取数据获取时间（如果有）
                data_acquisition_time = current_obs.get('data_acquisition_time', 0.0)

                # 执行推理
                inference_start = time.perf_counter()
                action = self.inference_engine.infer(current_obs)
                inference_time = (time.perf_counter() - inference_start) * 1000
                last_inference_time = time.time()  # 更新最后推理时间

                if action is None:
                    continue

                with self.stats_lock:
                    self.inference_count += 1
                    current_count = self.inference_count

                # 添加到推理缓存
                if isinstance(action, np.ndarray) and len(action.shape) == 2:
                    self.inference_cache.add_inference_result(
                        step_id=current_count,
                        timestamp=time.time(),
                        actions=action,
                        inference_time=inference_time
                    )

                # 每次推理都尝试时间聚合 - 使用历史5次数据
                final_action = None
                use_ensemble = False
                current_time = time.time()

                if len(self.inference_cache.cache) >= 2:
                    # 每次都执行基于时间的聚合，使用历史N次数据
                    recent_results = self.inference_cache.get_recent_results(self.history_size)
                    ensemble_result = self.temporal_ensemble.temporal_ensemble_actions(
                        recent_results, current_time, target_time_offset=self.target_time_offset
                    )

                    if ensemble_result is not None:
                        final_action, num_predictions, weights, debug_info = ensemble_result
                        use_ensemble = True

                        # 显示时间对齐调试信息（每10次显示一次详细信息）
                        if current_count % 10 == 0:
                            avg_time_offset = np.mean([info['time_offset'] for info in debug_info])
                            avg_action_index = np.mean([info['action_index'] for info in debug_info])
                            print(f"🔄 时间聚合[{current_count}]: 使用{num_predictions}个历史预测")
                            print(f"   ⏰ 平均时间偏移: {avg_time_offset*1000:.1f}ms, 平均动作索引: {avg_action_index:.1f}")
                            print(f"   ⚖️  权重范围: [{weights.min():.3f}-{weights.max():.3f}]")

                if final_action is None:
                    # 使用原始推理结果 - 基于时间偏移计算索引
                    if isinstance(action, np.ndarray) and len(action.shape) == 2:
                        # 计算目标时间对应的动作索引
                        target_index = int(round(self.target_time_offset / self.action_dt))
                        target_index = min(target_index, action.shape[0] - 1)  # 确保不超出范围
                        final_action = action[target_index]
                        print(f"📊 单次推理[{current_count}]: 使用索引{target_index} (时间偏移{self.target_time_offset*1000:.0f}ms)")
                    else:
                        final_action = action

                # 详细性能日志 - 每10次推理显示一次
                if current_count % 10 == 0:
                    ensemble_tag = "🔄聚合" if use_ensemble else "📊单次"
                    print(f"⏱️  性能[{current_count}] {ensemble_tag}: "
                          f"数据获取{data_acquisition_time:.1f}ms + "
                          f"数据读取{data_get_time:.2f}ms + "
                          f"推理{inference_time:.1f}ms = "
                          f"总计{data_acquisition_time + data_get_time + inference_time:.1f}ms")

                # 应用动作滤波和执行时间感知调度
                if final_action is not None and len(final_action) >= 14:
                    current_time_cmd = time.time()

                    # 检查是否应该等待当前命令执行完成
                    time_since_last = current_time_cmd - self.last_command_time

                    # 执行完整性保护：确保机械臂有足够时间执行到目标位置
                    execution_time_needed = self.target_time_offset * 0.7  # 预测时间的70%作为执行保护时间
                    should_wait_for_execution = time_since_last < execution_time_needed

                    if should_wait_for_execution:
                        # 等待当前命令执行期间：不发送新命令，保持当前动作
                        filtered_action = self.last_filtered_action.copy() if self.last_filtered_action is not None else final_action.copy()

                        if current_count % 2 == 0:  # 更频繁显示保护信息
                            wait_time = execution_time_needed - time_since_last
                            protection_progress = time_since_last / execution_time_needed * 100
                            print(f"⏳ 执行保护[{current_count}]: 进度{protection_progress:.0f}%, 还需{wait_time*1000:.0f}ms (保持当前命令)")

                    else:
                        # 可以发送新命令：直接使用推理结果，完全无滤波
                        filtered_action = final_action.copy()  # 直接使用推理结果，无任何滤波

                        command_data = {
                            'action': filtered_action,
                            'inference_id': current_count,
                            'use_ensemble': use_ensemble,
                            'inference_time': inference_time,
                            'timestamp': current_time_cmd,
                            'filter_applied': True,
                            'time_since_last': time_since_last,
                            'execution_protection': False
                        }

                        with self.command_lock:
                            self.latest_command = command_data
                            self.command_available.set()  # 通知命令线程有新命令

                        self.last_command_time = current_time_cmd

                        # 显示命令发送信息
                        action_diff = np.linalg.norm(final_action - self.last_filtered_action) if self.last_filtered_action is not None else 0
                        print(f"🚀 新命令[{current_count}]: 无滤波直接执行, 动作变化{action_diff:.4f}, 执行间隔{time_since_last*1000:.0f}ms")

                    # 更新滤波历史
                    self.last_filtered_action = filtered_action.copy()

                # 调试延迟 - 可配置的手动延迟
                if self.debug_delay > 0:
                    time.sleep(self.debug_delay)

            except Exception as e:
                print(f"❌ 推理错误: {e}")
                time.sleep(0.01)

        print("🧠 推理线程结束")

    def _command_loop(self):
        """命令执行线程 - 真正异步执行，只执行最新命令"""
        print("🤖 命令执行线程启动")

        current_command_id = None  # 当前正在执行的命令ID
        executing = False  # 是否正在执行命令

        while self.running:
            try:
                # 等待新命令或超时
                if self.command_available.wait(timeout=0.1):
                    # 获取最新命令
                    with self.command_lock:
                        if self.latest_command is not None:
                            command_data = self.latest_command.copy()
                            self.latest_command = None  # 清空，避免重复执行
                            self.command_available.clear()
                        else:
                            continue

                    # 检查是否需要执行新命令
                    new_command_id = command_data['inference_id']

                    if executing and current_command_id is not None:
                        # 如果正在执行命令，跳过旧命令，只执行最新的
                        if new_command_id <= current_command_id:
                            print(f"⏭️  跳过旧命令[{new_command_id}]，当前执行[{current_command_id}]")
                            continue
                        else:
                            print(f"🔄 新命令[{new_command_id}]覆盖旧命令[{current_command_id}]")

                    # 执行新命令
                    executing = True
                    current_command_id = new_command_id

                    action = command_data['action']
                    use_ensemble = command_data['use_ensemble']
                    inference_time = command_data['inference_time']

                    # 解析动作
                    if len(action) >= 14:
                        left_joints = action[:7]
                        right_joints = action[7:14]

                        # 发送命令（非阻塞）
                        ensemble_tag = "🔄聚合" if use_ensemble else "📊单次"
                        print(f"🚀 执行[{current_command_id}] {ensemble_tag}: 推理{inference_time:.1f}ms")

                        # 异步发送命令，不等待完成
                        success = self.command_sender.send_joint_command(left_joints, right_joints)

                        if success:
                            with self.stats_lock:
                                self.command_count += 1
                            print(f"✅ 命令[{current_command_id}]已发送")
                        else:
                            print(f"❌ 命令[{current_command_id}]发送失败")

                    executing = False
                    current_command_id = None

            except Exception as e:
                print(f"❌ 命令执行错误: {e}")
                executing = False
                current_command_id = None

        print("🤖 命令执行线程结束")

    def _stats_loop(self):
        """统计线程 - 实时显示性能统计"""
        print("📊 统计线程启动")

        last_inference_count = 0
        last_command_count = 0
        last_time = time.time()

        while self.running:
            try:
                time.sleep(2.0)  # 每2秒显示一次统计

                current_time = time.time()

                with self.stats_lock:
                    current_inference = self.inference_count
                    current_command = self.command_count

                # 计算频率
                time_diff = current_time - last_time
                inference_freq = (current_inference - last_inference_count) / time_diff
                command_freq = (current_command - last_command_count) / time_diff

                # 计算总体统计
                total_time = current_time - self.start_time
                avg_inference_freq = current_inference / total_time if total_time > 0 else 0
                avg_command_freq = current_command / total_time if total_time > 0 else 0

                # 最新命令状态
                with self.command_lock:
                    has_pending_command = self.latest_command is not None

                print(f"\n📊 异步系统状态 [{datetime.now().strftime('%H:%M:%S')}]:")
                print(f"   🧠 推理: {inference_freq:.1f}Hz (总计{current_inference}, 平均{avg_inference_freq:.1f}Hz)")
                print(f"   🤖 命令: {command_freq:.1f}Hz (总计{current_command}, 平均{avg_command_freq:.1f}Hz)")
                print(f"   📦 待执行: {'有' if has_pending_command else '无'}最新命令")
                print(f"   ⏱️  运行: {total_time:.1f}s")

                last_inference_count = current_inference
                last_command_count = current_command
                last_time = current_time

            except Exception as e:
                print(f"❌ 统计错误: {e}")

        print("📊 统计线程结束")

    def get_stats(self):
        """获取当前统计信息"""
        with self.stats_lock:
            total_time = time.time() - self.start_time if self.start_time else 0
            return {
                'inference_count': self.inference_count,
                'command_count': self.command_count,
                'total_time': total_time,
                'inference_freq': self.inference_count / total_time if total_time > 0 else 0,
                'command_freq': self.command_count / total_time if total_time > 0 else 0,
                'has_pending_command': self.latest_command is not None
            }


# 导入套接字客户端
from memory_socket_client import MemoryDataProvider





class OpenPIInferenceEngine:
    """OpenPI推理引擎 - 真正的OpenPI推理"""

    def __init__(self, config_path=None):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI推理引擎不可用")

        if config_path is None:
            config_path = "/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml"

        self.step_count = 0
        print(f"🤖 初始化OpenPI推理引擎: {config_path}")

        # 创建调试图像目录
        self.debug_dir = Path("debug_images")
        self.debug_dir.mkdir(exist_ok=True)
        print(f"📁 调试图像目录: {self.debug_dir}")

        try:
            # 直接初始化OpenPI推理引擎，不使用JointInference的数据获取功能
            from inference_agilexv2_openpi import PolicyInference
            self.inference_engine = PolicyInference(host="localhost", port=8000)
            print("✅ OpenPI推理引擎初始化成功")
        except Exception as e:
            print(f"❌ OpenPI推理引擎初始化失败: {e}")
            self.inference_engine = None
            raise

    def _save_debug_image(self, img, cam_name, step_count):
        """保存调试图像"""
        try:
            # 将float32图像转换为uint8用于保存
            if img.dtype == np.float32:
                # 假设图像已经归一化到[0,1]范围
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img

            # 保存图像
            filename = self.debug_dir / f"step_{step_count:04d}_{cam_name}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            print(f"💾 保存调试图像: {filename}")

        except Exception as e:
            print(f"❌ 保存调试图像失败 {cam_name}: {e}")

    def infer(self, obs):
        """执行OpenPI推理"""
        try:
            self.step_count += 1

            print(f"🤖 OpenPI推理: 步骤{self.step_count}")

            # 验证观测数据格式
            if 'arm_joints' not in obs or 'images' not in obs:
                print("❌ 观测数据格式错误")
                return None

            # 检查图像数据并保存调试图像
            required_cameras = ['left_wrist_0_rgb', 'right_wrist_0_rgb', 'base_0_rgb']
            debug_images = {}

            for cam_name in required_cameras:
                if cam_name not in obs['images']:
                    print(f"❌ 缺少相机数据: {cam_name}")
                    return None

                img = obs['images'][cam_name]
                # 套接字数据提供器已经转换为float32格式
                if img.shape != (224, 224, 3) or img.dtype != np.float32:
                    print(f"❌ 图像格式错误 {cam_name}: {img.shape}, {img.dtype}")
                    return None

                # 保存调试图像信息
                debug_images[cam_name] = {
                    'shape': img.shape,
                    'dtype': img.dtype,
                    'min': img.min(),
                    'max': img.max(),
                    'mean': img.mean()
                }

                # 每30步保存一次图像用于调试
                if self.step_count % 30 == 1:
                    self._save_debug_image(img, cam_name, self.step_count)

            # 显示图像调试信息
            if self.step_count % 5 == 1:
                print(f"🖼️  图像调试信息 [步骤{self.step_count}]:")
                for cam_name, info in debug_images.items():
                    print(f"   📷 {cam_name}: {info['shape']} {info['dtype']} "
                          f"范围[{info['min']:.3f}-{info['max']:.3f}] 均值{info['mean']:.3f}")

            # 检查关节数据
            if 'arm_joints' in obs:
                joints = obs['arm_joints']
                if self.step_count % 5 == 1:
                    print(f"🦾 关节调试信息 [步骤{self.step_count}]:")
                    for arm_name, joint_data in joints.items():
                        if isinstance(joint_data, (list, np.ndarray)):
                            joint_array = np.array(joint_data)
                            print(f"   🤖 {arm_name}: {len(joint_array)}个关节 "
                                  f"范围[{joint_array.min():.3f}-{joint_array.max():.3f}]")

            # 执行OpenPI推理
            start_time = time.time()
            if self.inference_engine is None:
                print("❌ OpenPI推理引擎未初始化")
                return None

            result = self.inference_engine.infer(obs)
            inference_time = (time.time() - start_time) * 1000

            print(f"🤖 OpenPI推理完成: {inference_time:.1f}ms")

            if result is None:
                print("❌ OpenPI推理返回None")
                return None

            # 处理推理结果
            if isinstance(result, np.ndarray):
                print(f"📊 推理结果: {result.shape}")
                return result
            elif isinstance(result, dict) and 'actions' in result:
                actions = result['actions']
                print(f"📊 推理动作: {actions.shape}")
                return actions
            else:
                print(f"⚠️  未知推理结果格式: {type(result)}")
                return result

        except Exception as e:
            print(f"❌ OpenPI推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class XROCSControlClient:
    """xROCS控制客户端 - 通过控制接口与xROCS服务通信"""

    def __init__(self):
        self.command_count = 0
        self.left_arm_fixed = False

        # 初始化控制接口
        if CONTROL_INTERFACE_AVAILABLE:
            self.control_client = InferenceControlClient()
            print("🤖 xROCS控制客户端初始化")
            print("🏠 home位置从配置文件读取: agilex_eggplant_config.toml")
            print(f"�️  安全控制: {'启用' if self.enable_safety_limits else '禁用'}")
            if self.enable_safety_limits:
                print(f"   🚨 安全模式: {self.safety_mode}")
                print(f"   ⚡ 最大速度: {self.max_joint_velocity:.2f} rad/s")
                print(f"   📈 最大加速度: {self.max_joint_acceleration:.2f} rad/s²")
                print(f"   📏 单步限制: {self.max_single_step_change:.3f} rad ({np.degrees(self.max_single_step_change):.1f}°)")
                print(f"   🚨 紧急阈值: {self.emergency_stop_threshold:.2f} rad ({np.degrees(self.emergency_stop_threshold):.1f}°)")
            print(f"�🔄 平滑衔接: {'启用' if self.enable_smooth_transition else '禁用'}")
            if self.enable_smooth_transition:
                print(f"   ⏱️  衔接时间: {self.transition_time:.3f}s")
        else:
            print("❌ 控制接口不可用，无法初始化控制客户端")
            self.control_client = None


    def _set_motion_parameters(self):
        """设置运动参数（速度、加速度、抖动限制）"""
        if not self.control_client:
            return False

        try:
            # 设置速度限制
            self.control_client.set_motion_parameters(
                speed_limit=self.speed_limit,
                acceleration_limit=self.accel_limit,
                jerk_limit=self.jerk_limit
            )
            print(f"✅ 运动参数设置成功:")
            print(f"   - 速度限制: {self.speed_limit} rad/s")
            print(f"   - 加速度限制: {self.accel_limit} rad/s²")
            print(f"   - 抖动限制: {self.jerk_limit} rad/s³")
            return True
        except Exception as e:
            print(f"⚠️ 运动参数设置失败: {e}")
            print(f"⚠️ 将使用默认运动参数")
            return False

    def set_motion_parameters(self, speed_limit=None, accel_limit=None, jerk_limit=None):
        """动态调整运动参数

        Args:
            speed_limit: 速度限制 (rad/s)
            accel_limit: 加速度限制 (rad/s²)
            jerk_limit: 抖动限制 (rad/s³)

        Returns:
            是否成功设置
        """
        if not self.control_client:
            print("❌ 控制客户端不可用，无法设置运动参数")
            return False

        # 更新参数值（如果提供）
        if speed_limit is not None:
            self.speed_limit = speed_limit
        if accel_limit is not None:
            self.accel_limit = accel_limit
        if jerk_limit is not None:
            self.jerk_limit = jerk_limit

        try:
            # 发送运动参数设置命令
            command_id = self.control_client.set_motion_parameters(
                speed_limit=self.speed_limit,
                acceleration_limit=self.accel_limit,
                jerk_limit=self.jerk_limit
            )

            print(f"✅ 运动参数设置命令已发送 (ID: {command_id}):")
            print(f"   - 速度限制: {self.speed_limit} rad/s")
            print(f"   - 加速度限制: {self.accel_limit} rad/s²")
            print(f"   - 抖动限制: {self.jerk_limit} rad/s³")
            return True
        except Exception as e:
            print(f"❌ 发送运动参数设置命令失败: {e}")
            return False



    def send_joint_command(self, left_joints=None, right_joints=None):
        """发送关节控制命令 - 简单直接版本"""
        self.command_count += 1

        if not self.control_client:
            print(f"❌ 命令{self.command_count}: 控制客户端不可用")
            return False

        try:
            total_commands = 0
            sent_commands = []

            # 发送左臂命令（如果左臂未固定）
            if left_joints is not None and not self.left_arm_fixed:
                total_commands += 1
                print(f"📤 发送关节命令 {self.command_count}: left {left_joints[:3]}...")

                # 发送命令
                command_id = self.control_client.send_joint_command(
                    left_joints,
                    arm="left"
                )

                sent_commands.append(f"L:{command_id}")
                print(f"📤 命令{self.command_count}L: 左臂命令已发送 (ID: {command_id})")

            # 发送右臂命令
            if right_joints is not None:
                total_commands += 1
                print(f"📤 发送关节命令 {self.command_count}: right {right_joints[:3]}...")

                # 发送命令
                command_id = self.control_client.send_joint_command(
                    right_joints,
                    arm="right"
                )

                sent_commands.append(f"R:{command_id}")
                print(f"📤 命令{self.command_count}R: 右臂命令已发送 (ID: {command_id})")

            # 总结发送结果
            if total_commands > 0:
                commands_str = ", ".join(sent_commands)
                if self.left_arm_fixed:
                    print(f"🚀 命令{self.command_count}: 只发送右臂命令 [{commands_str}] - 左臂保持固定")
                else:
                    print(f"🚀 命令{self.command_count}: {total_commands}臂命令已发送 [{commands_str}]")

            return True  # 总是返回True继续推理

        except Exception as e:
            print(f"❌ 命令{self.command_count}: 发送失败 - {e}")
            return False
    


    # 移除插值轨迹方法 - 直接使用xROCS内置轨迹规划

    def go_to_home(self):
        """让双臂机械臂回到home位置 - 发送命令后等待足够时间"""
        print("🏠 双臂机械臂回到home位置...")

        if not self.control_client:
            print("❌ 控制客户端不可用，无法回到home位置")
            return False

        try:
            # 发送左臂home命令
            command_id_left = self.control_client.send_home_command(arm="left")
            print(f"📤 左臂home命令已发送 (ID: {command_id_left})")

            time.sleep(0.5)  # 短暂延迟避免命令冲突
            # import pdb;pdb.set_trace()

            # 发送右臂home命令
            command_id_right = self.control_client.send_home_command(arm="right")
            print(f"📤 右臂home命令已发送 (ID: {command_id_right})")

            print(f"🚀 Home命令: 双臂home命令已发送 [L:{command_id_left}, R:{command_id_right}]")

            # 等待机械臂移动到home位置 - 给足够的时间
            print("⏳ 等待机械臂移动到home位置...")
            time.sleep(3.0)  # 等待3秒让机械臂移动到home位置
            
            # 设置左臂固定标志
            self.left_arm_fixed = True
            print("✅ 机械臂已到达home位置，左臂将保持固定，只有右臂会动")
            return True

        except Exception as e:
            print(f"❌ 回到home位置失败: {e}")
            return True  # 返回True继续推理


def main():
    """主推理循环"""
    print("🚀 启动OpenPI推理 (Python 3.11)")
    print("=" * 60)
    
    # 检查套接字服务
    socket_path = "/tmp/xrocs_memory_service.sock"
    if not Path(socket_path).exists():
        print("❌ 内存套接字服务未运行，请先启动:")
        print("   python simple_memory_service.py")
        print("   确保socket_server=True")
        return

    print("✅ 内存套接字服务检查通过")

    # 初始化组件
    print("\n🔧 初始化组件...")

    # 数据提供器 - 使用套接字通信
    data_provider = MemoryDataProvider()
    print("✅ 使用套接字数据提供器 (高性能进程间通信)")

    # OpenPI推理引擎
    if not OPENPI_AVAILABLE:
        print("❌ OpenPI推理引擎不可用")
        return

    try:
        inference_engine = OpenPIInferenceEngine()
    except Exception as e:
        print(f"❌ OpenPI推理引擎初始化失败: {e}")
        return

    # 命令发送器
    command_sender = XROCSControlClient()

    # 设置更适合实时控制的运动参数
    # 降低加速度限制可以使运动更平滑，但响应会变慢
    command_sender.set_motion_parameters(
        speed_limit=1.2,      # 适中的速度限制
        accel_limit=0.8,      # 较低的加速度限制，提高平滑性
        jerk_limit=3.0        # 较低的抖动限制，减少振动
    )

    print("\n⚙️ 运动参数配置:")
    print(f"   - 速度限制: 1.2 rad/s")
    print(f"   - 加速度限制: 0.8 rad/s²")
    print(f"   - 抖动限制: 3.0 rad/s³")
    
    # 在开始推理之前，先让机械臂回到home位置
    print("\n🏠 推理前准备：机械臂回到home位置...")
    try:
        if command_sender.go_to_home():
            print("✅ 机械臂已回到home位置，准备开始异步推理")
        else:
            print("⚠️  机械臂回home失败，但继续推理")
    except Exception as e:
        print(f"⚠️  机械臂回home过程中出错: {e}")
        print("⚠️  继续推理，但可能影响初始状态")

    # 初始化异步控制器
    print("\n🔧 初始化异步推理控制器...")

    # 时间配置 - 极度激进方案：彻底消除犹豫
    action_dt = 0.03  # 每个action之间30ms间隔
    target_time_offset = 0.9  # 预测未来500ms的动作 (激进预测)
    filter_alpha = 0.9 # 无滤波 (100%使用推理结果)
    history_size = 10   # 最少历史数据，最大响应性
    debug_delay = 0.0 # 秒，设置为0.0表示无延迟（真正异步）

    async_controller = AsyncInferenceController(
        data_provider=data_provider,
        inference_engine=inference_engine,
        command_sender=command_sender,
        history_size=history_size,  # 使用历史10次数据进行时间聚合
        target_time_offset=target_time_offset,  # 预测未来1000ms的动作
        action_dt=action_dt,  # 30ms间隔
        filter_alpha=filter_alpha,  # 动作滤波系数
        debug_delay=debug_delay  # 调试延迟
    )

    execution_protection_time = target_time_offset * 0.5 # 执行保护时间为预测时间的40% (激进)

    print(f"⚡ 异步控制器配置 (简化直接版):")
    print(f"   🔄 时间聚合: 每次推理使用历史{history_size}次数据")
    print(f"   ⏰ 动作间隔: {action_dt*1000:.0f}ms")
    print(f"   🎯 预测时间: {target_time_offset*1000:.0f}ms ({target_time_offset/action_dt:.1f}个动作步骤)")
    print(f"   🔧 滤波策略: 完全无滤波 (直接使用推理结果)")
    print(f"   ⚡ 执行保护: {execution_protection_time*1000:.0f}ms (保护期间保持当前命令)")
    print(f"   🚀 推理频率: ~11Hz (保持高频推理)")
    print(f"   ✅ 核心策略: 直接执行推理结果，无平滑衔接")
    print(f"   📊 工作模式:")
    print(f"     - 前{execution_protection_time*1000:.0f}ms: 保持当前命令不变")
    print(f"     - 后续时间: 直接执行新推理结果")
    print(f"   ⚡ 预期效果: 每{execution_protection_time*1000:.0f}ms发送新命令，直接响应")
    if debug_delay > 0:
        print(f"   ⏱️  调试延迟: {debug_delay}s ({1/debug_delay:.1f}Hz)")
    else:
        print(f"   ⚡ 执行模式: 无延迟(真正异步)")

    print("\n🎯 启动异步OpenPI推理系统...")
    print("按 Ctrl+C 停止")

    # 启动异步控制器
    async_controller.start()

    try:
        # 主线程等待用户中断
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  用户停止")
    except Exception as e:
        print(f"\n❌ 异步推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止异步控制器
        async_controller.stop()

        # 显示最终统计
        stats = async_controller.get_stats()
        print(f"\n📊 最终统计:")
        print(f"   推理次数: {stats['inference_count']}")
        print(f"   命令次数: {stats['command_count']}")
        print(f"   运行时间: {stats['total_time']:.1f}s")
        print(f"   推理频率: {stats['inference_freq']:.1f}Hz")
        print(f"   命令频率: {stats['command_freq']:.1f}Hz")
        print("🎯 异步OpenPI推理结束")


if __name__ == '__main__':
    main()
