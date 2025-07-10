#!/usr/bin/env python3
"""
xROCS控制服务 - 使用Python 3.8和xROCS API进行机械臂控制
需要在base环境(Python 3.8)中运行
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_file))
# 将xRocs目录添加到Python路径
xrocs_dir = os.path.join(project_root, 'xRocs')
sys.path.insert(0, xrocs_dir)  # 使用insert(0,...)确保它是第一个被搜索的路径

try:
    # 从xrocs模块导入
    from xrocs.core.station_loader import StationLoader
    from xrocs.common.data_type import Joints
    from xrocs.core.config_loader import ConfigLoader
    print("✅ xROCS模块导入成功")
except ImportError as e:
    print(f"❌ xROCS模块导入失败: {e}")
    print("请确保在base环境(Python 3.8)中运行此脚本")
    print(f"Python路径: {sys.path}")
    print(f"xRocs目录: {xrocs_dir}")
    if os.path.exists(xrocs_dir):
        print(f"xRocs目录内容: {os.listdir(xrocs_dir)}")
        xrocs_subdir = os.path.join(xrocs_dir, 'xrocs')
        if os.path.exists(xrocs_subdir):
            print(f"xrocs子目录内容: {os.listdir(xrocs_subdir)}")
    sys.exit(1)

from control_interface import ControlServiceServer


class XROCSControlService:
    """xROCS控制服务 - 带安全限制"""

    def __init__(self, config_path: str = "/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml"):
        self.config_path = config_path
        self.robot_station = None
        self.robot_handles = {}
        self.cfg_dict = None

        # 安全控制配置 - 从配置文件加载
        self.load_safety_config()

    def load_safety_config(self):
        """加载安全配置"""
        import json
        config_file = Path("safety_config.json")

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.enable_safety_limits = config.get("enable_safety_limits", True)
                self.max_single_step_change = config.get("max_single_step_change", 0.03)
                self.emergency_stop_threshold = config.get("emergency_stop_threshold", 0.3)
                self.max_velocity_estimate = config.get("max_velocity_estimate", 0.2)
                print(f"✅ 从 {config_file} 加载安全配置")
            except Exception as e:
                print(f"⚠️  加载安全配置失败: {e}，使用默认值")
                self.set_default_safety_config()
        else:
            print("📁 安全配置文件不存在，使用默认值")
            self.set_default_safety_config()

    def set_default_safety_config(self):
        """设置默认安全配置"""
        self.enable_safety_limits = True
        self.max_single_step_change = 0.03  # 单步最大变化约1.7度，非常保守
        self.emergency_stop_threshold = 0.3  # 紧急停止阈值约17度
        self.max_velocity_estimate = 0.2     # 最大速度估计 (rad/s)

        # 当前状态跟踪
        self.last_joint_positions = {}  # 上次关节位置
        self.last_command_time = {}     # 上次命令时间
        self.blocked_commands = 0       # 被阻止的命令数
        self.limited_commands = 0       # 被限制的命令数

        # 定义默认 home 位置（作为备用）
        self.home_positions = {
            "right": [-0.038238353639841, -0.0, 0.0, 0.040487524122, 0, -0.096918866038322, 0.02],
            "left": [0.038238353639841, 0.0, 0.0, -0.040487524122, 0, 0.096918866038322, -0.02]
        }
        
        print(f"🤖 xROCS控制服务初始化")
        print(f"📁 配置文件: {self.config_path}")
        print(f"🛡️  安全控制: {'启用' if self.enable_safety_limits else '禁用'}")
        if self.enable_safety_limits:
            print(f"   📏 单步限制: {self.max_single_step_change:.3f}rad ({np.degrees(self.max_single_step_change):.1f}°)")
            print(f"   🚨 紧急阈值: {self.emergency_stop_threshold:.3f}rad ({np.degrees(self.emergency_stop_threshold):.1f}°)")
            print(f"   ⚡ 最大速度: {self.max_velocity_estimate:.2f}rad/s")

        # 从配置文件加载 home 位置
        try:
            # 加载配置文件
            try:
                import tomli as tomllib
                with open(self.config_path, 'rb') as f:
                    config = tomllib.load(f)
                    self.cfg_dict = config
            except ImportError:
                # 如果 tomli 不可用，尝试使用 toml
                import toml
                with open(self.config_path, 'r') as f:
                    config = toml.load(f)
                    self.cfg_dict = config
            
            print(f"✅ 配置加载成功")
            
            # 从配置文件中读取 home 位置
            if 'robot' in config and 'arm' in config['robot'] and 'home' in config['robot']['arm']:
                if 'left' in config['robot']['arm']['home']:
                    self.home_positions['left'] = config['robot']['arm']['home']['left']
                    print(f"📍 从配置文件读取左臂 home 位置: {self.home_positions['left']}")
                
                if 'right' in config['robot']['arm']['home']:
                    self.home_positions['right'] = config['robot']['arm']['home']['right']
                    print(f"📍 从配置文件读取右臂 home 位置: {self.home_positions['right']}")
            else:
                print(f"⚠️ 配置文件中未找到 home 位置，使用默认值")
        
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}，使用默认 home 位置")
    
    def initialize_xrocs(self):
        """初始化xROCS系统"""
        try:
            print("🔧 初始化xROCS...")
            
            # 加载配置
            cfg_loader = ConfigLoader(self.config_path)
            self.cfg_dict = cfg_loader.get_config()
            print("✅ 配置加载成功")
            
            # 创建机器人站
            station_loader = StationLoader(self.cfg_dict)
            self.robot_station = station_loader.generate_station_handle()
            print("✅ 机器人站创建成功")
            
            # 连接机器人
            self.robot_station.connect()
            print("✅ 机器人连接成功")
            
            # 获取机器人句柄
            self.robot_handles = self.robot_station.get_robot_handle()
            print(f"✅ 获取机器人句柄: {list(self.robot_handles.keys())}")
            
            # 跳过机器人启用（Agilex机器人在ROS launch中已处理启用）
            print("ℹ️  跳过机器人启用 - Agilex机器人在ROS launch中已处理启用")
            
            return True
            
        except Exception as e:
            print(f"❌ xROCS初始化失败: {e}")
            return False

    def check_safety_limits(self, arm: str, target_joints: list) -> tuple:
        """
        检查安全限制

        返回:
        - (is_safe, limited_joints, reason)
        """
        target_joints = np.array(target_joints)

        # 获取当前关节位置
        current_joints = self.get_current_joints(arm)
        if not current_joints:
            print(f"⚠️  无法获取{arm}臂当前位置，允许执行但记录警告")
            return True, target_joints.tolist(), "no_current_position"

        current_joints = np.array(current_joints)

        # 检查关节数量匹配
        if len(target_joints) != len(current_joints):
            print(f"❌ 关节数量不匹配: 目标{len(target_joints)} vs 当前{len(current_joints)}")
            return False, target_joints.tolist(), "joint_count_mismatch"

        # 计算关节变化
        joint_diff = target_joints - current_joints
        max_change = np.max(np.abs(joint_diff))

        # 紧急停止检查
        if max_change > self.emergency_stop_threshold:
            print(f"🚨 紧急停止！{arm}臂最大关节变化 {max_change:.3f}rad ({np.degrees(max_change):.1f}°) 超过阈值 {self.emergency_stop_threshold:.3f}rad")
            print(f"   当前: {current_joints}...")
            print(f"   目标: {target_joints}...")
            print(f"   变化: {joint_diff}...")
            self.blocked_commands += 1
            return False, current_joints.tolist(), "emergency_stop"

        # 单步变化限制
        if max_change > self.max_single_step_change:
            # 限制每个关节的变化量
            limited_diff = np.clip(joint_diff, -self.max_single_step_change, self.max_single_step_change)
            limited_joints = current_joints + limited_diff

            print(f"⚠️  {arm}臂关节变化过大，已限制:")
            print(f"   原始变化: {max_change:.3f}rad ({np.degrees(max_change):.1f}°)")
            print(f"   限制变化: {np.max(np.abs(limited_diff)):.3f}rad ({np.degrees(np.max(np.abs(limited_diff))):.1f}°)")
            print(f"   目标: {target_joints[:3]}... -> 限制: {limited_joints[:3]}...")

            self.limited_commands += 1
            return True, limited_joints.tolist(), "step_limited"

        # 速度估计检查
        current_time = time.time()
        if arm in self.last_command_time:
            dt = current_time - self.last_command_time[arm]
            if dt > 0:
                estimated_velocity = max_change / dt
                if estimated_velocity > self.max_velocity_estimate:
                    # 基于速度限制重新计算目标
                    max_allowed_change = self.max_velocity_estimate * dt
                    scale_factor = max_allowed_change / max_change
                    limited_diff = joint_diff * scale_factor
                    limited_joints = current_joints + limited_diff

                    print(f"⚠️  {arm}臂速度过快，已限制:")
                    print(f"   估计速度: {estimated_velocity:.3f}rad/s")
                    print(f"   最大速度: {self.max_velocity_estimate:.3f}rad/s")
                    print(f"   缩放因子: {scale_factor:.3f}")

                    self.limited_commands += 1
                    return True, limited_joints.tolist(), "velocity_limited"

        # 更新时间记录
        self.last_command_time[arm] = current_time

        return True, target_joints.tolist(), "safe"

    def execute_joint_command(self, arm: str, joint_angles: list) -> bool:
        """
        执行关节控制命令 - 带安全检查

        参数:
        - arm: 机械臂名称 ("left" 或 "right")
        - joint_angles: 关节角度列表

        返回:
        - 是否成功执行
        """
        try:
            if arm not in self.robot_handles:
                print(f"❌ 未找到机械臂: {arm}")
                return False

            # 安全检查
            if self.enable_safety_limits:
                is_safe, safe_joints, reason = self.check_safety_limits(arm, joint_angles)

                if not is_safe:
                    print(f"🚨 {arm}臂命令被阻止: {reason}")
                    return False

                if reason in ["step_limited", "velocity_limited"]:
                    print(f"⚠️  {arm}臂命令已限制: {reason}")
                    joint_angles = safe_joints
                elif reason == "safe":
                    print(f"✅ {arm}臂命令安全检查通过")

            robot_handle = self.robot_handles[arm]

            # 创建关节对象
            joints = Joints(joint_angles, num_of_dofs=len(joint_angles))

            print(f"🎯 执行{arm}臂关节命令: {joint_angles[:3]}...")

            # 使用xROCS的顺滑控制
            success = robot_handle.reach_target_joint(joints)

            if success:
                print(f"✅ {arm}臂关节命令执行成功")
                # 更新位置记录
                self.last_joint_positions[arm] = joint_angles
            else:
                print(f"❌ {arm}臂关节命令执行失败")

            return success

        except Exception as e:
            print(f"❌ 关节命令执行异常: {e}")
            return False
    
    def execute_home_command(self, arm: str) -> bool:
        """
        执行回home命令 - 分步安全执行

        参数:
        - arm: 机械臂名称

        返回:
        - 是否成功执行
        """
        try:
            if arm not in self.home_positions:
                print(f"❌ 未定义{arm}臂的home位置")
                return False

            home_joints = self.home_positions[arm]
            print(f"🏠 {arm}臂安全回到home位置: {home_joints[:3]}...")

            # 获取当前位置
            current_joints = self.get_current_joints(arm)
            if not current_joints:
                print(f"⚠️  无法获取{arm}臂当前位置，直接执行home命令")
                return self.execute_joint_command(arm, home_joints)

            current_joints = np.array(current_joints)
            home_joints = np.array(home_joints)

            # 计算到home的距离
            total_diff = home_joints - current_joints
            max_diff = np.max(np.abs(total_diff))

            print(f"📏 {arm}臂到home距离: {max_diff:.3f}rad ({np.degrees(max_diff):.1f}°)")

            # 如果距离很小，直接执行
            if max_diff <= self.max_single_step_change:
                print(f"✅ {arm}臂距离home很近，直接执行")
                return self.execute_joint_command(arm, home_joints.tolist())

            # 如果距离很大，分步执行
            print(f"⚠️  {arm}臂距离home较远，分步安全执行...")

            # 计算需要的步数
            num_steps = int(np.ceil(max_diff / self.max_single_step_change))
            print(f"📊 计划分{num_steps}步执行home命令")

            # 分步执行
            for step in range(num_steps):
                # 计算当前步的目标
                progress = (step + 1) / num_steps
                step_target = current_joints + total_diff * progress

                print(f"🔄 执行第{step+1}/{num_steps}步 (进度{progress*100:.0f}%)")

                # 执行这一步
                success = self.execute_joint_command(arm, step_target.tolist())
                if not success:
                    print(f"❌ {arm}臂home命令第{step+1}步失败")
                    return False

                # 短暂等待让机械臂移动
                time.sleep(0.5)

                # 更新当前位置
                new_current = self.get_current_joints(arm)
                if new_current:
                    current_joints = np.array(new_current)

            print(f"✅ {arm}臂分步home命令执行完成")
            return True

        except Exception as e:
            print(f"❌ home命令执行异常: {e}")
            return False
    
    def get_current_joints(self, arm: str) -> list:
        """获取当前关节角度"""
        try:
            if arm not in self.robot_handles:
                return []

            robot_handle = self.robot_handles[arm]
            current_joints = robot_handle.get_current_joint()

            if current_joints:
                # 使用正确的方法获取关节角度值
                return current_joints.get_radian_ndarray().tolist()
            else:
                return []

        except Exception as e:
            print(f"❌ 获取{arm}臂关节角度失败: {e}")
            return []

    def execute_motion_parameters_command(self, command: dict) -> bool:
        """
        执行运动参数设置命令

        参数:
        - command: 命令字典，包含运动参数

        返回:
        - 是否成功执行
        """
        try:
            speed_limit = command.get("speed_limit")
            acceleration_limit = command.get("acceleration_limit")
            jerk_limit = command.get("jerk_limit")

            print("⚙️ 执行运动参数设置命令...")

            # 注意：xROCS的Agilex机器人可能不支持动态运动参数设置
            # 这里我们只是记录参数，实际的运动限制由机器人底层控制
            success = True

            if speed_limit is not None:
                print(f"   - 速度限制: {speed_limit} rad/s (记录)")
            if acceleration_limit is not None:
                print(f"   - 加速度限制: {acceleration_limit} rad/s² (记录)")
            if jerk_limit is not None:
                print(f"   - 抖动限制: {jerk_limit} rad/s³ (记录)")

            if success:
                print("✅ 运动参数设置命令执行成功 (参数已记录)")
            else:
                print("❌ 运动参数设置命令执行失败")

            return success

        except Exception as e:
            print(f"❌ 运动参数设置命令执行异常: {e}")
            return False
    
    def command_handler(self, command: dict) -> bool:
        """
        处理控制命令

        参数:
        - command: 命令字典

        返回:
        - 是否成功处理
        """
        try:
            command_type = command.get("type")
            arm = command.get("arm", "right")

            if command_type == "joint_control":
                joint_angles = command.get("joint_angles", [])
                return self.execute_joint_command(arm, joint_angles)

            elif command_type == "home":
                return self.execute_home_command(arm)

            elif command_type == "set_motion_parameters":
                return self.execute_motion_parameters_command(command)

            else:
                print(f"❌ 未知命令类型: {command_type}")
                return False

        except Exception as e:
            print(f"❌ 命令处理异常: {e}")
            return False
    
    def run_service(self, interface_dir: str = "./control_interface"):
        """运行控制服务"""
        print("🚀 启动xROCS控制服务...")
        
        # 初始化xROCS
        if not self.initialize_xrocs():
            print("❌ xROCS初始化失败，服务无法启动")
            return
        
        # 创建控制服务器
        server = ControlServiceServer(interface_dir)
        server.update_status("ready", {"arms": list(self.robot_handles.keys())})
        
        try:
            # 开始监控命令
            server.start_monitoring(self.command_handler)

        except KeyboardInterrupt:
            print("⏹️  用户中断服务")

            # 显示安全统计
            if self.enable_safety_limits:
                print(f"\n📊 安全统计:")
                print(f"   🚨 阻止的危险命令: {self.blocked_commands}")
                print(f"   ⚠️  限制的命令: {self.limited_commands}")
                total_commands = self.blocked_commands + self.limited_commands
                if total_commands > 0:
                    print(f"   📈 安全干预率: {total_commands} 次安全干预")
        
        finally:
            # 清理资源
            print("🧹 清理资源...")
            if self.robot_station:
                try:
                    self.robot_station.disconnect()
                    print("✅ 机器人连接已断开")
                except:
                    pass
            
            server.update_status("stopped")
            print("🛑 xROCS控制服务已停止")


def main():
    """主函数"""
    print("🎛️  xROCS控制服务启动")
    print("📋 请确保:")
    print("   1. 在base环境(Python 3.8)中运行")
    print("   2. ROS已启动 (roscore)")
    print("   3. 相机节点已启动")
    print("   4. Piper机器人服务已启动")
    print()
    
    # 检查Python版本
    if sys.version_info.major != 3 or sys.version_info.minor != 8:
        print(f"⚠️  当前Python版本: {sys.version}")
        print("建议使用Python 3.8以确保xROCS兼容性")
    
    # 创建并运行服务
    service = XROCSControlService()
    service.run_service()


if __name__ == "__main__":
    main()
