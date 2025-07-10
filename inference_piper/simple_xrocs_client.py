#!/usr/bin/env python3
"""
简化的xROCS控制客户端 - 恢复到平滑衔接之前的版本
"""

import time
import numpy as np

# 导入控制接口
try:
    from control_interface import InferenceControlClient
    CONTROL_INTERFACE_AVAILABLE = True
    print("✅ 控制接口可用")
except ImportError:
    print("❌ 控制接口不可用")
    CONTROL_INTERFACE_AVAILABLE = False


class SimpleXROCSControlClient:
    """简化的xROCS控制客户端 - 直接发送指令，无平滑衔接"""

    def __init__(self):
        self.command_count = 0
        self.left_arm_fixed = False

        # 初始化控制接口
        if CONTROL_INTERFACE_AVAILABLE:
            self.control_client = InferenceControlClient()
            print("🤖 简化xROCS控制客户端初始化")
            print("🏠 home位置从配置文件读取: agilex_eggplant_config.toml")
        else:
            print("❌ 控制接口不可用，无法初始化控制客户端")
            self.control_client = None
    
    def set_motion_parameters(self, speed_limit=None, accel_limit=None, jerk_limit=None):
        """设置运动参数"""
        if not self.control_client:
            print("❌ 控制客户端不可用，无法设置运动参数")
            return False

        try:
            # 发送运动参数设置命令
            command_id = self.control_client.set_motion_parameters(
                speed_limit=speed_limit,
                acceleration_limit=accel_limit,
                jerk_limit=jerk_limit
            )

            print(f"✅ 运动参数设置命令已发送 (ID: {command_id}):")
            if speed_limit is not None:
                print(f"   - 速度限制: {speed_limit} rad/s")
            if accel_limit is not None:
                print(f"   - 加速度限制: {accel_limit} rad/s²")
            if jerk_limit is not None:
                print(f"   - 抖动限制: {jerk_limit} rad/s³")
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
    
    def go_to_home(self):
        """让双臂机械臂回到home位置"""
        print("🏠 双臂机械臂回到home位置...")

        if not self.control_client:
            print("❌ 控制客户端不可用，无法回到home位置")
            return False

        try:
            # 发送左臂home命令
            command_id_left = self.control_client.send_home_command(arm="left")
            print(f"📤 左臂home命令已发送 (ID: {command_id_left})")

            time.sleep(0.5)  # 短暂延迟避免命令冲突

            # 发送右臂home命令
            command_id_right = self.control_client.send_home_command(arm="right")
            print(f"📤 右臂home命令已发送 (ID: {command_id_right})")

            print(f"🚀 Home命令: 双臂home命令已发送 [L:{command_id_left}, R:{command_id_right}]")

            # 等待机械臂移动到home位置
            print("⏳ 等待机械臂移动到home位置...")
            time.sleep(3.0)  # 等待3秒让机械臂移动到home位置
            
            # 设置左臂固定标志
            self.left_arm_fixed = True
            print("✅ 机械臂已到达home位置，左臂将保持固定，只有右臂会动")
            return True

        except Exception as e:
            print(f"❌ 回到home位置失败: {e}")
            return True  # 返回True继续推理


# 测试函数
def test_simple_client():
    """测试简化客户端"""
    print("🧪 测试简化xROCS控制客户端")
    
    client = SimpleXROCSControlClient()
    
    # 测试运动参数设置
    client.set_motion_parameters(
        speed_limit=1.2,
        accel_limit=0.8,
        jerk_limit=3.0
    )
    
    # 测试home命令
    client.go_to_home()
    
    # 测试关节命令
    test_right_joints = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]
    client.send_joint_command(right_joints=test_right_joints)
    
    print("✅ 简化客户端测试完成")


if __name__ == "__main__":
    test_simple_client()
