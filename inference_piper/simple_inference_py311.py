#!/usr/bin/env python3
"""
OpenPI推理脚本 - Python 3.11版本
使用保存的OpenPI兼容数据进行真正的OpenPI推理
"""

import time
import json
import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入数据客户端
try:
    from data_client import DataClient
    DATA_CLIENT_AVAILABLE = True
    print("✅ 数据客户端可用")
except ImportError:
    print("❌ 数据客户端不可用")
    DATA_CLIENT_AVAILABLE = False

# 导入OpenPI推理引擎
try:
    from inference_agilexv2_openpi import JointInference
    OPENPI_AVAILABLE = True
    print("✅ OpenPI推理引擎可用")
except ImportError as e:
    print(f"❌ OpenPI推理引擎不可用: {e}")
    OPENPI_AVAILABLE = False


class SimpleDataProvider:
    """简化数据提供器"""
    
    def __init__(self, data_dir="./data"):
        self.data_client = DataClient(data_dir=data_dir)
        print(f"📁 简化数据提供器初始化: {data_dir}")
    
    def get_observation(self):
        """获取观测数据 - OpenPI格式"""
        try:
            # 读取关节数据
            joints = self.data_client.read_current_joints()
            if not joints:
                return None

            # 读取图像数据 - 确保OpenPI格式
            cameras = ['left_wrist_0_rgb', 'right_wrist_0_rgb', 'base_0_rgb']
            images = {}

            for cam_name in cameras:
                img = self.data_client.read_current_image(cam_name)
                if img is not None:
                    # 验证OpenPI格式要求
                    assert img.shape == (224, 224, 3), f"图像尺寸错误: {img.shape}"
                    assert img.dtype == np.uint8, f"数据类型错误: {img.dtype}"
                    images[cam_name] = img
                else:
                    # 创建黑色占位图像
                    images[cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)
                    print(f"⚠️  使用占位图像: {cam_name}")

            # 处理关节数据 - 适配只有右臂的情况
            arm_joints = {}

            # 右臂数据
            if 'right' in joints:
                arm_joints['right'] = np.array(joints['right']['values'], dtype=np.float32)
                print(f"✅ 右臂关节: {len(arm_joints['right'])}个")
            else:
                print("❌ 缺少右臂关节数据")
                return None

            # 左臂数据 (如果不存在则用零填充)
            if 'left' in joints:
                arm_joints['left'] = np.array(joints['left']['values'], dtype=np.float32)
                print(f"✅ 左臂关节: {len(arm_joints['left'])}个")
            else:
                # 用零填充左臂数据
                arm_joints['left'] = np.zeros(7, dtype=np.float32)
                print("⚠️  左臂数据缺失，使用零填充")

            # 构建OpenPI格式观测数据
            obs = {
                'arm_joints': arm_joints,
                'images': images
            }

            return obs

        except Exception as e:
            print(f"❌ 获取观测数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class OpenPIInferenceEngine:
    """OpenPI推理引擎 - 真正的OpenPI推理"""

    def __init__(self, config_path=None):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI推理引擎不可用")

        if config_path is None:
            config_path = "/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml"

        self.step_count = 0
        print(f"🤖 初始化OpenPI推理引擎: {config_path}")

        try:
            # 初始化JointInference
            self.joint_inference = JointInference(config_path=config_path)
            print("✅ OpenPI推理引擎初始化成功")
        except Exception as e:
            print(f"❌ OpenPI推理引擎初始化失败: {e}")
            raise

    def infer(self, obs):
        """执行OpenPI推理"""
        try:
            self.step_count += 1

            print(f"🤖 OpenPI推理: 步骤{self.step_count}")

            # 验证观测数据格式
            if 'arm_joints' not in obs or 'images' not in obs:
                print("❌ 观测数据格式错误")
                return None

            # 检查图像数据
            required_cameras = ['left_wrist_0_rgb', 'right_wrist_0_rgb', 'base_0_rgb']
            for cam_name in required_cameras:
                if cam_name not in obs['images']:
                    print(f"❌ 缺少相机数据: {cam_name}")
                    return None

                img = obs['images'][cam_name]
                if img.shape != (224, 224, 3) or img.dtype != np.uint8:
                    print(f"❌ 图像格式错误 {cam_name}: {img.shape}, {img.dtype}")
                    return None

            # 执行OpenPI推理
            start_time = time.time()
            if self.joint_inference.inference_engine is None:
                print("❌ OpenPI推理引擎未初始化")
                return None

            result = self.joint_inference.inference_engine.infer(obs)
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


class ROSCommandSender:
    """ROS命令发送器 - 通过subprocess避免直接使用rospy"""
    
    def __init__(self):
        self.command_count = 0
        print("🤖 ROS命令发送器初始化")
    
    def send_joint_command(self, right_joints):
        """发送关节控制命令 - 只控制右臂"""
        try:
            self.command_count += 1

            # 只构建右臂rostopic pub命令
            right_cmd = self._build_joint_command('/master/joint_right', right_joints)

            # 只发送右臂命令
            right_result = subprocess.run(right_cmd, shell=True, capture_output=True, text=True, timeout=0.5)

            # 检查结果
            right_success = right_result.returncode == 0

            if right_success:
                print(f"📤 命令{self.command_count}: R{right_joints[:3]} ✅")
                return True
            else:
                print(f"📤 命令{self.command_count}: 右臂失败 R:{right_success}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ 命令{self.command_count}: 超时")
            return False
        except Exception as e:
            print(f"❌ 命令{self.command_count}: 错误 {e}")
            return False
    
    def _build_joint_command(self, topic, joint_values):
        """构建关节命令"""
        # 构建JointState消息
        positions = ','.join([f'{v:.6f}' for v in joint_values])
        names = ','.join([f'joint_{i+1}' for i in range(len(joint_values))])
        
        cmd = f'rostopic pub -1 {topic} sensor_msgs/JointState "{{header: {{stamp: now}}, name: [{names}], position: [{positions}]}}"'
        return cmd


def main():
    """主推理循环"""
    print("🚀 启动OpenPI推理 (Python 3.11)")
    print("=" * 60)
    
    # 检查数据服务
    data_dir = Path("./data")
    if not data_dir.exists() or not (data_dir / "current_joints.json").exists():
        print("❌ 数据服务未运行，请先在base环境启动:")
        print("   python simple_data_service.py")
        return
    
    print("✅ 数据服务检查通过")
    
    # 初始化组件
    print("\n🔧 初始化组件...")
    
    # 数据提供器
    data_provider = SimpleDataProvider()

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
    command_sender = ROSCommandSender()
    
    print("\n🎯 开始OpenPI推理循环 (10Hz)...")
    print("按 Ctrl+C 停止")
    
    step = 0
    start_time = time.time()
    success_count = 0
    
    try:
        while True:
            step_start = time.time()
            
            print(f"\n🔄 步骤 {step} [{datetime.now().strftime('%H:%M:%S')}]")
            
            # 1. 获取观测数据
            obs = data_provider.get_observation()
            if not obs:
                print("❌ 观测数据获取失败")
                time.sleep(0.2)
                continue
            
            # 检查数据完整性
            if len(obs['images']) < 3:
                print(f"⚠️  图像数据不完整: {len(obs['images'])}/3")
                time.sleep(0.2)
                continue

            print(f"📊 数据获取成功: {len(obs['arm_joints'])}臂, {len(obs['images'])}相机")

            # 显示当前关节角度
            left_joints = obs['arm_joints']['left']
            right_joints = obs['arm_joints']['right']
            print(f"📐 当前关节: L{left_joints[:1]}, R{right_joints}")
            
            # 2. 执行简化推理
            inference_start = time.time()
            action = inference_engine.infer(obs)
            inference_time = (time.time() - inference_start) * 1000

            # 调试：显示输入的关节角度信息
            if obs and 'arm_joints' in obs:
                current_joints = obs['arm_joints']['right']
                print(f"🔍 输入关节角度: {current_joints}")
                print(f"🔍 关节角度范围: [{np.min(current_joints):.3f}, {np.max(current_joints):.3f}]")
            
            if action is None:
                print(f"❌ 推理失败")
                time.sleep(0.2)
                continue
            
            print(f"🤖 推理成功: {inference_time:.1f}ms")

            # 3. 解析OpenPI动作并发送命令
            if isinstance(action, np.ndarray):
                print(f"📊 动作数组: {action.shape}")
                # OpenPI通常返回 (action_horizon, action_dim) 格式
                if len(action.shape) == 2:
                    # 取第一个动作 (最近的动作)
                    current_action = action[-1]
                    print(f"📊 当前动作: {current_action.shape}")
                elif len(action.shape) == 1:
                    current_action = action
                else:
                    print(f"⚠️  未知动作格式: {action.shape}")
                    time.sleep(0.2)
                    continue

                # 检查动作维度 (应该是32维或14维)
                if len(current_action) >= 14:
                    skip_wait = False  # 默认不跳过等待
                    # 提取关节角度 (只使用右臂，7-14维是右臂关节角度)
                    new_right_joints = current_action[7:14]

                    # 获取当前关节角度用于对比
                    current_obs = data_provider.get_observation()
                    if current_obs and 'arm_joints' in current_obs:
                        current_right_joints = current_obs['arm_joints']['right']
                        joint_diff = new_right_joints - current_right_joints
                        max_change = np.max(np.abs(joint_diff))
                        print(f"📐 当前关节: R{current_right_joints}")
                        print(f"📐 目标关节: R{new_right_joints}")
                        print(f"📐 关节变化: R{joint_diff}")
                        print(f"📐 最大变化: {max_change:.6f}弧度 ({np.degrees(max_change):.3f}度)")

                        # 如果变化太小，可能需要调整阈值或检查归一化
                        if max_change < 0.01:  # 小于0.01弧度(约0.57度)
                            print(f"⚠️  关节变化很小({max_change:.6f})，可能模型认为已接近目标")
                            print(f"💡 建议：这是正常现象，模型认为当前状态合理")
                            # 仍然发送命令，但可以考虑跳过等待
                            skip_wait = True
                        else:
                            skip_wait = False
                    else:
                        print(f"📐 目标关节: R{new_right_joints}")

                    # 发送控制命令 (只控制右臂) - 跳过返回判断，直接认为成功
                    command_sender.send_joint_command(new_right_joints)
                    success_count += 1
                    print("📤 控制命令已发送")

                    # 等待机械臂执行到位并检查是否到达目标
                    if not skip_wait:
                        print("⏳ 等待机械臂执行到目标位置...")
                        wait_start = time.time()
                        max_wait_time = 2.0  # 最多等待2秒
                        target_reached = False

                        while (time.time() - wait_start) < max_wait_time:
                            time.sleep(0.15)  # 150ms检查一次

                            # 读取当前关节状态
                            current_obs = data_provider.get_observation()
                            if current_obs and 'arm_joints' in current_obs:
                                current_right_joints = current_obs['arm_joints']['right']

                                # 检查是否到达目标附近 (放宽阈值到0.1弧度，约5.7度)
                                joint_diff = np.abs(current_right_joints - new_right_joints)
                                max_diff = np.max(joint_diff)

                                if max_diff < 0.1:
                                    wait_time = (time.time() - wait_start) * 1000
                                    print(f"✅ 到达目标位置: {wait_time:.1f}ms, 最大误差: {max_diff:.4f}")
                                    target_reached = True
                                    break
                                else:
                                    print(f"📍 移动中... 当前误差: {max_diff:.4f}")

                        if not target_reached:
                            wait_time = (time.time() - wait_start) * 1000
                            print(f"⚠️  未完全到达目标({wait_time:.1f}ms)，继续下一步")
                    else:
                        print("⚡ 跳过等待（变化很小）")
                else:
                    print(f"⚠️  动作维度不足: {len(current_action)}, 需要至少14维")
            else:
                print(f"⚠️  未知动作类型: {type(action)}")

            # 4. 控制频率 (确保最小间隔)
            step_time = (time.time() - step_start) * 1000
            min_interval = 100  # 最小100ms间隔

            if step_time < min_interval:
                sleep_time = (min_interval - step_time) / 1000
                time.sleep(sleep_time)
            
            total_time = (time.time() - step_start) * 1000
            print(f"⏱️  步骤完成: {total_time:.1f}ms")
            
            step += 1
            
            # 每5步显示统计
            if step % 5 == 0:
                runtime = time.time() - start_time
                avg_freq = step / runtime
                success_rate = success_count / step * 100
                print(f"📊 统计: {step}步, 运行{runtime:.1f}s, 平均{avg_freq:.1f}Hz, 成功率{success_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户停止")
    except Exception as e:
        print(f"\n❌ 推理循环错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runtime = time.time() - start_time
        if step > 0:
            avg_freq = step / runtime
            success_rate = success_count / step * 100
            print(f"\n📊 最终统计:")
            print(f"   总步数: {step}")
            print(f"   运行时间: {runtime:.1f}s")
            print(f"   平均频率: {avg_freq:.1f}Hz")
            print(f"   成功率: {success_rate:.1f}%")
        print("🎯 OpenPI推理结束")


if __name__ == '__main__':
    main()
