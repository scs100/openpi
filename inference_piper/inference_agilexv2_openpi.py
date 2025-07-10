import time
import numpy as np
import sys
import os
import cv2
import threading
from collections import deque
from PIL import Image
import io
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取项目根目录（假设根目录是当前文件的父目录的父目录）
project_root = os.path.dirname(os.path.dirname(current_file))
# 将根目录添加到 Python 路径
sys.path.append(project_root)

from xRocs.xrocs.utils.logger.logger_loader import logger
from xRocs.xrocs.core.station_loader import StationLoader
from xRocs.xrocs.core.config_loader import ConfigLoader
from xRocs.xrocs.common.data_type import Joints



# 导入推理相关模块 - 替换为OpenPI推理
try:
    from openpi_client import websocket_client_policy
    OPENPI_AVAILABLE = True
    print("✅ OpenPI客户端模块可用")
except ImportError:
    print("❌ OpenPI客户端模块不可用，请确保已安装openpi_client")
    OPENPI_AVAILABLE = False

# 导入数据客户端
try:
    from data_client import DataClient
    DATA_CLIENT_AVAILABLE = True
    print("✅ 数据客户端模块可用")
except ImportError:
    print("❌ 数据客户端模块不可用")
    DATA_CLIENT_AVAILABLE = False

# 导入 ROS 相关模块（用于控制）
try:
    import rospy
    from sensor_msgs.msg import JointState, Image, CompressedImage
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("⚠️  ROS 模块不可用，将跳过直接 ROS 数据读取功能")
    ROS_AVAILABLE = False


class FileDataBuffer:
    """基于文件API的数据缓冲器 - 从数据服务读取文件"""

    def __init__(self, data_dir="./data", target_sizes=None):
        if not DATA_CLIENT_AVAILABLE:
            raise ImportError("数据客户端模块不可用")

        # 统一推理分辨率
        if target_sizes is None:
            target_sizes = {
                'left': (224, 224),    # 左相机：224x224
                'right': (224, 224),   # 右相机：224x224
                'front': (224, 224)    # 前相机：224x224
            }

        self.target_sizes = target_sizes
        self.data_dir = data_dir

        # 初始化数据客户端
        self.data_client = DataClient(data_dir=data_dir)

        # 相机名称映射
        self.camera_mapping = {
            'left': 'left_wrist_0_rgb',
            'right': 'right_wrist_0_rgb',
            'front': 'base_0_rgb'
        }

        # 最新数据缓存
        self.latest_joint_data = {}
        self.latest_image_data = {}
        self.data_lock = threading.RLock()

        # 性能监控
        self.frame_count = 0
        self.start_time = time.time()

        print("🚀 初始化文件数据缓冲器...")
        print(f"� 数据目录: {data_dir}")
        print(f"📏 目标图像尺寸: {self.target_sizes}")

        # 等待数据可用
        self._wait_for_data()

        print("✅ 文件数据缓冲器初始化完成")

    def _wait_for_data(self):
        """等待数据文件可用"""
        print("⏳ 等待数据文件...")
        if self.data_client.wait_for_data(timeout=10):
            print("✅ 数据文件已就绪")
        else:
            raise RuntimeError("❌ 数据文件不可用，请确保数据服务正在运行")

    def get_current_data(self):
        """获取当前数据"""
        try:
            start_time = time.time()

            # 从文件读取完整观测数据
            obs = self.data_client.get_observation()
            if obs is None:
                return None

            # 处理关节数据
            joint_data = {}
            if 'arm_joints' in obs:
                for arm_name, joint_values in obs['arm_joints'].items():
                    joint_data[arm_name] = joint_values

            # 处理图像数据
            image_data = {}
            if 'images' in obs:
                for cam_key, cam_name in self.camera_mapping.items():
                    if cam_name in obs['images']:
                        img = obs['images'][cam_name]
                        # 调整图像尺寸
                        target_size = self.target_sizes[cam_key]
                        if img.shape[:2] != target_size:
                            img = cv2.resize(img, target_size)
                        image_data[cam_key] = img

            # 更新缓存
            with self.data_lock:
                self.latest_joint_data = joint_data
                self.latest_image_data = image_data
                self.frame_count += 1

            elapsed = (time.time() - start_time) * 1000

            return {
                'joint_data': joint_data,
                'image_data': image_data,
                'timestamp': obs.get('timestamp', time.time()),
                'elapsed_ms': elapsed
            }

        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None

    def get_performance_stats(self):
        """获取性能统计"""
        runtime = time.time() - self.start_time
        return {
            'runtime': runtime,
            'frame_count': self.frame_count,
            'fps': self.frame_count / runtime if runtime > 0 else 0
        }

    def get_obs(self):
        """获取观测数据 - 直接从缓存读取，极快速度"""
        obs_start_time = time.time()

        with self.data_lock:
            # 直接复制缓存数据，无需网络请求
            obs = {
                'arm_joints': self.joint_data.copy(),
                'images': self.image_data.copy()
            }

        obs_time = (time.time() - obs_start_time) * 1000
        print(f"📊 订阅者观测数据获取完成，耗时: {obs_time:.1f}ms")

        return obs

    def is_data_ready(self):
        """检查数据是否准备就绪"""
        with self.data_lock:
            joints_ready = len(self.joint_data) >= 2
            images_ready = len(self.image_data) >= 1
        return joints_ready and images_ready

    def stop(self):
        """停止订阅者"""
        print("🛑 停止ROS订阅者...")

        # 取消订阅
        for subscriber in self.joint_subscribers.values():
            subscriber.unregister()
        for subscriber in self.image_subscribers.values():
            subscriber.unregister()

        print("✅ ROS订阅者已停止")

    def _check_ros_environment(self):
        """检查 ROS 环境"""
        try:
            import subprocess
            result = subprocess.run(['rostopic', 'list'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("✅ ROS 环境可用")
            else:
                raise Exception("rostopic 命令失败")
        except Exception as e:
            print(f"❌ ROS 环境检查失败: {e}")
            raise

    def _start_image_saver(self):
        """启动预置的图像保存器"""
        try:
            import tempfile
            import subprocess

            # 创建持久的临时目录
            self.image_save_dir = tempfile.mkdtemp(prefix='ros_images_')
            print(f"📁 图像保存目录: {self.image_save_dir}")

            # 启动持久的 image_saver 进程
            saver_cmd = [
                'rosrun', 'image_view', 'image_saver',
                'image:=/camera_l/color/image_raw',
                '_save_all_image:=true',
                '_filename_format:=img_%04i.jpg',
                '__name:=persistent_image_saver'
            ]

            self.image_saver_process = subprocess.Popen(
                saver_cmd,
                cwd=self.image_save_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            print("✅ 预置图像保存器启动成功")
            time.sleep(1)  # 等待启动完成

        except Exception as e:
            print(f"❌ 启动图像保存器失败: {e}")
            self.image_saver_process = None
            self.image_save_dir = None









    def is_data_ready(self):
        """检查数据是否准备就绪"""
        with self.data_lock:
            joints_ready = len(self.cached_joint_data) >= 2
            images_ready = len(self.cached_image_data) >= 1
        return joints_ready and images_ready

    def stop(self):
        """停止后台更新和清理资源"""
        print("🛑 停止 ROS 数据读取器...")
        self.running = False

        # 停止图像保存器进程
        if self.image_saver_process:
            try:
                self.image_saver_process.terminate()
                self.image_saver_process.wait(timeout=3)
                print("✅ 图像保存器已停止")
            except:
                print("⚠️  强制停止图像保存器")
                try:
                    self.image_saver_process.kill()
                except:
                    pass

        # 清理临时目录
        if self.image_save_dir:
            try:
                import shutil
                shutil.rmtree(self.image_save_dir)
                print("✅ 临时目录已清理")
            except:
                print("⚠️  临时目录清理失败")


class PolicyInference:
    def __init__(self, host="localhost", port=8000):
        """
        参数：
        host (str): OpenPI推理服务器主机地址
        port (int): OpenPI推理服务器端口
        """
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI客户端模块不可用，无法进行推理")

        self.host = host
        self.port = port
        self.policy = None
        self.cnt = 0

        # 连接到OpenPI推理服务器
        self._connect_to_server()

    def _connect_to_server(self):
        """连接到OpenPI推理服务器"""
        try:
            print(f"🔗 连接到OpenPI推理服务器 {self.host}:{self.port}...")
            self.policy = websocket_client_policy.WebsocketClientPolicy(
                host=self.host, port=self.port
            )
            print("✅ OpenPI推理服务器连接成功")
            return True
        except Exception as e:
            print(f"❌ OpenPI推理服务器连接失败: {e}")
            self.policy = None
            return False

    def reset(self):
        """重置策略状态"""
        if self.policy is not None:
            try:
                # OpenPI策略可能没有reset方法，这里做安全处理
                if hasattr(self.policy, 'reset'):
                    self.policy.reset()
            except Exception as e:
                print(f"⚠️  策略重置失败: {e}")

    def prepare_inference_obs(self, obs):
        """
        准备用于OpenPI推理的观测数据。

        参数：
        obs (dict): 包含图像和位置信息的观测数据。

        返回：
        inference_data (dict): 处理后的观测数据，可用于OpenPI推理。
        """
        # OpenPI使用的图像键映射
        image_key_map = {
            "left": "left_wrist_0_rgb",
            "right": "right_wrist_0_rgb",
            "front": "base_0_rgb"
        }

        # 准备图像数据字典 - 使用OpenPI格式
        images = {}
        available_cameras = ['left', 'right', 'front']

        for cam_name in available_cameras:
            # 使用正确的图像键映射
            mapped_key = image_key_map.get(cam_name, cam_name)

            if mapped_key in obs['images']:
                cam_img = obs['images'][mapped_key]

                # 确保图像是numpy数组格式
                if isinstance(cam_img, np.ndarray):
                    # 确保是 224x224 尺寸和RGB格式
                    if cam_img.shape[:2] != (224, 224):
                        cam_img = cv2.resize(cam_img, dsize=(224, 224))
                    # 确保是RGB格式（如果是BGR则转换）
                    if len(cam_img.shape) == 3 and cam_img.shape[2] == 3:
                        # 假设输入已经是RGB格式
                        pass
                else:
                    # 如果不是numpy数组格式，直接报错
                    raise ValueError(f"图像数据格式错误，期望numpy数组，实际得到: {type(cam_img)}, shape: {getattr(cam_img, 'shape', 'N/A')}")
                   

                # 使用OpenPI期望的相机名称
                images[mapped_key] = cam_img
                
                print(f"📷 处理 {cam_name} -> {mapped_key} 图像: {cam_img.shape}")
            else:
                print(f"⚠️  缺少 {cam_name} 相机数据")
                # 创建黑色占位图像
                images[mapped_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 处理状态信息 - 扩展到32维以匹配OpenPI服务器期望
        qpos = np.concatenate([obs['arm_joints']['left'], obs['arm_joints']['right']])
        padded_state = np.zeros(32, dtype=np.float32)
        padded_state[:len(qpos)] = qpos

        # 使用OpenPI tokenizer处理提示词
        prompt_text = "pick and place purple long eggplant"

        # 简单的tokenize函数
        def simple_tokenize(text, max_length=48):
            """简单的tokenize函数，将每个字符转为ASCII码"""
            tokens = [ord(c) % 1000 for c in text]
            result = np.zeros((1, max_length), dtype=np.int32)
            mask = np.zeros((1, max_length), dtype=bool)
            length = min(len(tokens), max_length)
            result[0, :length] = tokens[:length]
            mask[0, :length] = True
            return result, mask

        tokens, tokenized_prompt_mask = simple_tokenize(prompt_text)

        # 构建OpenPI格式的观测数据
        inference_data = {
            "state": padded_state,
            "image": images,
            "prompt": prompt_text,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": tokenized_prompt_mask
        }

        self.cnt += 1
        print(f"🔧 准备OpenPI推理数据 (第{self.cnt}次): state维度={len(padded_state)}, 图像数量={len(images)}")

        # 调试：只在第一次保存图像以检查数据
        if self.cnt == 1:  # 只保存第一次的图像用于调试
            print("🖼️  保存第一次推理的图像用于调试...")
            self._save_debug_images(images, self.cnt)

        return inference_data

    def _save_debug_images(self, images, step):
        """保存调试图像以检查数据质量"""
        try:
            import os
            debug_dir = "/tmp/openpi_debug_images"
            os.makedirs(debug_dir, exist_ok=True)

            for cam_name, img_data in images.items():
                if isinstance(img_data, np.ndarray):
                    # 保存为图像文件
                    import cv2
                    filename = f"{debug_dir}/step_{step}_{cam_name}.jpg"
                    # 转换RGB到BGR用于OpenCV保存
                    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, img_bgr)
                    print(f"🖼️  调试图像已保存: {filename}")
        except Exception as e:
            print(f"⚠️  保存调试图像失败: {e}")

    def infer(self, obs=None):
        """
        使用OpenPI模型进行推理。

        参数：
        obs (dict, optional): 观测数据。如果未提供，则返回错误。

        返回：
        output_dict (dict): OpenPI模型的输出结果。
        """
        if self.policy is None:
            raise ValueError("OpenPI策略未初始化 - 服务器连接失败")

        if obs is None:
            raise ValueError("观测数据不能为空")

        # 准备OpenPI格式的输入数据
        input_data = self.prepare_inference_obs(obs)

        # 使用OpenPI策略进行推理
        try:
            start_time = time.time()
            result = self.policy.infer(input_data)
            inference_time = time.time() - start_time

            print(f"🤖 OpenPI推理完成，耗时: {inference_time:.3f}秒")
            print(f"🔍 推理结果键: {list(result.keys())}")

            # 处理推理结果，提取动作或状态
            if "actions" in result:
                action_data = result["actions"]
                print(f"📊 获取到动作数据: {type(action_data)}")
                return action_data
            elif "state" in result:
                state_data = result["state"]
                print(f"📊 获取到状态数据: {type(state_data)}")
                return state_data
            else:
                print(f"⚠️  未找到预期的动作或状态数据，返回原始结果")
                return result

        except Exception as e:
            print(f"❌ OpenPI推理失败: {e}")
            # 尝试重新连接
            if "connection" in str(e).lower() or "websocket" in str(e).lower():
                print("🔄 尝试重新连接OpenPI服务器...")
                if self._connect_to_server():
                    print("✅ 重连成功，重试推理...")
                    return self.infer(obs)
            raise


class JointInference:
    def __init__(self, config_path=None, host="localhost", port=8000):
        if config_path == None:
            config_path = "/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml"

        print(f"🔧 加载配置文件: {config_path}")
        print(f"🔧 OpenPI推理服务器: {host}:{port}")

        # 使用文件数据缓冲器
        print("🔧 使用文件数据缓冲器，从数据服务读取数据")
        print("✅ 这解决了 Python 3.11 与 ROS Noetic 的兼容性问题")

        # 初始化文件数据缓冲器
        try:
            self.data_buffer = FileDataBuffer(data_dir="./data")
            print("✅ 文件数据缓冲器初始化成功")
        except Exception as e:
            print(f"❌ 文件数据缓冲器初始化失败: {e}")
            raise

        # 用于监控关节状态变化
        self.previous_joint_state = None

        # 机械臂使能状态（假设ROS launch已经使能）
        self.robot_enabled = True

        # 帧计数器
        self.frame_count = 0

        # 加载配置文件获取 home 位置
        try:
            try:
                import tomli as tomllib
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)
            except ImportError:
                # 如果 tomli 不可用，尝试使用 toml
                import toml
                with open(config_path, 'r') as f:
                    config = toml.load(f)

            print(f"✅ 配置加载成功，机器人类型: {config['basic']['station_type']}")

            if 'robot' in config and 'arm' in config['robot'] and 'home' in config['robot']['arm']:
                self.home_left = config['robot']['arm']['home']['left']
                self.home_right = config['robot']['arm']['home']['right']
                print(f"📍 左臂 home 位置: {self.home_left}")
                print(f"📍 右臂 home 位置: {self.home_right}")
            else:
                # 使用默认 home 位置
                self.home_left = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.home_right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                print("⚠️  使用默认 home 位置")

        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            # 使用默认 home 位置
            self.home_left = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.home_right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            print("⚠️  使用默认 home 位置")

        # 不使用 xROCS 工作站
        self.robot_station = None

        # 暂停xROCS和ROS客户端，恢复到之前的方案
        self.ros_client = None
        self.data_reader = None

        # 直接使用内置的数据获取方法（已优化超时时间）
        print("✅ 使用内置的优化数据获取方法")

        # 初始化OpenPI推理引擎
        try:
            self.inference_engine = PolicyInference(host=host, port=port)
            print("✅ OpenPI推理引擎初始化成功!")
        except Exception as e:
            print(f"❌ OpenPI推理引擎初始化失败: {e}")
            self.inference_engine = None

        print("✅ JointInference 初始化完成!")

    def stop(self):
        """停止推理系统并清理资源"""
        if hasattr(self, 'ros_client') and self.ros_client and hasattr(self.ros_client, 'disconnect'):
            self.ros_client.disconnect()
        print("🛑 JointInference 已停止")

    def get_obs(self):
        """
        获取观测数据 - 使用文件数据缓冲器

        返回：
        obs (dict): 包含 'arm_joints' 和 'images' 的观测数据
        """
        obs_start_time = time.time()

        # 使用文件数据缓冲器获取数据
        data_result = self.data_buffer.get_current_data()

        if data_result is None:
            print("❌ 无法从文件获取数据")
            return None

        # 转换为标准格式
        obs = {
            'arm_joints': data_result['joint_data'],
            'images': {}
        }

        # 转换图像数据格式 (从 left/right/front 到标准相机名称)
        image_mapping = {
            'left': 'left_wrist_0_rgb',
            'right': 'right_wrist_0_rgb',
            'front': 'base_0_rgb'
        }

        for key, cam_name in image_mapping.items():
            if key in data_result['image_data']:
                obs['images'][cam_name] = data_result['image_data'][key]
            else:
                # 如果某个相机数据缺失，创建黑色占位图像
                obs['images'][cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 更新帧计数
        self.frame_count += 1

        total_obs_time = (time.time() - obs_start_time) * 1000
        file_time = data_result.get('elapsed_ms', 0)
        print(f"📊 文件数据获取 - 文件读取: {file_time:.1f}ms, 总计: {total_obs_time:.1f}ms")

        # 监控关节状态变化
        if obs and 'arm_joints' in obs:
            self._monitor_joint_changes(obs['arm_joints'])

        return obs

    def prepare_inference_obs(self, obs):
        """
        准备用于OpenPI推理的观测数据。

        参数：
        obs (dict): 包含图像和位置信息的观测数据。

        返回：
        inference_data (dict): 处理后的观测数据，可用于OpenPI推理。
        """
        # OpenPI使用的图像键映射
        image_key_map = {
            'left': 'left_wrist_0_rgb',
            'right': 'right_wrist_0_rgb',
            'front': 'base_0_rgb'
        }

        # 可用的相机列表
        available_cameras = ['left', 'right', 'front']

        # 处理图像数据
        images = {}
        for cam_name in available_cameras:
            # 使用正确的图像键映射
            mapped_key = image_key_map.get(cam_name, cam_name)

            if mapped_key in obs['images']:
                cam_img = obs['images'][mapped_key]

                # 确保图像是numpy数组格式
                if isinstance(cam_img, np.ndarray):
                    # 确保是 224x224 尺寸和RGB格式
                    if cam_img.shape[:2] != (224, 224):
                        cam_img = cv2.resize(cam_img, dsize=(224, 224))
                    # 确保是RGB格式（如果是BGR则转换）
                    if len(cam_img.shape) == 3 and cam_img.shape[2] == 3:
                        # 假设输入已经是RGB格式
                        pass
                else:
                    # 如果是编码格式，解码并转换
                    cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
                    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
                    cam_img = cv2.resize(cam_img, dsize=(224, 224))

                # 使用OpenPI期望的相机名称
                images[mapped_key] = cam_img

                print(f"📷 处理 {cam_name} -> {mapped_key} 图像: {cam_img.shape}")
            else:
                print(f"⚠️  缺少 {cam_name} 相机数据")
                # 创建黑色占位图像
                images[mapped_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 处理状态信息 - 扩展到32维以匹配OpenPI服务器期望
        qpos = np.concatenate([obs['arm_joints']['left'], obs['arm_joints']['right']])
        padded_state = np.zeros(32, dtype=np.float32)
        padded_state[:len(qpos)] = qpos

        # 使用OpenPI tokenizer处理提示词
        prompt_text = "pick and place purple long eggplant"

        # 简单的tokenize函数
        def simple_tokenize(text, max_length=77):
            # 简单的字符级tokenization
            tokens = [ord(c) for c in text[:max_length]]
            # 填充到固定长度
            while len(tokens) < max_length:
                tokens.append(0)
            mask = [1] * min(len(text), max_length) + [0] * max(0, max_length - len(text))
            return np.array(tokens, dtype=np.int32), np.array(mask, dtype=np.bool_)

        tokens, tokenized_prompt_mask = simple_tokenize(prompt_text)

        # 构建OpenPI格式的观测数据
        inference_data = {
            "state": padded_state,
            "image": images,
            "prompt": prompt_text,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": tokenized_prompt_mask
        }

        print(f"🔧 准备OpenPI推理数据 (第{self.frame_count}次): state维度={len(padded_state)}, 图像数量={len(images)}")

        return inference_data

    def is_data_ready(self):
        """检查数据是否准备就绪"""
        try:
            # 检查数据客户端是否可以获取数据
            info = self.data_buffer.data_client.get_data_info()
            return info['joint_file_exists'] and len(info['available_cameras']) > 0
        except Exception as e:
            print(f"⚠️  数据准备状态检查失败: {e}")
            return False



    def _monitor_joint_changes(self, current_joints):
        """监控关节状态变化"""
        try:
            if self.previous_joint_state is not None:
                # 计算关节变化
                left_diff = np.abs(np.array(current_joints['left']) - np.array(self.previous_joint_state['left']))
                right_diff = np.abs(np.array(current_joints['right']) - np.array(self.previous_joint_state['right']))

                max_left_change = np.max(left_diff)
                max_right_change = np.max(right_diff)

                if max_left_change > 0.001 or max_right_change > 0.001:
                    print(f"🔄 关节状态变化 - 左臂最大变化: {max_left_change:.4f}, 右臂最大变化: {max_right_change:.4f}")
                else:
                    print("⚠️  关节状态无明显变化，机械臂可能未响应命令")

            # 更新上一次状态
            self.previous_joint_state = {
                'left': current_joints['left'].copy() if hasattr(current_joints['left'], 'copy') else list(current_joints['left']),
                'right': current_joints['right'].copy() if hasattr(current_joints['right'], 'copy') else list(current_joints['right'])
            }
        except Exception as e:
            print(f"⚠️  关节状态监控失败: {e}")

    def is_data_ready(self):
        """检查观测数据是否准备就绪"""
        if self.robot_station is not None:
            return True  # xROCS 工作站可用
        else:
            # 对于内置数据获取方法，总是返回True（实时获取）
            return True

    def prepare(self):
        """
        将机械臂移动到 home 位置
        优先使用 xROCS 功能，如果失败则回退到直接 ROS 控制
        """
        print("🏠 开始执行 prepare - 机械臂回到 home 位置...")

        # 如果有 xROCS 工作站，使用 xROCS 方法
        if self.robot_station is not None:
            try:
                print("🔧 使用 xROCS 工作站控制机械臂...")

                # 使用 xROCS 的方式移动到 home 位置
                for name, _robot in self.robot_station.get_robot_handle().items():
                    home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                                  num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
                    _robot.reach_target_joint(home)

                # 如果有夹爪，打开夹爪
                for gripper in self.robot_station.get_gripper_handle().values():
                    gripper.open()

                time.sleep(2)
                logger.success('🎉 Resetting to home success!')
                print("🎉 xROCS Prepare 执行成功!")
                return True

            except Exception as e:
                print(f"❌ xROCS 控制失败: {e}")
                print("⚠️  回退到直接 ROS 控制...")
                # 继续执行下面的直接 ROS 控制代码

        # 直接 ROS 控制方法（回退方案）- 使用新的关节控制功能
        try:
            # 获取 home 位置
            if hasattr(self, 'home_left') and hasattr(self, 'home_right'):
                home_left = self.home_left
                home_right = self.home_right
            else:
                # 从标准配置中获取
                home_left = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                home_right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            print(f"🎯 左臂 home 位置: {home_left}")
            print(f"🎯 右臂 home 位置: {home_right}")

            # 跳过使能检查（假设ROS launch已经使能）
            print("ℹ️  跳过使能检查，假设ROS launch已经使能机械臂")

            # 使用异步双臂控制发送 home 位置命令
            print("🎯 发送双臂 home 位置命令...")
            if not self.send_dual_arm_command_async(home_left, home_right):
                raise Exception("双臂 home 位置控制失败")

            # 等待机器人移动
            print("⏳ 等待机器人移动到 home 位置...")
            time.sleep(3)

            logger.success('🎉 Resetting to home success!')
            print("🎉 关节控制 Prepare 执行成功!")
            return True

        except Exception as e:
            print(f"❌ Prepare 执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def enable_robot(self, force=False):
        """使能机械臂（只在需要时执行）"""
        if self.robot_enabled and not force:
            print("ℹ️  机械臂已使能，跳过重复使能")
            return True

        try:
            import subprocess
            enable_start_time = time.time()
            cmd = 'timeout 2 rostopic pub /enable_flag std_msgs/Bool "data: true" -r 10'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
            enable_time = (time.time() - enable_start_time) * 1000

            if result.returncode == 0:
                print(f"✅ 机械臂使能成功，耗时: {enable_time:.1f}ms")
                self.robot_enabled = True
                return True
            else:
                print(f"❌ 机械臂使能失败，耗时: {enable_time:.1f}ms，错误: {result.stderr}")
                return False
        except Exception as e:
            enable_time = (time.time() - enable_start_time) * 1000
            print(f"❌ 机械臂使能异常，耗时: {enable_time:.1f}ms，错误: {e}")
            return False
    def send_joint_command_simple(self, arm='left', joint_positions=None):
        """发送简洁的关节控制命令（底层已处理所有安全措施）"""
        if joint_positions is None:
            joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            import subprocess
            cmd_start_time = time.time()

            # 转换为Python列表（如果是numpy数组）
            if hasattr(joint_positions, 'tolist'):
                joint_positions = joint_positions.tolist()

            # 直接发送，不等待确认（移除--once以避免等待）
            cmd = f'''timeout 0.5 rostopic pub /master/joint_{arm} sensor_msgs/JointState "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
name: ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
position: {joint_positions}
velocity: []
effort: []" -r 10'''

            # 使用异步方式发送，不等待返回
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd_end_time = time.time()
            cmd_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
            cmd_time = (cmd_end_time - cmd_start_time) * 1000

            print(f"✅ {arm}臂关节命令已发送 [{cmd_end_timestamp}] (耗时: {cmd_time:.1f}ms)")
            return True

        except Exception as e:
            cmd_time = (time.time() - cmd_start_time) * 1000
            print(f"❌ {arm}臂关节控制异常，耗时: {cmd_time:.1f}ms，错误: {e}")
            return False

    def send_joint_command(self, arm='left', joint_positions=None):
        """发送关节控制命令（简洁版本）"""
        return self.send_joint_command_simple(arm, joint_positions)

    def send_dual_arm_command_async(self, left_positions=None, right_positions=None):
        """异步发送双臂关节控制命令"""
        import concurrent.futures

        if left_positions is None:
            left_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if right_positions is None:
            right_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        start_time = time.time()

        # 使用线程池同时发送左右臂命令
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个任务
            left_future = executor.submit(self.send_joint_command_simple, 'left', left_positions)
            right_future = executor.submit(self.send_joint_command_simple, 'right', right_positions)

            # 等待两个任务完成
            left_success = left_future.result()
            right_success = right_future.result()

        total_time = (time.time() - start_time) * 1000

        if left_success and right_success:
            print(f"✅ 双臂关节命令发送成功，总耗时: {total_time:.1f}ms")
            return True
        else:
            print(f"❌ 双臂关节命令发送失败，总耗时: {total_time:.1f}ms")
            return False



    def send_cartesian_command(self, arm='left', x=0.06, y=0.0, z=0.22, roll=0.0, pitch=1.35, yaw=0.0, gripper=0.0003, mode1=1, mode2=1):
        """发送笛卡尔坐标控制命令"""
        try:
            import subprocess

            cmd = f'''rostopic pub /puppet/pos_cmd_{arm} piper_msgs/PosCmd "x: {x}
y: {y}
z: {z}
roll: {roll}
pitch: {pitch}
yaw: {yaw}
gripper: {gripper}
mode1: {mode1}
mode2: {mode2}" --once'''

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {arm}臂笛卡尔控制命令发送成功")
                return True
            else:
                print(f"❌ {arm}臂笛卡尔控制命令发送失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ {arm}臂笛卡尔控制异常: {e}")
            return False

    def _execute_action_via_ros(self, action_pred, step_count, control_mode='joint', action_step=-1):
        """
        通过ROS命令执行动作

        参数：
        action_pred: OpenPI推理得到的动作数据，形状为(50, 32)或类似
        step_count: 当前步骤计数
        control_mode: 控制模式，'joint'为关节控制，'cartesian'为笛卡尔控制
        action_step: 选择动作序列中的哪一步，-1表示最后一步，0表示第一步

        返回：
        bool: 执行是否成功
        """
        try:
            # OpenPI返回的action_pred形状是(50, 32)，根据action_step参数选择动作
            # 因为训练数据是100fps，20步代表0.2秒后的动作序列
            if len(action_pred.shape) == 2:
                if action_step == 0:
                    current_action = action_pred[0]  # 取20步的动作（0.2秒后的目标）
                    print(f"🎯 使用动作序列的最后一步 (第{len(action_pred)}步，约0.2秒后的目标动作)")
                else:
                    current_action = action_pred[action_step]  # 取指定步的动作
                    print(f"🎯 使用动作序列的第{action_step+1}步动作")
            else:
                current_action = action_pred

            # 确保动作维度正确（应该是32维，前14维是左臂+夹爪，后14维是右臂+夹爪）
            if len(current_action) >= 14:
                # 分离左右臂动作（前7维和后7维）
                left_action = current_action[:7]   # 左臂7个关节（包括夹爪）
                right_action = current_action[7:14] if len(current_action) >= 14 else current_action[7:]  # 右臂7个关节（包括夹爪）

                # 转换为列表以便安全格式化
                left_preview = [f"{x:.3f}" for x in left_action[:3]]
                right_preview = [f"{x:.3f}" for x in right_action[:3]]
                print(f"🎯 步骤 {step_count}: 执行动作({control_mode}模式) - 左臂: {left_preview}..., 右臂: {right_preview}...")

                # 跳过使能检查（假设ROS launch已经使能）
                # 简洁的关节控制（底层已处理所有安全措施）- 使用异步双臂控制
                if control_mode == 'joint':
                    success = self.send_dual_arm_command_async(left_action, right_action)
                    return success

                elif control_mode == 'cartesian':
                    # 笛卡尔控制模式 - 需要将关节动作转换为笛卡尔坐标
                    # 这里简化处理，直接使用前6个值作为位置和姿态
                    if len(left_action) >= 6:
                        success_left = self.send_cartesian_command(
                            'left',
                            x=left_action[0], y=left_action[1], z=left_action[2],
                            roll=left_action[3], pitch=left_action[4], yaw=left_action[5],
                            gripper=left_action[6] if len(left_action) > 6 else 0.0003
                        )
                    else:
                        success_left = False

                    if len(right_action) >= 6:
                        success_right = self.send_cartesian_command(
                            'right',
                            x=right_action[0], y=right_action[1], z=right_action[2],
                            roll=right_action[3], pitch=right_action[4], yaw=right_action[5],
                            gripper=right_action[6] if len(right_action) > 6 else 0.0005
                        )
                    else:
                        success_right = False

                    if success_left and success_right:
                        print(f"✅ 步骤 {step_count}: 笛卡尔控制命令发送成功")
                        return True
                    else:
                        print(f"❌ 步骤 {step_count}: 笛卡尔控制命令发送失败")
                        return False
                else:
                    print(f"❌ 不支持的控制模式: {control_mode}")
                    return False

            else:
                print(f"❌ 动作维度不正确: {len(current_action)}，期望至少14维")
                return False

        except Exception as e:
            print(f"❌ ROS动作执行异常: {e}")
            return False

    def inference(self, data_dir: str, task_name: str):
        """
        推理方法 - 获取观测数据并进行推理控制
        """
        print("🤖 开始推理...")
        print(f"   数据目录: {data_dir}")
        print(f"   任务名称: {task_name}")

        if self.inference_engine is None:
            print("❌ 推理引擎未初始化，无法进行推理")
            return

        if not self.is_data_ready():
            print("❌ 观测数据源未准备就绪")
            return

        # 在开始推理之前，先让机械臂回到home状态
        print("🏠 推理前准备：机械臂回到home位置...")
        try:
            if self.prepare():
                print("✅ 机械臂已回到home位置，准备开始推理")
            else:
                print("⚠️  机械臂回home失败，但继续推理")
        except Exception as e:
            print(f"⚠️  机械臂回home过程中出错: {e}")
            print("⚠️  继续推理，但可能影响初始状态")

        try:
            print("📊 开始推理循环...")
            step_count = 0
            max_steps = 100  # 设置较小的最大步数用于测试

            # 用于控制推理频率的时间记录
            target_interval = 0.1  # 10Hz = 100ms间隔
            last_step_time = time.time()

            # 快速检查数据准备状态，不等待
            print("⏳ 检查观测数据准备状态...")
            if not self.is_data_ready():
                print("⚠️  观测数据未完全准备就绪，但继续推理（将使用实时获取）")
            else:
                print("✅ 观测数据准备就绪")

            print("🚀 开始连续推理循环（无延迟模式）")

            while step_count < max_steps:
                step_start_time = time.time()
                step_start_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                print(f"\n🎯 推理步骤 {step_count}... [开始: {step_start_timestamp}]")

                # 获取观测数据
                obs_start_time = time.time()
                obs_start_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                print(f"📊 开始获取观测数据... [{obs_start_timestamp}]")

                obs = self.get_obs()
                obs_end_time = time.time()
                obs_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                obs_time = (obs_end_time - obs_start_time) * 1000

                if obs is None:
                    print(f"❌ 无法获取观测数据，停止推理 [结束: {obs_end_timestamp}] (耗时: {obs_time:.1f}ms)")
                    print("🛑 真实部署环境，数据获取失败必须停止")
                    break

                print(f"📊 观测数据获取完成 [结束: {obs_end_timestamp}] (耗时: {obs_time:.1f}ms)")

                # 检查观测数据是否有效
                if 'images' not in obs or 'arm_joints' not in obs:
                    print(f"❌ 观测数据不完整，停止推理 [结束: {obs_end_timestamp}] (耗时: {obs_time:.1f}ms)")
                    print("🛑 真实部署环境，数据不完整必须停止")
                    break

                # 检查是否有图像数据
                if not obs['images'] or len(obs['images']) == 0:
                    print(f"❌ 没有图像数据，停止推理 [结束: {obs_end_timestamp}] (耗时: {obs_time:.1f}ms)")
                    print("🛑 真实部署环境，图像数据缺失必须停止")
                    break

                # 使用OpenPI推理引擎进行推理
                try:
                    inference_start_time = time.time()
                    inference_start_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                    print(f"🤖 开始OpenPI推理... [{inference_start_timestamp}]")

                    result = self.inference_engine.infer(obs)

                    inference_end_time = time.time()
                    inference_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                    inference_time = (inference_end_time - inference_start_time) * 1000
                    print(f"🤖 OpenPI推理完成 [结束: {inference_end_timestamp}] (耗时: {inference_time:.1f}ms)")

                    # 处理OpenPI的推理结果
                    process_start_time = time.time()
                    if isinstance(result, dict):
                        # 如果返回字典，尝试提取动作数据
                        if "actions" in result:
                            action_pred = result["actions"]
                        elif "state" in result:
                            action_pred = result["state"]
                        else:
                            print(f"⚠️  未找到预期的动作数据，使用原始结果")
                            action_pred = result
                    else:
                        # 如果直接返回数组
                        action_pred = result

                    # 确保action_pred是numpy数组格式
                    if hasattr(action_pred, 'cpu'):
                        action_pred = action_pred.cpu().numpy()
                    elif not isinstance(action_pred, np.ndarray):
                        action_pred = np.array(action_pred)

                    process_time = (time.time() - process_start_time) * 1000
                    print(f"📊 动作数据处理完成，耗时: {process_time:.1f}ms，形状: {action_pred.shape if hasattr(action_pred, 'shape') else type(action_pred)}")

                    # 执行动作 - 优先使用xROCS，回退到ROS关节控制命令
                    action_start_time = time.time()
                    action_start_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                    print(f"🎯 开始执行动作... [{action_start_timestamp}]")

                    if self.robot_station is not None:
                        robot_targets = self.robot_station.decompose_action(action_pred)
                        obs = self.robot_station.step(robot_targets)
                        action_end_time = time.time()
                        action_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                        action_time = (action_end_time - action_start_time) * 1000
                        print(f"✅ xROCS动作执行完成 [结束: {action_end_timestamp}] (耗时: {action_time:.1f}ms)")
                    elif self.ros_client is not None:
                        # 使用ROS客户端执行动作（使用最后一步的动作）
                        final_action = action_pred[-1] if len(action_pred.shape) > 1 else action_pred
                        success = self.ros_client.execute_action(final_action)
                        action_end_time = time.time()
                        action_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                        action_time = (action_end_time - action_start_time) * 1000
                        if success:
                            print(f"✅ ROS客户端动作执行完成 [结束: {action_end_timestamp}] (耗时: {action_time:.1f}ms)")
                        else:
                            print(f"❌ ROS客户端动作执行失败 [结束: {action_end_timestamp}] (耗时: {action_time:.1f}ms)")
                    else:
                        # 使用ROS关节控制命令执行动作（回退方案）
                        success = self._execute_action_via_ros(action_pred, step_count, control_mode='joint', action_step=-1)
                        action_end_time = time.time()
                        action_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                        action_time = (action_end_time - action_start_time) * 1000
                        if success:
                            print(f"✅ ROS关节控制动作执行完成 [结束: {action_end_timestamp}] (耗时: {action_time:.1f}ms)")
                        else:
                            print(f"❌ ROS关节控制动作执行失败 [结束: {action_end_timestamp}] (耗时: {action_time:.1f}ms)")

                except Exception as infer_e:
                    inference_error_time = (time.time() - inference_start_time) * 1000
                    print(f"❌ 步骤 {step_count} 推理失败，耗时: {inference_error_time:.1f}ms，错误: {infer_e}")
                    # 继续下一步，不中断整个循环

                # 计算整个步骤的总耗时
                step_end_time = time.time()
                step_end_timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                step_total_time = (step_end_time - step_start_time) * 1000
                print(f"⏱️  步骤 {step_count} 完成 [结束: {step_end_timestamp}] (总耗时: {step_total_time:.1f}ms)")

                step_count += 1

                # 控制推理频率为10Hz（100ms间隔）
                current_time = time.time()
                elapsed = current_time - last_step_time
                if elapsed < target_interval:
                    sleep_time = target_interval - elapsed
                    print(f"⏳ 等待 {sleep_time*1000:.1f}ms 以维持10Hz频率")
                    time.sleep(sleep_time)
                last_step_time = time.time()

                # 每10步打印一次进度和性能统计
                if step_count % 10 == 0:
                    actual_interval = (time.time() - last_step_time + target_interval) * 1000
                    print(f"🔄 已执行 {step_count} 步推理，实际间隔: {actual_interval:.1f}ms")

            print(f"✅ 推理完成，共执行 {step_count} 步")

        except KeyboardInterrupt:
            print("⏹️  用户中断推理")
        except Exception as e:
            print(f"❌ 推理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()



