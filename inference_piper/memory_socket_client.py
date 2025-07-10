#!/usr/bin/env python3
"""
内存套接字客户端 - 高性能进程间通信
用于从simple_memory_service获取实时数据
"""

import socket
import pickle
import numpy as np
import time
import os
from pathlib import Path


class MemorySocketClient:
    """内存套接字客户端 - 高性能数据获取"""

    def __init__(self, socket_path="/tmp/xrocs_memory_service.sock"):
        self.socket_path = socket_path
        self.socket = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.last_connection_attempt = 0
        self.connection_retry_interval = 2.0  # 2秒重试间隔
        print(f"🔌 内存套接字客户端初始化: {socket_path}")

    def connect(self):
        """连接到内存服务 - 带重试机制"""
        current_time = time.time()

        # 检查重试间隔
        if (current_time - self.last_connection_attempt) < self.connection_retry_interval:
            return False

        self.last_connection_attempt = current_time

        try:
            # 检查套接字文件是否存在
            if not os.path.exists(self.socket_path):
                if self.connection_attempts == 0:  # 只在第一次尝试时打印
                    print(f"⚠️  套接字文件不存在: {self.socket_path}")
                    print("   请确保 simple_memory_service.py 正在运行")
                self.connection_attempts += 1
                return False

            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.settimeout(3.0)  # 减少超时时间到3秒
            self.socket.connect(self.socket_path)

            # 连接成功，重置计数器
            if self.connection_attempts > 0:
                print("✅ 重新连接到内存服务成功")
            else:
                print("✅ 已连接到内存服务")
            self.connection_attempts = 0
            return True

        except Exception as e:
            self.connection_attempts += 1
            if self.connection_attempts <= 3:  # 只在前3次尝试时打印详细错误
                print(f"❌ 连接失败 (尝试 {self.connection_attempts}): {e}")
            elif self.connection_attempts == 4:
                print(f"❌ 连接持续失败，将减少错误输出...")

            # 清理socket
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("🔌 已断开连接")
    
    def get_latest_data(self):
        """获取最新数据 - 关节+图像，带自动重连"""
        # 尝试连接（如果未连接）
        if not self.socket:
            if not self.connect():
                return None, None

        try:
            # 发送请求
            request = b"GET_LATEST_DATA"
            self.socket.send(request)

            # 接收数据长度（确保接收完整的8字节）
            length_data = b""
            while len(length_data) < 8:
                chunk = self.socket.recv(8 - len(length_data))
                if not chunk:
                    # 连接断开，尝试重连
                    print("⚠️  连接断开，尝试重连...")
                    self.disconnect()
                    if self.connect():
                        # 重连成功，重新发送请求
                        self.socket.send(request)
                        length_data = b""
                        continue
                    else:
                        return None, None
                length_data += chunk

            data_length = int.from_bytes(length_data, byteorder='big')

            # 减少日志输出频率
            if self.connection_attempts == 0:  # 只在连接正常时偶尔打印
                print(f"📦 准备接收数据: {data_length} 字节")

            # 接收实际数据（确保接收完整）
            data_bytes = b""
            received = 0
            while received < data_length:
                remaining = data_length - received
                chunk_size = min(8192, remaining)  # 增大缓冲区到8KB
                chunk = self.socket.recv(chunk_size)
                if not chunk:
                    print("⚠️  数据接收中断，尝试重连...")
                    self.disconnect()
                    return None, None
                data_bytes += chunk
                received += len(chunk)

            if len(data_bytes) != data_length:
                print(f"❌ 数据接收不完整: {len(data_bytes)}/{data_length}")
                self.disconnect()
                return None, None

            # 减少成功日志输出
            if self.connection_attempts == 0 and data_length > 100000:  # 只在大数据包时打印
                print(f"✅ 数据接收完成: {len(data_bytes)} 字节")

            # 反序列化数据
            data = pickle.loads(data_bytes)
            joint_data = data.get('joints')
            image_data = data.get('images')

            return joint_data, image_data

        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # 网络连接错误，尝试重连
            print(f"⚠️  网络连接错误: {e}")
            self.disconnect()
            return None, None
        except Exception as e:
            # 其他错误
            if self.connection_attempts <= 3:
                print(f"❌ 获取数据失败: {e}")
            self.disconnect()
            return None, None
    
    def get_latest_joint_data(self):
        """获取最新关节数据"""
        joint_data, _ = self.get_latest_data()
        return joint_data
    
    def get_latest_image_data(self):
        """获取最新图像数据"""
        _, image_data = self.get_latest_data()
        return image_data


class MemoryDataProvider:
    """内存数据提供器 - 使用套接字通信"""

    def __init__(self):
        self.client = MemorySocketClient()
        print("📁 内存数据提供器初始化: 套接字通信")

        # 相机名称映射：memory_service -> OpenPI
        self.camera_mapping = {
            'camera_f': 'base_0_rgb',
            'camera_l': 'left_wrist_0_rgb',
            'camera_r': 'right_wrist_0_rgb'
        }

        # 数据获取统计
        self.data_request_count = 0
        self.successful_requests = 0
        self.last_successful_time = 0

        print(f"🎥 相机映射: {self.camera_mapping}")
    
    def get_observation(self):
        """获取观测数据 - 通过套接字获取，带统计和错误处理"""
        self.data_request_count += 1
        current_time = time.time()

        try:
            # 通过套接字获取最新数据
            joint_data, image_data = self.client.get_latest_data()

            if not joint_data:
                # 减少无数据时的日志输出频率
                if self.data_request_count % 100 == 1:  # 每100次请求打印一次
                    print(f"⚠️  无关节数据 (请求 {self.data_request_count})")
                    if self.last_successful_time > 0:
                        time_since_last = current_time - self.last_successful_time
                        print(f"   距离上次成功获取数据: {time_since_last:.1f}秒")
                    print("   请检查 simple_memory_service.py 是否正在运行")
                return None

            # 成功获取数据
            self.successful_requests += 1
            self.last_successful_time = current_time

            # 减少成功日志的输出频率
            if self.data_request_count % 50 == 1:  # 每50次请求打印一次
                success_rate = (self.successful_requests / self.data_request_count) * 100
                print(f"📊 数据获取统计: 成功率 {success_rate:.1f}% ({self.successful_requests}/{self.data_request_count})")
                print(f"🔍 关节数据键: {list(joint_data.keys())}")

            # 处理图像数据并转换格式
            images = {}
            
            if image_data:
                for memory_cam, openpi_cam in self.camera_mapping.items():
                    if memory_cam in image_data:
                        img_info = image_data[memory_cam]
                        img_data = img_info.get('data')
                        
                        if img_data is not None:
                            # 验证格式（应该已经是224x224x3 RGB uint8）
                            if img_data.shape == (224, 224, 3) and img_data.dtype == np.uint8:
                                # 转换为float32并归一化到[0,1]范围（模型需要）
                                img_float = img_data.astype(np.float32) / 255.0
                                images[openpi_cam] = img_float
                                print(f"✅ {openpi_cam}: {img_data.shape} -> {img_float.shape} float32")
                            else:
                                print(f"⚠️  {memory_cam}图像格式错误: {img_data.shape}, {img_data.dtype}")
                                images[openpi_cam] = np.zeros((224, 224, 3), dtype=np.float32)
                        else:
                            print(f"⚠️  {memory_cam}图像数据为空")
                            images[openpi_cam] = np.zeros((224, 224, 3), dtype=np.float32)
                    else:
                        print(f"⚠️  {memory_cam}不在图像数据中")
                        images[openpi_cam] = np.zeros((224, 224, 3), dtype=np.float32)
            else:
                # 创建占位图像
                for openpi_cam in self.camera_mapping.values():
                    images[openpi_cam] = np.zeros((224, 224, 3), dtype=np.float32)

            # 处理关节数据
            arm_joints = {}

            # 右臂数据
            if 'right' in joint_data:
                if isinstance(joint_data['right'], dict) and 'values' in joint_data['right']:
                    arm_joints['right'] = np.array(joint_data['right']['values'], dtype=np.float32)
                else:
                    arm_joints['right'] = np.array(joint_data['right'], dtype=np.float32)
                print(f"✅ 右臂关节: {len(arm_joints['right'])}个")
            else:
                print("❌ 缺少右臂关节数据")
                print(f"🔍 当前关节数据内容: {joint_data}")
                return None

            # 左臂数据
            if 'left' in joint_data:
                if isinstance(joint_data['left'], dict) and 'values' in joint_data['left']:
                    arm_joints['left'] = np.array(joint_data['left']['values'], dtype=np.float32)
                else:
                    arm_joints['left'] = np.array(joint_data['left'], dtype=np.float32)
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
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'client'):
            self.client.disconnect()


if __name__ == "__main__":
    # 测试客户端
    client = MemorySocketClient()
    
    try:
        if client.connect():
            print("🧪 测试数据获取...")
            joint_data, image_data = client.get_latest_data()
            
            if joint_data:
                print(f"✅ 关节数据: {list(joint_data.keys())}")
            if image_data:
                print(f"✅ 图像数据: {list(image_data.keys())}")
                
    finally:
        client.disconnect()
