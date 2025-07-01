#!/usr/bin/env python3
"""
ROS客户端 - 用于openpi推理脚本连接ROS数据服务
Python 3.11兼容，无需ROS依赖
"""

import socket
import json
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple

class ROSClient:
    """ROS数据服务客户端"""
    
    def __init__(self, host='localhost', port=8888, timeout=5.0):
        """
        初始化ROS客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 连接超时时间
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        
        print(f"🔧 初始化ROS客户端: {host}:{port}")

    def connect(self) -> bool:
        """连接到ROS数据服务"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✅ 连接到ROS数据服务成功: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ 连接ROS数据服务失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        print("🔌 已断开ROS数据服务连接")

    def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送请求并接收响应"""
        if not self.connected:
            print("❌ 未连接到ROS数据服务")
            return None
        
        try:
            # 发送请求
            request_data = json.dumps(request, default=self._json_serializer)
            self.socket.send(request_data.encode('utf-8'))
            
            # 接收响应（增加缓冲区大小）
            response_data = self.socket.recv(1024 * 1024)  # 1MB缓冲区
            response = json.loads(response_data.decode('utf-8'))
            
            return response
            
        except Exception as e:
            print(f"❌ 请求发送失败: {e}")
            self.connected = False
            return None

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)

    def get_obs(self) -> Optional[Dict[str, Any]]:
        """
        获取观测数据
        
        Returns:
            观测数据字典，包含 'arm_joints' 和 'images'
        """
        start_time = time.time()
        
        request = {'command': 'get_obs'}
        response = self._send_request(request)
        
        if response and response.get('success'):
            elapsed = (time.time() - start_time) * 1000
            server_elapsed = response.get('elapsed_ms', 0)
            
            print(f"📊 观测数据获取成功 - 客户端耗时: {elapsed:.1f}ms, 服务端耗时: {server_elapsed:.1f}ms")
            
            # 获取图像数据（分别请求每个相机）
            obs_data = response['data']
            if 'images' in obs_data:
                images = {}
                for cam_name, img_meta in obs_data['images'].items():
                    img_data = self.get_image(cam_name)
                    if img_data is not None:
                        images[cam_name] = img_data
                obs_data['images'] = images
            
            # 转换关节数据回numpy数组
            if 'arm_joints' in obs_data:
                for arm_name, joint_list in obs_data['arm_joints'].items():
                    if joint_list is not None:
                        obs_data['arm_joints'][arm_name] = np.array(joint_list, dtype=np.float32)
            
            return obs_data
        else:
            error = response.get('error', 'Unknown error') if response else 'No response'
            print(f"❌ 获取观测数据失败: {error}")
            return None

    def get_image(self, camera_name: str) -> Optional[np.ndarray]:
        """
        获取单个相机的图像数据

        Args:
            camera_name: 相机名称 ('left', 'right', 'front')

        Returns:
            图像numpy数组 (128, 128, 3) uint8 RGB格式
        """
        request = {'command': 'get_image', 'camera': camera_name}
        response = self._send_request(request)

        if response and response.get('success'):
            img_list = response.get('image')
            if img_list:
                img_array = np.array(img_list, dtype=np.uint8)
                return img_array

        return None

    def execute_action(self, action: np.ndarray) -> bool:
        """
        执行动作
        
        Args:
            action: 动作数组
            
        Returns:
            是否执行成功
        """
        start_time = time.time()
        
        request = {
            'command': 'execute_action',
            'action': action.tolist() if isinstance(action, np.ndarray) else action
        }
        
        response = self._send_request(request)
        
        if response and response.get('success'):
            elapsed = (time.time() - start_time) * 1000
            server_elapsed = response.get('elapsed_ms', 0)
            
            print(f"✅ 动作执行成功 - 客户端耗时: {elapsed:.1f}ms, 服务端耗时: {server_elapsed:.1f}ms")
            return True
        else:
            error = response.get('error', 'Unknown error') if response else 'No response'
            print(f"❌ 动作执行失败: {error}")
            return False

    def prepare_robot(self) -> bool:
        """
        机器人准备（回到home位置）
        
        Returns:
            是否准备成功
        """
        request = {'command': 'prepare'}
        response = self._send_request(request)
        
        if response and response.get('success'):
            print("✅ 机器人准备成功")
            return True
        else:
            error = response.get('error', 'Unknown error') if response else 'No response'
            print(f"❌ 机器人准备失败: {error}")
            return False

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取服务统计信息
        
        Returns:
            统计信息字典
        """
        request = {'command': 'get_stats'}
        response = self._send_request(request)
        
        if response:
            return response
        else:
            print("❌ 获取统计信息失败")
            return None

    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.connected

    def wait_for_connection(self, max_wait=30.0, retry_interval=1.0) -> bool:
        """
        等待连接到服务
        
        Args:
            max_wait: 最大等待时间（秒）
            retry_interval: 重试间隔（秒）
            
        Returns:
            是否连接成功
        """
        start_time = time.time()
        
        print(f"⏳ 等待连接到ROS数据服务...")
        
        while time.time() - start_time < max_wait:
            if self.connect():
                return True
            
            print(f"   重试中... ({time.time() - start_time:.1f}s/{max_wait}s)")
            time.sleep(retry_interval)
        
        print(f"❌ 连接超时 ({max_wait}s)")
        return False

class MockROSClient:
    """模拟ROS客户端，用于测试"""
    
    def __init__(self):
        print("🔧 初始化模拟ROS客户端")
        self.connected = True

    def connect(self) -> bool:
        print("✅ 模拟连接成功")
        return True

    def disconnect(self):
        print("🔌 模拟断开连接")
        self.connected = False

    def get_obs(self) -> Dict[str, Any]:
        """生成模拟观测数据"""
        start_time = time.time()
        
        obs = {
            'arm_joints': {
                'left': np.random.uniform(-1, 1, 7).astype(np.float32),
                'right': np.random.uniform(-1, 1, 7).astype(np.float32)
            },
            'images': {
                'left': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                'right': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                'front': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            }
        }
        
        elapsed = (time.time() - start_time) * 1000
        print(f"📊 模拟观测数据生成完成，耗时: {elapsed:.1f}ms")
        
        return obs

    def execute_action(self, action: np.ndarray) -> bool:
        """模拟执行动作"""
        start_time = time.time()
        time.sleep(0.001)  # 模拟执行时间
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ 模拟动作执行完成，耗时: {elapsed:.1f}ms")
        return True

    def prepare_robot(self) -> bool:
        """模拟机器人准备"""
        print("✅ 模拟机器人准备完成")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """返回模拟统计信息"""
        return {
            'runtime': 100.0,
            'obs_count': 1000,
            'control_count': 500,
            'obs_rate': 10.0,
            'control_rate': 5.0
        }

    def is_connected(self) -> bool:
        return self.connected

    def wait_for_connection(self, max_wait=30.0, retry_interval=1.0) -> bool:
        return True

def test_client():
    """测试客户端"""
    print("🧪 测试ROS客户端...")
    
    # 尝试连接真实服务
    client = ROSClient()
    
    if client.wait_for_connection(max_wait=5.0):
        print("✅ 使用真实ROS客户端")
        
        # 测试获取观测数据
        obs = client.get_obs()
        if obs:
            print(f"  - 关节数据: {list(obs['arm_joints'].keys())}")
            print(f"  - 图像数据: {list(obs['images'].keys())}")
        
        # 测试统计信息
        stats = client.get_stats()
        if stats:
            print(f"  - 运行时间: {stats.get('runtime', 0):.1f}s")
            print(f"  - 观测频率: {stats.get('obs_rate', 0):.1f}Hz")
        
        client.disconnect()
    else:
        print("⚠️  真实服务不可用，使用模拟客户端")
        client = MockROSClient()
        
        # 测试模拟功能
        obs = client.get_obs()
        print(f"  - 关节数据: {list(obs['arm_joints'].keys())}")
        print(f"  - 图像数据: {list(obs['images'].keys())}")
        
        stats = client.get_stats()
        print(f"  - 模拟统计: {stats}")
    
    print("✅ 测试完成")

if __name__ == '__main__':
    test_client()
