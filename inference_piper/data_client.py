#!/usr/bin/env python3
"""
数据客户端 - 用于OpenPI读取ROS数据服务保存的文件
支持Python 3.11，不依赖ROS
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class DataClient:
    """数据客户端 - 从文件读取机器人数据"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化数据客户端
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.joint_file = self.data_dir / "current_joints.json"
        self.image_dir = self.data_dir / "images"
        
        print(f"📁 数据客户端初始化，数据目录: {data_dir}")
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            print(f"⚠️  数据目录不存在: {data_dir}")
            print("💡 请先启动ROS数据服务来创建数据目录")

        if not self.image_dir.exists():
            print(f"⚠️  图像目录不存在: {self.image_dir}")

    def read_current_joints(self) -> Optional[Dict]:
        """
        读取当前关节角数据
        
        Returns:
            关节数据字典，格式: {
                'left': {'values': [joint_angles], 'timestamp': timestamp},
                'right': {'values': [joint_angles], 'timestamp': timestamp}
            }
        """
        try:
            if not self.joint_file.exists():
                print("⚠️  关节数据文件不存在")
                return None
                
            with open(self.joint_file, 'r') as f:
                joint_data = json.load(f)
                
            return joint_data
            
        except Exception as e:
            print(f"❌ 读取关节数据失败: {e}")
            return None

    def read_current_image(self, camera_name: str) -> Optional[np.ndarray]:
        """
        读取当前相机图像数据

        Args:
            camera_name: 相机名称

        Returns:
            图像数组 (H, W, C) 或 None
        """
        try:
            img_file = self.image_dir / f"{camera_name}.npy"

            if not img_file.exists():
                print(f"⚠️  相机{camera_name}图像文件不存在")
                return None

            # 检查文件大小
            if img_file.stat().st_size == 0:
                print(f"⚠️  相机{camera_name}图像文件为空")
                return None

            img_data = np.load(img_file)

            # 检查图像数据有效性
            if img_data.size == 0:
                print(f"⚠️  相机{camera_name}图像数据为空")
                return None

            if len(img_data.shape) != 3 or img_data.shape[2] != 3:
                print(f"⚠️  相机{camera_name}图像格式错误: {img_data.shape}")
                return None

            return img_data

        except Exception as e:
            print(f"❌ 读取{camera_name}图像失败: {e}")
            return None

    def list_available_cameras(self) -> List[str]:
        """
        列出可用的相机
        
        Returns:
            相机名称列表
        """
        try:
            camera_files = list(self.image_dir.glob("*.npy"))
            cameras = [f.stem for f in camera_files]
            return cameras
            
        except Exception as e:
            print(f"❌ 列出相机失败: {e}")
            return []

    def get_observation(self) -> Optional[Dict]:
        """
        获取完整的观测数据 (关节角 + 所有相机图像)
        
        Returns:
            观测数据字典，格式: {
                'arm_joints': {arm_name: joint_values},
                'images': {camera_name: image_array},
                'timestamp': timestamp
            }
        """
        try:
            # 读取关节数据
            joint_data = self.read_current_joints()
            if joint_data is None:
                return None
                
            # 读取图像数据
            cameras = self.list_available_cameras()
            images = {}
            
            for cam_name in cameras:
                img = self.read_current_image(cam_name)
                if img is not None:
                    images[cam_name] = img
            
            # 构建观测数据
            obs = {
                'arm_joints': {},
                'images': images,
                'timestamp': time.time()
            }
            
            # 提取关节角度值
            for arm_name, arm_data in joint_data.items():
                if 'values' in arm_data:
                    obs['arm_joints'][arm_name] = np.array(arm_data['values'], dtype=np.float32)
            
            return obs
            
        except Exception as e:
            print(f"❌ 获取观测数据失败: {e}")
            return None

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """
        等待数据文件可用
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否成功等到数据
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.joint_file.exists():
                cameras = self.list_available_cameras()
                if len(cameras) > 0:
                    print("✅ 数据文件已就绪")
                    return True
                    
            time.sleep(0.1)
            
        print("⏰ 等待数据超时")
        return False

    def get_data_info(self) -> Dict:
        """
        获取数据信息
        
        Returns:
            数据信息字典
        """
        info = {
            'joint_file_exists': self.joint_file.exists(),
            'available_cameras': self.list_available_cameras(),
            'data_dir': str(self.data_dir)
        }
        
        # 获取文件修改时间
        if self.joint_file.exists():
            info['joint_file_mtime'] = self.joint_file.stat().st_mtime
            
        return info


def test_data_client():
    """测试数据客户端功能"""
    print("🧪 测试数据客户端...")
    
    try:
        # 创建客户端
        client = DataClient()
        
        # 获取数据信息
        info = client.get_data_info()
        print(f"📊 数据信息: {info}")
        
        # 等待数据
        if client.wait_for_data(timeout=10):
            # 读取关节数据
            joints = client.read_current_joints()
            print(f"🦾 关节数据: {joints}")
            
            # 读取图像数据
            cameras = client.list_available_cameras()
            print(f"📷 可用相机: {cameras}")
            
            for cam_name in cameras[:2]:  # 只测试前两个相机
                img = client.read_current_image(cam_name)
                if img is not None:
                    print(f"🖼️  {cam_name}: {img.shape}, dtype: {img.dtype}")
            
            # 获取完整观测
            obs = client.get_observation()
            if obs:
                print(f"👁️  观测数据: 关节{len(obs['arm_joints'])}个, 图像{len(obs['images'])}个")
                
        else:
            print("❌ 无法获取数据")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == '__main__':
    test_data_client()
