#!/usr/bin/env python3
"""
简化的数据服务 - 专门用于文件保存模式
在base环境下运行，获取ROS话题数据并保存到文件
不依赖xROCS，直接使用ROS话题
"""

import time
import numpy as np
import threading
import json
import cv2
import os
from pathlib import Path
import signal
import sys

# ROS相关导入
try:
    import rospy
    from sensor_msgs.msg import JointState, Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("⚠️  ROS未安装，将使用模拟数据模式")
    ROS_AVAILABLE = False


class SimpleDataService:
    """简化的数据服务 - 直接从ROS话题获取数据并保存到文件"""
    
    def __init__(self, data_dir="./data"):
        """
        初始化数据服务

        Args:
            data_dir: 数据保存目录
        """
        # 数据保存目录
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 数据文件路径
        self.joint_file = self.data_dir / "current_joints.json"
        self.image_dir = self.data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 创建images目录用于保存原始图像
        self.images_save_dir = Path("./images")
        self.images_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self.latest_joint_data = {}
        self.latest_image_data = {}
        self.data_lock = threading.RLock()
        
        # 数据更新线程
        self.data_thread = None
        self.running = False
        
        # ROS相关
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        self.joint_subscribers = {}
        self.image_subscribers = {}
        
        # 性能统计
        self.stats = {
            'joint_updates': 0,
            'image_updates': 0,
            'start_time': time.time()
        }
        
        print(f"🚀 初始化简化数据服务，数据目录: {data_dir}")
        
        if ROS_AVAILABLE:
            self._init_ros()
        else:
            print("❌ ROS不可用，无法获取真实数据")
            sys.exit(1)

    def _init_ros(self):
        """初始化ROS订阅者"""
        try:
            rospy.init_node('simple_data_service', anonymous=True)
            print("✅ ROS节点初始化成功")
            
            # 订阅关节话题
            joint_topics = {
                'left': '/puppet/joint_left',
                'right': '/puppet/joint_right'
            }
            
            for arm_name, topic in joint_topics.items():
                try:
                    self.joint_subscribers[arm_name] = rospy.Subscriber(
                        topic, JointState,
                        lambda msg, arm=arm_name: self._joint_callback(msg, arm),
                        queue_size=1  # 使用最小队列，保持最新数据
                    )
                    print(f"✅ 订阅关节话题: {topic}")
                except Exception as e:
                    print(f"❌ 订阅关节话题失败 {topic}: {e}")

            # 订阅图像话题 - 使用实际可用的相机话题
            image_topics = {
                'left_wrist_0_rgb': '/camera_l/color/image_raw',
                'right_wrist_0_rgb': '/camera_r/color/image_raw',
                'base_0_rgb': '/camera_f/color/image_raw'
            }

            for cam_name, topic in image_topics.items():
                try:
                    self.image_subscribers[cam_name] = rospy.Subscriber(
                        topic, Image,
                        lambda msg, cam=cam_name: self._image_callback(msg, cam),
                        queue_size=1  # 使用最小队列，保持最新数据
                    )
                    print(f"✅ 订阅图像话题: {topic}")
                except Exception as e:
                    print(f"❌ 订阅图像话题失败 {topic}: {e}")

            print("✅ ROS订阅者初始化完成")

        except Exception as e:
            print(f"❌ ROS初始化失败: {e}")
            print("💡 请确保ROS正在运行且话题可用")
            sys.exit(1)


    def _joint_callback(self, msg, arm_name):
        """关节数据回调函数"""
        try:
            current_time = time.time()
            joint_array = np.array(msg.position, dtype=np.float32)
            
            with self.data_lock:
                self.latest_joint_data[arm_name] = {
                    'values': joint_array.tolist(),
                    'timestamp': current_time
                }
                self.stats['joint_updates'] += 1
                
            # 保存到文件
            self._save_joint_data()
            
        except Exception as e:
            print(f"⚠️  关节数据回调错误 {arm_name}: {e}")

    def _image_callback(self, msg, cam_name):
        """图像数据回调函数 - 生成OpenPI推理兼容数据"""
        try:
            current_time = time.time()

            # 转换ROS图像到OpenCV格式 (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 转换为RGB格式 (OpenPI要求RGB)
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # 保存原始图像到images目录（只保存最新的）
            self._save_original_image(cam_name, img_rgb)

            # 调整大小到224x224 (OpenPI固定要求)
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            # 确保数据类型为uint8，数值范围0-255 (OpenPI兼容格式)
            img_openpi = img_resized.astype(np.uint8)

            # 验证OpenPI格式要求
            assert img_openpi.shape == (224, 224, 3), f"OpenPI格式错误: 尺寸 {img_openpi.shape}"
            assert img_openpi.dtype == np.uint8, f"OpenPI格式错误: 数据类型 {img_openpi.dtype}"
            assert 0 <= img_openpi.min() <= img_openpi.max() <= 255, f"OpenPI格式错误: 数值范围 [{img_openpi.min()}, {img_openpi.max()}]"

            with self.data_lock:
                self.latest_image_data[cam_name] = {
                    'data': img_openpi,
                    'timestamp': current_time
                }
                self.stats['image_updates'] += 1

            # 保存OpenPI推理格式的图像数据
            self._save_image_data(cam_name, img_openpi)

        except Exception as e:
            print(f"⚠️  图像数据回调错误 {cam_name}: {e}")
            import traceback
            traceback.print_exc()


    def _save_joint_data(self):
        """保存关节数据到文件"""
        try:
            with self.data_lock:
                if self.latest_joint_data:
                    with open(self.joint_file, 'w') as f:
                        json.dump(self.latest_joint_data, f, indent=2)
        except Exception as e:
            print(f"❌ 保存关节数据失败: {e}")

    def _save_original_image(self, cam_name, img_rgb):
        """保存原始图像到images目录（只保存最新的）"""
        try:
            # 保存为jpg格式，便于查看
            img_file = self.images_save_dir / f"{cam_name}_latest.jpg"
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_file), img_bgr)

            # 同时保存为npy格式
            npy_file = self.images_save_dir / f"{cam_name}_latest.npy"
            np.save(npy_file, img_rgb)

        except Exception as e:
            print(f"❌ 保存{cam_name}原始图像失败: {e}")

    def _save_image_data(self, cam_name, img_data):
        """保存处理后的图像数据到npy文件 - OpenPI推理格式"""
        try:
            # 确保数据完全符合OpenPI要求
            # 1. 尺寸必须是 224x224x3
            assert img_data.shape == (224, 224, 3), f"图像尺寸错误: {img_data.shape}, 期望 (224, 224, 3)"

            # 2. 数据类型必须是 uint8
            assert img_data.dtype == np.uint8, f"数据类型错误: {img_data.dtype}, 期望 uint8"

            # 3. 数值范围必须是 0-255
            assert img_data.min() >= 0 and img_data.max() <= 255, f"数值范围错误: [{img_data.min()}, {img_data.max()}], 期望 [0, 255]"

            # 4. 确保是RGB格式 (通过变量名确认，实际数据已经是RGB)
            # 保存为OpenPI推理可直接使用的格式
            img_file = self.image_dir / f"{cam_name}.npy"
            np.save(img_file, img_data)

            # 验证保存的文件
            if img_file.exists():
                # 快速验证保存的数据
                saved_data = np.load(img_file)
                assert saved_data.shape == (224, 224, 3), "保存验证失败：尺寸"
                assert saved_data.dtype == np.uint8, "保存验证失败：数据类型"

        except Exception as e:
            print(f"❌ 保存{cam_name}OpenPI格式图像失败: {e}")
            print(f"   图像信息: shape={img_data.shape}, dtype={img_data.dtype}, range=[{img_data.min()}, {img_data.max()}]")

    def get_stats(self):
        """获取统计信息"""
        with self.data_lock:
            runtime = time.time() - self.stats['start_time']
            return {
                'runtime': runtime,
                'joint_updates': self.stats['joint_updates'],
                'image_updates': self.stats['image_updates'],
                'joint_rate': self.stats['joint_updates'] / runtime if runtime > 0 else 0,
                'image_rate': self.stats['image_updates'] / runtime if runtime > 0 else 0,
                'mode': 'ROS_Real_Data',
                'joint_subscribers': len(self.joint_subscribers),
                'image_subscribers': len(self.image_subscribers)
            }

    def start(self):
        """启动服务"""
        self.running = True
        print("🌐 数据服务已启动")
        
        try:
            while self.running:
                # 打印统计信息
                stats = self.get_stats()
                print(f"📊 运行时间: {stats['runtime']:.1f}s")
                print(f"   🦾 关节更新: {stats['joint_updates']} ({stats['joint_rate']:.1f}Hz)")
                print(f"   📷 图像更新: {stats['image_updates']} ({stats['image_rate']:.1f}Hz)")
                print(f"   🔗 订阅者: 关节{stats['joint_subscribers']}个, 图像{stats['image_subscribers']}个")
                print(f"   🎯 模式: {stats['mode']}")

                # 显示最新数据状态
                with self.data_lock:
                    if self.latest_joint_data:
                        print(f"   📈 最新关节数据:")
                        for arm_name, data in self.latest_joint_data.items():
                            values = data.get('values', [])
                            timestamp = data.get('timestamp', 0)
                            age = time.time() - timestamp
                            print(f"      {arm_name}: {len(values)}关节, {age:.1f}s前更新")

                    if self.latest_image_data:
                        print(f"   🖼️  最新图像数据:")
                        for cam_name, img_info in self.latest_image_data.items():
                            img_data = img_info.get('data')
                            timestamp = img_info.get('timestamp', 0)
                            age = time.time() - timestamp
                            if img_data is not None:
                                print(f"      {cam_name}: {img_data.shape}, {age:.1f}s前更新")
                            else:
                                print(f"      {cam_name}: 无数据")

                print()
                time.sleep(5)  # 每5秒打印一次统计
                
        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
        finally:
            self.stop()

    def stop(self):
        """停止服务"""
        print("🛑 停止数据服务...")
        self.running = False
        print("✅ 数据服务已停止")


def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n🛑 接收到停止信号，正在关闭服务...")
    global service
    if 'service' in globals():
        service.stop()
    sys.exit(0)


def main():
    """主函数"""
    global service
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 启动简化数据服务...")
        print("📝 使用说明:")
        print("   - 数据保存在 ./data/ 目录")
        print("   - 关节数据: ./data/current_joints.json")
        print("   - OpenPI推理图像: ./data/images/*.npy (224x224, RGB, uint8)")
        print("   - 原始图像保存在 ./images/ 目录")
        print("   - 原始图像: ./images/*_latest.jpg (可视化)")
        print("   - 原始图像: ./images/*_latest.npy (原始尺寸)")
        print("   - 更新频率: 最大速度")
        print("   - OpenPI兼容: 完全符合推理格式要求")
        print("   - 按 Ctrl+C 停止服务")
        print()
        
        # 创建数据服务
        service = SimpleDataService(data_dir="./data")
        
        # 等待一下让数据开始更新
        print("⏳ 等待数据更新...")
        time.sleep(2)
        
        # 检查数据文件
        joint_file = Path("./data/current_joints.json")
        image_dir = Path("./data/images")
        images_save_dir = Path("./images")

        if joint_file.exists():
            print(f"✅ 关节数据文件已创建: {joint_file}")
        else:
            print(f"⚠️  关节数据文件未找到: {joint_file}")

        if image_dir.exists():
            image_files = list(image_dir.glob("*.npy"))
            print(f"✅ 处理后图像文件已创建: {len(image_files)} 个")
            for img_file in image_files:
                print(f"   📷 {img_file.name}")
        else:
            print(f"⚠️  处理后图像目录未找到: {image_dir}")

        if images_save_dir.exists():
            jpg_files = list(images_save_dir.glob("*.jpg"))
            npy_files = list(images_save_dir.glob("*.npy"))
            print(f"✅ 原始图像文件已创建: {len(jpg_files)} 个JPG, {len(npy_files)} 个NPY")
            for img_file in jpg_files:
                print(f"   �️  {img_file.name}")
        else:
            print(f"⚠️  原始图像目录未找到: {images_save_dir}")
        
        print()
        print("🌐 服务运行中...")
        
        # 启动服务 (会阻塞)
        service.start()
        
    except Exception as e:
        print(f"❌ 服务错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'service' in locals():
            service.stop()


if __name__ == '__main__':
    main()
