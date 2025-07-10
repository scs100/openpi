#!/usr/bin/env python3
"""
简化内存数据服务 - 纯内存模式，30Hz实时更新
在base环境下运行，获取ROS话题数据并保存在内存中
"""

import time
import numpy as np
import threading
import json
import cv2
import socket
import pickle
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
    print("⚠️  ROS未安装，请在base环境下运行")
    ROS_AVAILABLE = False
    sys.exit(1)


class SimpleMemoryService:
    """简化内存数据服务 - 纯内存模式"""
    
    def __init__(self):
        """初始化内存数据服务"""
        print("🚀 启动简化内存数据服务...")
        
        # 初始化ROS
        rospy.init_node('simple_memory_service', anonymous=True)
        self.bridge = CvBridge()
        
        # 内存数据存储（最新数据）
        self.latest_joint_data = {}
        self.latest_image_data = {}
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'joint_updates': 0,
            'image_updates': 0,
            'joint_updates_by_arm': {},  # 分别统计每个关节
            'image_updates_by_cam': {},  # 分别统计每个相机
        }

        # 时间戳历史记录（用于计算真实频率）
        self.joint_timestamps = {}  # 每个关节的时间戳历史
        self.image_timestamps = {}  # 每个相机的时间戳历史
        self.max_history = 100  # 保留最近100个时间戳

        # 频率控制（允许稍高频率来补偿处理延迟）
        self.target_hz = 35.0  # 稍微高一点，补偿处理延迟
        self.min_interval = 1.0 / self.target_hz  # 约28.6ms间隔
        self.last_joint_update = {}  # 每个关节的最后更新时间
        self.last_image_update = {}  # 每个相机的最后更新时间

        print(f"🎯 频率控制: 目标{self.target_hz}Hz, 最小间隔{self.min_interval*1000:.1f}ms")
        
        # 服务状态
        self.running = False

        # 调试保存功能（可选）
        self.debug_save = False  # 设置为False可关闭调试保存
        if self.debug_save:
            self.debug_dir = Path("debug_images")
            self.debug_dir.mkdir(exist_ok=True)
            print(f"🔍 调试模式: 图像将保存到 {self.debug_dir}")

        # 套接字服务器（高性能进程间通信）
        self.socket_server = True  # 启用套接字服务器
        self.socket_path = "/tmp/xrocs_memory_service.sock"
        self.server_socket = None
        self.client_connections = []
        if self.socket_server:
            print(f"🔌 套接字服务器: {self.socket_path}")

        # 数据导出功能（用于推理）
        self.export_for_inference = False  # 关闭文件导出，使用套接字
        if self.export_for_inference:
            self.export_dir = Path("memory_export")
            self.export_dir.mkdir(exist_ok=True)
            print(f"📤 推理导出: 数据将导出到 {self.export_dir}")

        # 设置ROS订阅者
        self._setup_subscribers()
        
        print("✅ 简化内存数据服务初始化完成")
        print("   模式: 纯内存（30Hz实时更新）")
        print("   存储: 最新数据覆盖旧数据")

    def _setup_subscribers(self):
        """设置ROS订阅者"""
        # 关节数据订阅
        joint_topics = {
            'left': '/puppet/joint_left',
            'right': '/puppet/joint_right'
        }
        
        self.joint_subscribers = {}
        for arm_name, topic in joint_topics.items():
            try:
                subscriber = rospy.Subscriber(
                    topic, 
                    JointState, 
                    lambda msg, name=arm_name: self._joint_callback(msg, name),
                    queue_size=1
                )
                self.joint_subscribers[arm_name] = subscriber
                print(f"✅ 关节订阅: {topic}")
            except Exception as e:
                print(f"❌ 关节订阅失败 {topic}: {e}")

        # 图像数据订阅
        image_topics = {
            'camera_f': '/camera_f/color/image_raw',
            'camera_l': '/camera_l/color/image_raw', 
            'camera_r': '/camera_r/color/image_raw'
        }
        
        self.image_subscribers = {}
        for cam_name, topic in image_topics.items():
            try:
                subscriber = rospy.Subscriber(
                    topic,
                    Image,
                    lambda msg, name=cam_name: self._image_callback(msg, name),
                    queue_size=1
                )
                self.image_subscribers[cam_name] = subscriber
                print(f"✅ 图像订阅: {topic}")
            except Exception as e:
                print(f"❌ 图像订阅失败 {topic}: {e}")

    def _joint_callback(self, msg, arm_name):
        """关节数据回调函数 - 30Hz频率限制"""
        try:
            current_time = time.time()

            # 频率限制（可选 - 当前禁用以测试自然频率）
            # if arm_name in self.last_joint_update:
            #     time_since_last = current_time - self.last_joint_update[arm_name]
            #     if time_since_last < self.min_interval:
            #         return  # 跳过此次更新

            joint_array = np.array(msg.position, dtype=np.float32)

            # 更新内存数据（覆盖旧数据）
            with self.data_lock:
                self.latest_joint_data[arm_name] = {
                    'values': joint_array.tolist(),
                    'timestamp': current_time
                }
                self.stats['joint_updates'] += 1

                # 分别统计每个关节
                if arm_name not in self.stats['joint_updates_by_arm']:
                    self.stats['joint_updates_by_arm'][arm_name] = 0
                self.stats['joint_updates_by_arm'][arm_name] += 1

                self.last_joint_update[arm_name] = current_time

                # 记录时间戳历史（用于计算真实频率）
                if arm_name not in self.joint_timestamps:
                    self.joint_timestamps[arm_name] = []
                self.joint_timestamps[arm_name].append(current_time)
                # 保持历史记录在限制范围内
                if len(self.joint_timestamps[arm_name]) > self.max_history:
                    self.joint_timestamps[arm_name] = self.joint_timestamps[arm_name][-self.max_history:]

                # 调试保存关节数据（每100次保存一次）
                if self.debug_save and self.stats['joint_updates_by_arm'][arm_name] % 100 == 0:
                    try:
                        # 计算基于时间戳的真实频率
                        real_freq = self.calculate_real_frequency(self.joint_timestamps[arm_name])

                        # 保存最新关节数据
                        joints_path = self.debug_dir / "latest_joints.json"
                        joints_data = {
                            'timestamp': current_time,
                            'joints': self.latest_joint_data.copy(),
                            'frequency_analysis': {
                                arm_name: {
                                    'count': self.stats['joint_updates_by_arm'][arm_name],
                                    'real_frequency_hz': real_freq,
                                    'timestamp_count': len(self.joint_timestamps[arm_name])
                                }
                            }
                        }
                        with open(joints_path, 'w') as f:
                            json.dump(joints_data, f, indent=2)

                        print(f"💾 {arm_name}关节: 第{self.stats['joint_updates_by_arm'][arm_name]}次, 真实频率: {real_freq:.1f}Hz")
                    except Exception as e:
                        print(f"⚠️  关节调试保存失败: {e}")

        except Exception as e:
            print(f"⚠️  关节数据回调错误 {arm_name}: {e}")

    def _image_callback(self, msg, cam_name):
        """图像数据回调函数 - 30Hz频率限制"""
        try:
            current_time = time.time()

            # 频率限制（可选 - 当前禁用以测试自然频率）
            # if cam_name in self.last_image_update:
            #     time_since_last = current_time - self.last_image_update[cam_name]
            #     if time_since_last < self.min_interval:
            #         return  # 跳过此次更新

            # 快速图像处理
            # 检查编码格式并正确处理
            if msg.encoding == "rgb8":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                img_rgb = cv_image  # 已经是RGB格式
            else:
                print(f"⚠️  {cam_name}: 不支持的编码格式 {msg.encoding}")
                return

            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_openpi = img_resized.astype(np.uint8)

            # 快速验证
            if img_openpi.shape != (224, 224, 3) or img_openpi.dtype != np.uint8:
                print(f"⚠️  {cam_name}格式错误: {img_openpi.shape}, {img_openpi.dtype}")
                return

            # 更新内存数据（覆盖旧数据）
            with self.data_lock:
                self.latest_image_data[cam_name] = {
                    'data': img_openpi,
                    'timestamp': current_time
                }
                self.stats['image_updates'] += 1

                # 分别统计每个相机
                if cam_name not in self.stats['image_updates_by_cam']:
                    self.stats['image_updates_by_cam'][cam_name] = 0
                self.stats['image_updates_by_cam'][cam_name] += 1

                self.last_image_update[cam_name] = current_time

                # 记录时间戳历史（用于计算真实频率）
                if cam_name not in self.image_timestamps:
                    self.image_timestamps[cam_name] = []
                self.image_timestamps[cam_name].append(current_time)
                # 保持历史记录在限制范围内
                if len(self.image_timestamps[cam_name]) > self.max_history:
                    self.image_timestamps[cam_name] = self.image_timestamps[cam_name][-self.max_history:]

                # 调试保存（每10次保存一次，避免过多文件）
                if self.debug_save and self.stats['image_updates_by_cam'][cam_name] % 10 == 0:
                    try:
                        # 保存最新图像
                        jpg_path = self.debug_dir / f"{cam_name}_latest.jpg"
                        npy_path = self.debug_dir / f"{cam_name}_latest.npy"

                        # 保存JPG（用于查看）
                        img_bgr = cv2.cvtColor(img_openpi, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(jpg_path), img_bgr)

                        # 保存NPY（用于模型）
                        np.save(str(npy_path), img_openpi)

                        # 保存带时间戳的版本（每50次）
                        if self.stats['image_updates_by_cam'][cam_name] % 50 == 0:
                            # 计算真实频率
                            real_freq = self.calculate_real_frequency(self.image_timestamps[cam_name])

                            timestamp_str = f"{int(current_time)}_{self.stats['image_updates_by_cam'][cam_name]:04d}"
                            jpg_ts_path = self.debug_dir / f"{cam_name}_{timestamp_str}.jpg"
                            npy_ts_path = self.debug_dir / f"{cam_name}_{timestamp_str}.npy"
                            cv2.imwrite(str(jpg_ts_path), img_bgr)
                            np.save(str(npy_ts_path), img_openpi)
                            print(f"💾 {cam_name}: 第{self.stats['image_updates_by_cam'][cam_name]}次, 真实频率: {real_freq:.1f}Hz")

                    except Exception as e:
                        print(f"⚠️  {cam_name}调试保存失败: {e}")

        except Exception as e:
            print(f"⚠️  图像数据回调错误 {cam_name}: {e}")

    def get_latest_joint_data(self):
        """获取最新关节数据"""
        with self.data_lock:
            return self.latest_joint_data.copy()

    def get_latest_image_data(self):
        """获取最新图像数据"""
        with self.data_lock:
            return self.latest_image_data.copy()

    def calculate_real_frequency(self, timestamps):
        """根据时间戳计算真实频率"""
        if len(timestamps) < 2:
            return 0.0

        # 使用最近的时间戳计算平均间隔
        recent_timestamps = timestamps[-min(50, len(timestamps)):]  # 最近50个
        if len(recent_timestamps) < 2:
            return 0.0

        # 计算时间间隔
        intervals = []
        for i in range(1, len(recent_timestamps)):
            interval = recent_timestamps[i] - recent_timestamps[i-1]
            if interval > 0:  # 避免除零错误
                intervals.append(interval)

        if not intervals:
            return 0.0

        # 计算平均间隔和频率
        avg_interval = sum(intervals) / len(intervals)
        frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0

        return frequency

    def export_data_for_inference(self):
        """导出数据供推理使用"""
        if not self.export_for_inference:
            return

        try:
            current_time = time.time()

            with self.data_lock:
                # 导出关节数据
                joint_export = {
                    'data': self.latest_joint_data.copy(),
                    'export_time': current_time
                }
                joint_file = self.export_dir / "latest_joints.json"
                with open(joint_file, 'w') as f:
                    json.dump(joint_export, f)

                # 导出图像数据（NPY格式，供推理使用）
                for cam_name, img_info in self.latest_image_data.items():
                    img_data = img_info.get('data')
                    if img_data is not None:
                        # 保存NPY文件（224x224x3 RGB uint8）
                        npy_path = self.export_dir / f"{cam_name}_latest.npy"
                        np.save(str(npy_path), img_data)

        except Exception as e:
            print(f"⚠️  数据导出失败: {e}")

    def start_socket_server(self):
        """启动套接字服务器"""
        if not self.socket_server:
            return

        try:
            # 删除已存在的套接字文件
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            # 创建Unix域套接字
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)

            print(f"🔌 套接字服务器已启动: {self.socket_path}")

            # 启动服务器线程
            server_thread = threading.Thread(target=self._socket_server_loop, daemon=True)
            server_thread.start()

        except Exception as e:
            print(f"❌ 套接字服务器启动失败: {e}")

    def _socket_server_loop(self):
        """套接字服务器主循环"""
        while self.running and self.server_socket:
            try:
                client_socket, _ = self.server_socket.accept()
                print("🔌 新客户端连接")

                # 启动客户端处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()

            except Exception as e:
                if self.running:
                    print(f"⚠️  套接字服务器错误: {e}")
                break

    def _handle_client(self, client_socket):
        """处理客户端请求"""
        try:
            while self.running:
                # 接收请求
                request = client_socket.recv(1024)
                if not request:
                    break

                if request == b"GET_LATEST_DATA":
                    # 准备数据
                    with self.data_lock:
                        data = {
                            'joints': self.latest_joint_data.copy(),
                            'images': self.latest_image_data.copy()
                        }

                    # 序列化数据
                    data_bytes = pickle.dumps(data)
                    data_length = len(data_bytes)

                    # 发送数据长度（8字节）
                    length_bytes = data_length.to_bytes(8, byteorder='big')
                    client_socket.sendall(length_bytes)

                    print(f"📤 发送数据: {data_length} 字节")

                    # 发送实际数据（确保完整发送）
                    sent = 0
                    while sent < data_length:
                        chunk_size = min(8192, data_length - sent)
                        chunk = data_bytes[sent:sent + chunk_size]
                        bytes_sent = client_socket.send(chunk)
                        sent += bytes_sent

                        # 显示进度
                        if sent % 50000 == 0 or sent == data_length:
                            progress = (sent / data_length) * 100
                            print(f"📤 发送进度: {sent}/{data_length} ({progress:.1f}%)")

                    print(f"✅ 数据发送完成: {sent} 字节")

        except Exception as e:
            print(f"⚠️  客户端处理错误: {e}")
        finally:
            client_socket.close()
            print("🔌 客户端断开连接")

    def stop_socket_server(self):
        """停止套接字服务器"""
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

        # 删除套接字文件
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        print("🔌 套接字服务器已停止")

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
                'mode': 'Pure_Memory_30Hz',
                'joint_subscribers': len(self.joint_subscribers),
                'image_subscribers': len(self.image_subscribers)
            }


    def start(self):
        """启动服务"""
        self.running = True

        # 启动套接字服务器
        self.start_socket_server()

        print("🌐 简化内存服务已启动")
        print("   ⚡ 纯内存模式: 最大性能")
        
        try:
            export_counter = 0
            while self.running:
                time.sleep(5)  # 每5秒显示一次统计
                export_counter += 1
                
                stats = self.get_stats()
                print(f"\n📊 内存服务运行时间: {stats['runtime']:.1f}s")
                print(f"   🦾 关节更新总计: {stats['joint_updates']} ({stats['joint_rate']:.1f}Hz) [自然频率]")
                print(f"   📷 图像更新总计: {stats['image_updates']} ({stats['image_rate']:.1f}Hz) [自然频率]")

                # 显示每个设备的详细频率
                with self.data_lock:
                    runtime = stats['runtime']
                    if self.stats['joint_updates_by_arm']:
                        print(f"   🦾 关节详细:")
                        for arm_name, count in self.stats['joint_updates_by_arm'].items():
                            rate = count / runtime if runtime > 0 else 0
                            # 计算基于时间戳的真实频率
                            real_freq = 0.0
                            if arm_name in self.joint_timestamps:
                                real_freq = self.calculate_real_frequency(self.joint_timestamps[arm_name])
                            print(f"      {arm_name}: {count} (统计:{rate:.1f}Hz, 真实:{real_freq:.1f}Hz)")

                    if self.stats['image_updates_by_cam']:
                        print(f"   📷 图像详细:")
                        for cam_name, count in self.stats['image_updates_by_cam'].items():
                            rate = count / runtime if runtime > 0 else 0
                            # 计算基于时间戳的真实频率
                            real_freq = 0.0
                            if cam_name in self.image_timestamps:
                                real_freq = self.calculate_real_frequency(self.image_timestamps[cam_name])
                            print(f"      {cam_name}: {count} (统计:{rate:.1f}Hz, 真实:{real_freq:.1f}Hz)")

                print(f"   🎯 模式: {stats['mode']}")
                print(f"   📡 订阅者: 关节{stats['joint_subscribers']}个, 图像{stats['image_subscribers']}个")

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
                
        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
        finally:
            self.stop()

    def stop(self):
        """停止服务"""
        print("🛑 停止简化内存数据服务...")
        self.running = False

        # 停止套接字服务器
        self.stop_socket_server()

        print("✅ 简化内存数据服务已停止")


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
    
    print("🚀 启动简化内存数据服务")
    print("=" * 50)
    print("🔧 配置:")
    print("   模式: 纯内存（无文件保存）")
    print("   频率: ~30Hz（跟随ROS话题）")
    print("   存储: 最新数据覆盖旧数据")
    print("   性能: 最大化（无I/O瓶颈）")
    print()
    
    try:
        # 创建简化内存数据服务
        service = SimpleMemoryService()
        
        # 等待一下让数据开始更新
        print("⏳ 等待数据更新...")
        time.sleep(2)
        
        print("⚡ 纯内存模式: 无文件保存，最大性能")
        print()
        print("🌐 内存服务运行中...")
        
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
