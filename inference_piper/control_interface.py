#!/usr/bin/env python3
"""
控制接口 - 用于OpenPI推理服务和xROCS控制服务之间的通信
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class ControlInterface:
    """控制接口 - 处理推理服务和控制服务之间的通信"""

    def __init__(self, interface_dir: str = "./control_interface"):
        self.interface_dir = Path(interface_dir)
        self.interface_dir.mkdir(exist_ok=True)

        # 文件路径
        self.command_file = self.interface_dir / "command.json"
        self.status_file = self.interface_dir / "status.json"
        self.feedback_file = self.interface_dir / "feedback.json"

        # 锁文件，防止并发访问
        self.command_lock = self.interface_dir / "command.lock"
        self.status_lock = self.interface_dir / "status.lock"
        self.feedback_lock = self.interface_dir / "feedback.lock"

        # 获取当前命令ID，避免冲突
        self.command_id = self._get_current_command_id()

        print(f"🔗 控制接口初始化: {self.interface_dir}")
        print(f"🔢 起始命令ID: {self.command_id}")

    def _get_current_command_id(self) -> int:
        """获取当前命令ID，避免与正在运行的命令冲突"""
        try:
            max_id = 0

            # 读取当前命令文件
            current_command = self._safe_read_json(self.command_file, self.command_lock)
            if current_command and "command_id" in current_command:
                max_id = max(max_id, current_command["command_id"])

            # 读取反馈文件
            feedback = self._safe_read_json(self.feedback_file, self.feedback_lock)
            if feedback and "command_id" in feedback:
                max_id = max(max_id, feedback["command_id"])

            # 返回下一个可用的ID
            return max_id
        except Exception as e:
            print(f"⚠️  获取命令ID失败: {e}")
            return 0

    def _safe_write_json(self, file_path: Path, data: Dict, lock_path: Path):
        """安全写入JSON文件"""
        try:
            # 创建锁文件
            lock_path.touch()
            
            # 写入临时文件然后重命名（原子操作）
            temp_file = file_path.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            
            temp_file.rename(file_path)
            
        finally:
            # 删除锁文件
            if lock_path.exists():
                lock_path.unlink()
    
    def _safe_read_json(self, file_path: Path, lock_path: Path) -> Optional[Dict]:
        """安全读取JSON文件"""
        # 等待锁文件释放
        max_wait = 50  # 最多等待50ms
        wait_count = 0
        while lock_path.exists() and wait_count < max_wait:
            time.sleep(0.001)  # 等待1ms
            wait_count += 1
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def _json_serializer(self, obj):
        """JSON序列化器，处理numpy数组"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class InferenceControlClient(ControlInterface):
    """推理服务端的控制客户端"""

    def __init__(self, interface_dir: str = "./control_interface"):
        super().__init__(interface_dir)
        # 不要重置command_id，使用父类中正确获取的值
        print("🤖 推理控制客户端初始化")
        print(f"🔢 使用命令ID: {self.command_id}")
    
    def send_joint_command(self, joint_angles: List[float], arm: str = "right") -> int:
        """
        发送关节控制命令

        参数:
        - joint_angles: 关节角度列表
        - arm: 机械臂名称 ("left" 或 "right")

        返回:
        - command_id: 命令ID
        """
        self.command_id += 1
        
        command = {
            "command_id": self.command_id,
            "type": "joint_control",
            "arm": arm,
            "joint_angles": joint_angles,
            "timestamp": time.time()
        }
        
        self._safe_write_json(self.command_file, command, self.command_lock)
        print(f"📤 发送关节命令 {self.command_id}: {arm} {joint_angles[:3]}...")
        
        return self.command_id
    
    def send_home_command(self, arm: str = "right") -> int:
        """发送回home命令"""
        self.command_id += 1
        
        command = {
            "command_id": self.command_id,
            "type": "home",
            "arm": arm,
            "timestamp": time.time()
        }
        
        self._safe_write_json(self.command_file, command, self.command_lock)
        print(f"🏠 发送home命令 {self.command_id}: {arm}")
        
        return self.command_id
    
    def get_status(self) -> Optional[Dict]:
        """获取控制服务状态"""
        return self._safe_read_json(self.status_file, self.status_lock)
    
    def get_feedback(self) -> Optional[Dict]:
        """获取执行反馈"""
        return self._safe_read_json(self.feedback_file, self.feedback_lock)

    def set_motion_parameters(self, speed_limit: float = None, acceleration_limit: float = None, jerk_limit: float = None) -> int:
        """
        设置运动参数

        参数:
        - speed_limit: 速度限制 (rad/s)
        - acceleration_limit: 加速度限制 (rad/s²)
        - jerk_limit: 抖动限制 (rad/s³)

        返回:
        - command_id: 命令ID
        """
        self.command_id += 1

        command = {
            "command_id": self.command_id,
            "type": "set_motion_parameters",
            "speed_limit": speed_limit,
            "acceleration_limit": acceleration_limit,
            "jerk_limit": jerk_limit,
            "timestamp": time.time()
        }

        self._safe_write_json(self.command_file, command, self.command_lock)
        print(f"⚙️ 发送运动参数设置命令 {self.command_id}")
        if speed_limit is not None:
            print(f"   - 速度限制: {speed_limit} rad/s")
        if acceleration_limit is not None:
            print(f"   - 加速度限制: {acceleration_limit} rad/s²")
        if jerk_limit is not None:
            print(f"   - 抖动限制: {jerk_limit} rad/s³")

        return self.command_id
    
    def wait_for_completion(self, command_id: int, timeout: float = 3.0) -> bool:
        """
        等待命令执行完成
        
        参数:
        - command_id: 命令ID
        - timeout: 超时时间(秒)
        
        返回:
        - 是否成功完成
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            feedback = self.get_feedback()
            if feedback and feedback.get("command_id") == command_id:
                if feedback.get("status") == "completed":
                    return True
                elif feedback.get("status") == "failed":
                    print(f"❌ 命令 {command_id} 执行失败: {feedback.get('error', 'Unknown error')}")
                    return False
            
            time.sleep(0.05)  # 50ms检查一次
        
        print(f"⏰ 命令 {command_id} 执行超时")
        return False


class ControlServiceServer(ControlInterface):
    """控制服务端的服务器"""

    def __init__(self, interface_dir: str = "./control_interface"):
        super().__init__(interface_dir)
        self.running = False
        # 使用当前命令ID作为起始点，避免重复处理
        self.last_command_id = self.command_id
        print("🎛️  控制服务器初始化")
        print(f"🔢 服务器起始命令ID: {self.last_command_id}")
    
    def update_status(self, status: str, details: Dict = None):
        """更新服务状态"""
        status_data = {
            "status": status,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        self._safe_write_json(self.status_file, status_data, self.status_lock)
    
    def send_feedback(self, command_id: int, status: str, error: str = None, details: Dict = None):
        """发送执行反馈"""
        feedback = {
            "command_id": command_id,
            "status": status,  # "executing", "completed", "failed"
            "timestamp": time.time(),
            "error": error,
            "details": details or {}
        }
        
        self._safe_write_json(self.feedback_file, feedback, self.feedback_lock)
    
    def get_command(self) -> Optional[Dict]:
        """获取新的控制命令"""
        command = self._safe_read_json(self.command_file, self.command_lock)

        if command and command.get("command_id", 0) > self.last_command_id:
            self.last_command_id = command["command_id"]
            print(f"🔍 处理命令 {command['command_id']}: {command.get('type', 'unknown')}")
            return command

        return None
    
    def start_monitoring(self, command_handler):
        """开始监控命令"""
        self.running = True
        self.update_status("ready")
        
        print("🔄 开始监控控制命令...")
        
        while self.running:
            try:
                command = self.get_command()
                if command:
                    print(f"📨 收到命令 {command['command_id']}: {command['type']}")
                    self.send_feedback(command["command_id"], "executing")
                    
                    try:
                        # 执行命令
                        success = command_handler(command)
                        
                        if success:
                            self.send_feedback(command["command_id"], "completed")
                        else:
                            self.send_feedback(command["command_id"], "failed", "Execution failed")
                            
                    except Exception as e:
                        print(f"❌ 命令执行异常: {e}")
                        self.send_feedback(command["command_id"], "failed", str(e))
                
                time.sleep(0.01)  # 10ms检查一次
                
            except KeyboardInterrupt:
                print("⏹️  用户中断监控")
                break
            except Exception as e:
                print(f"❌ 监控异常: {e}")
                time.sleep(0.1)
        
        self.update_status("stopped")
        print("🛑 控制命令监控已停止")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试控制接口...")
    
    # 创建客户端
    client = InferenceControlClient()
    
    # 发送测试命令
    cmd_id = client.send_joint_command([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    # 检查状态
    status = client.get_status()
    print(f"状态: {status}")
    
    print("✅ 控制接口测试完成")
