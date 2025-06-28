import time
import numpy as np
import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取项目根目录（假设根目录是当前文件的父目录的父目录）
project_root = os.path.dirname(os.path.dirname(current_file))
# 将根目录添加到 Python 路径
sys.path.append(project_root)

from xRocs.xrocs.utils.logger.logger_loader import logger
from xRocs.xrocs.core.station_loader import StationLoader
from xRocs.xrocs.common.data_type import Joints



class JointInference:
    def __init__(self, config_path = None, model_path = None):
        if config_path == None:
            config_path = "/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml"

        print(f"🔧 加载配置文件: {config_path}")

        # 直接加载 Agilex 配置文件
        try:
            import tomli as tomllib
        except ImportError:
            print("❌ 缺少 tomli 模块，请在 openpi 环境中运行")
            raise

        with open(config_path, 'rb') as f:
            self.agilex_config = tomllib.load(f)

        print(f"✅ 配置加载成功，机器人类型: {self.agilex_config['global_setting']['robot_type']}")

        # 获取 home 位置，用于 prepare 方法
        self.home_left = self.agilex_config['home_pose']['left']
        self.home_right = self.agilex_config['home_pose']['right']

        print(f"📍 左臂 home 位置: {self.home_left}")
        print(f"📍 右臂 home 位置: {self.home_right}")

        # 跳过 xROCS 工作站初始化，避免 ROS 节点冲突
        print("⚠️  跳过 xROCS 工作站初始化，使用直接 ROS 控制")
        self.robot_station = None  # 不使用 xROCS 工作站

        print("✅ JointInference 初始化完成!")

    def convert_agilex_config_to_xrocs_format(self, agilex_config):
        """将 Agilex 配置格式转换为 xROCS 标准格式"""
        xrocs_config = {
            'basic': {
                'station_type': agilex_config['global_setting']['robot_type']
            },
            'robot': {
                'arm': {
                    'home': {},
                    'ip': {}
                }
            },
            'camera': {}
        }

        # 转换 home 位置
        if 'home_pose' in agilex_config:
            for arm_name, home_joints in agilex_config['home_pose'].items():
                xrocs_config['robot']['arm']['home'][arm_name] = home_joints

        # 转换机器人 IP (如果有的话)
        if 'robot_ip' in agilex_config:
            for arm_name, ip_list in agilex_config['robot_ip'].items():
                if ip_list:  # 如果不是空列表
                    xrocs_config['robot']['arm']['ip'][arm_name] = ip_list[0] if isinstance(ip_list, list) else ip_list
                else:
                    # 如果没有 IP，使用默认值
                    xrocs_config['robot']['arm']['ip'][arm_name] = '127.0.0.1'

        # 转换相机配置
        if 'camera_dict' in agilex_config:
            for camera_name, camera_config in agilex_config['camera_dict'].items():
                if camera_config:  # 如果不是空列表
                    xrocs_config['camera'][camera_name] = {
                        'port': camera_config[0] if isinstance(camera_config, list) and camera_config else 4277,
                        'serial': [],
                        'freq': 15,
                        'size': [640, 480]
                    }

        return xrocs_config


    def prepare(self):
        """
        将机械臂移动到 home 位置
        使用直接 ROS 话题控制，已验证可以正常工作！
        """
        print("🏠 开始执行 prepare - 机械臂回到 home 位置...")

        try:
            import subprocess

            def run_command(cmd, timeout=10):
                """运行命令并返回结果"""
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
                    return result.returncode == 0, result.stdout, result.stderr
                except subprocess.TimeoutExpired:
                    return False, "", "命令超时"

            # 构建 rostopic pub 命令
            left_cmd = f"""rostopic pub /master/joint_left sensor_msgs/JointState "{{
header: {{stamp: now}},
name: ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
position: {self.home_left}
}}" -1"""

            right_cmd = f"""rostopic pub /master/joint_right sensor_msgs/JointState "{{
header: {{stamp: now}},
name: ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
position: {self.home_right}
}}" -1"""

            print("🎯 发送左臂 home 位置命令...")
            success, stdout, stderr = run_command(left_cmd)
            if success:
                print("✅ 左臂 home 命令发送成功")
            else:
                print(f"❌ 左臂 home 命令发送失败: {stderr}")
                raise Exception(f"左臂控制失败: {stderr}")

            print("🎯 发送右臂 home 位置命令...")
            success, stdout, stderr = run_command(right_cmd)
            if success:
                print("✅ 右臂 home 命令发送成功")
            else:
                print(f"❌ 右臂 home 命令发送失败: {stderr}")
                raise Exception(f"右臂控制失败: {stderr}")

            # 等待机器人移动
            print("⏳ 等待机器人移动到 home 位置...")
            time.sleep(3)

            logger.success('🎉 Resetting to home success!')
            print("🎉 Prepare 执行成功!")
            return True

        except Exception as e:
            print(f"❌ Prepare 执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def inference(self, data_dir: str, task_name: str):
        """
        推理方法 - 暂时注释掉，先专注于测试 prepare 方法
        """
        print("🤖 开始推理...")
        print(f"   数据目录: {data_dir}")
        print(f"   任务名称: {task_name}")

        obs = self.robot_station.get_obs()
        print(f"📊 获取到观测数据: {type(obs)}")

        # 暂时注释掉推理循环，避免无限循环
        # while True:
        #     action_pred: np.ndarray = self.inference_engine.infer(obs)
        #     action_pred = action_pred[0].cpu().numpy()
        #     robot_targets = self.robot_station.decompose_action(action_pred)
        #     obs = self.robot_station.step(robot_targets)
        #     time.sleep(0.05)

        print("ℹ️  推理方法当前被注释，只测试 prepare 功能")


if __name__ == '__main__':
    print("🚀 开始 Agilex 机械臂控制测试...")

    config_path = '/home/agilex/code/opensource/openpi/agilex_eggplant_config.toml'

    try:
        print("🔧 创建 JointInference 实例...")
        data = JointInference(config_path)

        print("🏠 测试 prepare 方法...")
        data.prepare()

        print("🎉 prepare 方法测试成功!")

        # 暂时不运行推理，只测试 prepare
        # data.inference('/home/eai/data/example_dir',"example_task")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
