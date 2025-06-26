import time
import numpy as np
from pathlib import Path
import torch
import cv2

# xrocs imports with error handling
try:
    from xrocs.common.data_type import Joints
    from xrocs.utils.logger.logger_loader import logger
    from xrocs.core.station_loader import StationLoader
    XROCS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: xrocs modules not fully available: {e}")
    XROCS_AVAILABLE = False
    # Create dummy classes for development/testing
    class Joints:
        def __init__(self, *args, **kwargs):
            pass
    logger = None

# ConfigLoader with flexible path handling
try:
    from xrocs.core.config_loader import ConfigLoader
    CONFIG_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ConfigLoader not available: {e}")
    CONFIG_LOADER_AVAILABLE = False
    ConfigLoader = None

# lerobot imports
try:
    from lerobot.common.policies.act.modeling_act import ACTPolicy as Policy
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: lerobot not available: {e}")
    LEROBOT_AVAILABLE = False
    Policy = None

# Optional xlinker import
try:
    from xlinker.agent.linker_agent import LinkerAgent
    XLINKER_AVAILABLE = True
except ImportError:
    XLINKER_AVAILABLE = False
    LinkerAgent = None


class PolicyInference:
    def __init__(self, model_path):
        """ 

        参数：
        model_path (str): 预训练模型的路径。
        """
        self.model_path = Path(model_path)
        self.device = self._get_device()
        self.policy = self._load_policy()
        self.cnt = 0

    def _get_device(self):
        """
        获取可用设备（GPU 或 CPU）。

        返回：
        device (torch.device): 模型运行的设备。
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Device set to:", device)
        else:
            device = torch.device("cpu")
            print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        return device

    def _load_policy(self):
        """
        加载预训练模型并将其移动到指定设备。

        返回：
        policy (DiffusionPolicy): 加载后的模型。
        """
        print(f"Loading model from: {self.model_path}")
        policy = Policy.from_pretrained(self.model_path)
        policy.eval()
        policy.to(self.device)
        return policy

    def generate_obs(self):
        """
        生成模拟观测数据。

        返回：
        obs (dict): 包含图像和位置信息的观测数据。
        """
        obs = {
            'images': {
                'left': cv2.imencode('.jpg', np.random.randn(480, 640, 3))[1],
                'right': cv2.imencode('.jpg', np.random.randn(480, 640, 3))[1],
                'top': cv2.imencode('.jpg', np.random.randn(480, 640, 3))[1],
            },
            'qpos': np.random.randn(8,),
        }
        return obs

    def reset(self):
        self.policy.reset() 
        
    def prepare_inference_obs(self, obs):
        """
        准备用于推理的观测数据。

        参数：
        obs (dict): 包含图像和位置信息的观测数据。

        返回：
        inference_data (dict): 处理后的观测数据，可用于推理。
        """
        inference_data = {}
        camera_names_1 = ['front','left']
        camera_names_2 = ['top','right']
        # 处理图像数据
        for cam_name in camera_names_1:
            cam_img = obs['images'][cam_name]
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            cam_img = cv2.resize(cam_img, dsize=(320, 180))

            # cv2.imwrite(f"vis/{cam_name}_{self.cnt}.png", cam_img)

            cam_img_tensor = torch.from_numpy(cam_img).permute(2, 0, 1).float() / 255.
            # / 255.0
            inference_data[f'observation.images.camera_{cam_name}'] = cam_img_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        for cam_name in camera_names_2:
            cam_img = obs['images'][cam_name]
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            cam_img = cv2.resize(cam_img, dsize=(320, 240))

            # cv2.imwrite(f"vis/{cam_name}_{self.cnt}.png", cam_img)

            cam_img_tensor = torch.from_numpy(cam_img).permute(2, 0, 1).float() / 255.
            # / 255.0
            inference_data[f'observation.images.camera_{cam_name}'] = cam_img_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        self.cnt += 1

        # 处理位置信息

        qpos = np.concatenate([obs['arm_joints']['left'], obs['arm_joints']['right']])
        qpos_data = torch.from_numpy(qpos).float()
        inference_data['observation.state'] = qpos_data.unsqueeze(0).to(self.device, non_blocking=True)

        return inference_data

    def infer(self, obs=None):
        """
        使用模型进行推理。

        参数：
        obs (dict, optional): 观测数据。如果未提供，则生成模拟数据。

        返回：
        output_dict (dict): 模型的输出结果。
        """
        if obs is None:
            obs = self.generate_obs()
        input_data = self.prepare_inference_obs(obs)
        output_dict = self.policy.select_action(input_data)
        return output_dict




def safe_config_loader(config_path=None):
    """
    安全的配置加载器，支持多种配置文件路径
    """
    if not CONFIG_LOADER_AVAILABLE:
        print("Warning: ConfigLoader not available, returning empty config")
        return {}

    # 尝试多个可能的配置文件路径
    possible_paths = [
        config_path,
        "/home/testuser/Documents/configuration.toml",
        "/home/eai/Documents/configuration.toml",
        "./configuration.toml",
        "../configuration.toml"
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            try:
                cfg_loader = ConfigLoader(path)
                return cfg_loader.get_config()
            except Exception as e:
                print(f"Failed to load config from {path}: {e}")
                continue

    print("Warning: No valid configuration file found, returning empty config")
    return {}


class DataCollector:
    """
    模块化的数据采集器
    """
    def __init__(self, cfg_dict=None):
        self.cfg_dict = cfg_dict or {}
        self.robot_station = None

        if XROCS_AVAILABLE and cfg_dict:
            try:
                station_loader = StationLoader(self.cfg_dict)
                self.robot_station = station_loader.generate_station_handle()
                self.robot_station.connect()
                print("Robot station connected successfully")
            except Exception as e:
                print(f"Failed to initialize robot station: {e}")
                self.robot_station = None
        else:
            print("Warning: xrocs not available or no config provided, running in simulation mode")

    def prepare(self):
        """准备机器人到初始位置"""
        if not self.robot_station:
            print("Warning: No robot station available, skipping prepare")
            return

        try:
            for name, _robot in self.robot_station.get_robot_handle().items():
                home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                              num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
                _robot.reach_target_joint(home)
            for gripper in self.robot_station.get_gripper_handle().values():
                gripper.open()
            time.sleep(2)
            if logger:
                logger.success('Resetting to home success!')
            else:
                print('Resetting to home success!')
        except Exception as e:
            print(f"Error during prepare: {e}")

    def start_collect(self, data_dir, tasks):
        """
        开始数据采集

        参数:
        data_dir (str): 数据保存目录
        tasks (list): 任务列表，每个任务包含name和description
        """
        print(f"Starting data collection in: {data_dir}")
        print(f"Tasks to collect: {len(tasks)}")

        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}: {task['name']}")
            print(f"Description: {task['description']}")

            if self.robot_station:
                # 实际的数据采集逻辑
                self._collect_task_data(data_dir, task)
            else:
                # 模拟模式
                print("Simulation mode: would collect data for this task")
                time.sleep(1)  # 模拟采集时间

    def _collect_task_data(self, data_dir, task):
        """采集单个任务的数据"""
        try:
            # 这里实现具体的数据采集逻辑
            obs = self.robot_station.get_obs()
            # 保存数据的逻辑...
            print(f"Collected data for task: {task['name']}")
        except Exception as e:
            print(f"Error collecting data for task {task['name']}: {e}")


class JointInference:
    def __init__(self, config_path=None, model_path=None):
        # 使用安全的配置加载器
        self.cfg_dict = safe_config_loader(config_path)

        # 初始化机器人站点
        if XROCS_AVAILABLE and self.cfg_dict:
            try:
                station_loader = StationLoader(self.cfg_dict)
                self.robot_station = station_loader.generate_station_handle()
                self.robot_station.connect()
                print("Robot station initialized successfully")
            except Exception as e:
                print(f"Failed to initialize robot station: {e}")
                self.robot_station = None
        else:
            print("Warning: Running without robot station")
            self.robot_station = None

        # 初始化推理引擎
        if model_path is None:
            model_path = "/home/eai/workspace/neil/models/act_franka_dr3_cross_put_the_corn_into_the_pot/checkpoints/200000/pretrained_model"

        if LEROBOT_AVAILABLE:
            try:
                self.inference_engine = PolicyInference(model_path)
                print("Policy inference engine initialized successfully")
            except Exception as e:
                print(f"Failed to initialize inference engine: {e}")
                self.inference_engine = None
        else:
            print("Warning: lerobot not available, inference engine not initialized")
            self.inference_engine = None


    def prepare(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                          num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
        logger.success('Resetting to home success!')

    def inference(self, data_dir: str, task_name: str):
        print("11111111")
        obs = self.robot_station.get_obs()
        while True:
            action_pred: np.ndarray = self.inference_engine.infer(obs)
            action_pred = action_pred[0].cpu().numpy()

            robot_targets = self.robot_station.decompose_action(action_pred)
            obs = self.robot_station.step(robot_targets)
            time.sleep(0.05)


if __name__ == '__main__':
    config_path = '/home/eai/Documents/configuration.toml'
    data = JointInference(config_path)
    data.prepare()
    data.inference('/home/eai/data/example_dir',"example_task")
