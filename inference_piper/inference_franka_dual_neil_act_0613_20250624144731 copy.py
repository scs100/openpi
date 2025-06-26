import time
import numpy as np
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader

from pathlib import Path
import torch
import numpy as np
import cv2
import time
# from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy as Policy
from xrocs.common.data_type import Joints
# from xlinker.agent.linker_agent import LinkerAgent


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




class JointInference:
    def __init__(self, config_path = None, model_path = None):
        if config_path == None:
            config_path = "/home/eai/Documents/configuration.toml"
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        # 初始化推理类
        model_path = "/home/eai/workspace/neil/models/act_franka_dr3_cross_put_the_corn_into_the_pot/checkpoints/200000/pretrained_model"
        # model_path = "/home/eai/workspace/neil/models/020000/pretrained_model"
        self.inference_engine = PolicyInference(model_path)
        print("@@@@@@@@@@@")


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
