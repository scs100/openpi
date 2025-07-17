#!/usr/bin/env python3
"""
启动训练好的OpenPI模型推理服务器
"""
import os
import sys

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取项目根目录（假设根目录是当前文件的父目录的父目录）
project_root = os.path.dirname(os.path.dirname(current_file))
# 将根目录添加到 Python 路径
sys.path.append(project_root)

import logging
import dataclasses

# 设置环境变量
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# 导入OpenPI模块
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models import pi0
from openpi.serving import websocket_policy_server
import socket

# 设置日志
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class TrainedModelConfig:
    """训练模型配置"""
    checkpoint_dir: str
    config_name: str
    default_prompt: str
    dataset_name: str
    data_path: str
    port: int = 8000
    host: str = "0.0.0.0"

def create_trained_model_config(config_name: str, dataset_name: str, data_path: str, default_prompt: str):
    """创建与训练时相同的模型配置"""
    # 导入数据配置
    from lora_training.lora_train_config import CustomDataConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    config = _config.TrainConfig(
        name=config_name,

        # 模型配置 - 与训练时保持一致
        model=pi0.Pi0Config(
            action_dim=32,          # 必须保持32，与预训练模型匹配
            action_horizon=30,
            max_token_len=48,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora"
        ),

        # 数据配置 - 使用命令行传入的参数
        data=CustomDataConfig(
            data_path=data_path,
            default_prompt=default_prompt,
            dataset_name=f"{dataset_name}_del_step30",  # 修复：使用与训练时一致的norm stats路径
            use_delta_joint_actions=True  # 显式启用delta actions，与训练时保持一致
        ),

        # 权重加载器
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    )
    return config

def create_policy(model_config: TrainedModelConfig):
    """创建训练好的策略"""
    logger.info(f"正在加载训练好的模型...")
    logger.info(f"检查点目录: {model_config.checkpoint_dir}")
    logger.info(f"数据路径: {model_config.data_path}")
    logger.info(f"数据集名称: {model_config.dataset_name}")

    # 创建配置
    config = create_trained_model_config(
        model_config.config_name,
        model_config.dataset_name,
        model_config.data_path,
        model_config.default_prompt
    )

    # 创建训练好的策略
    policy = _policy_config.create_trained_policy(
        config,
        model_config.checkpoint_dir,
        default_prompt=model_config.default_prompt
    )

    logger.info("模型加载成功!")
    return policy
"""
使用示例:
# 所有主要参数都是必需的
conda activate openpi;
python inference_piper/serve_trained_model.py \
     --checkpoint_dir   checkpoints/lora_training/ae_lora_adamw_delta_actions_15w/50000 \
     --config_name lora_training \
     --default_prompt "pass me the drink" \
     --dataset_name "pass_drink" \
     --data_path "/home/testuser/data/pass_drink/openpi" \
     --port 8000 \
     --host 0.0.0.0

"""
def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动训练好的OpenPI模型推理服务器")
    parser.add_argument("--checkpoint_dir",
                        required=True,
                        help="检查点目录路径 (必需)")
    parser.add_argument("--config_name",
                        required=True,
                        help="配置名称 (必需)")
    parser.add_argument("--default_prompt",
                        required=True,
                        help="默认提示词 (必需)")
    parser.add_argument("--dataset_name",
                        required=True,
                        help="数据集基础名称 (如 'pass_drink')，程序会自动添加 '_del_step30' 后缀来匹配训练时的norm stats路径 (必需)")
    parser.add_argument("--data_path",
                        required=True,
                        help="数据集路径 (必需)")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")

    args = parser.parse_args()

    # 使用命令行参数创建配置
    model_config = TrainedModelConfig(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        default_prompt=args.default_prompt,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        port=args.port,
        host=args.host
    )

    try:
        # 创建策略
        policy = create_policy(model_config)
        
        # 获取策略元数据
        policy_metadata = policy.metadata
        
        # 获取主机信息
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"创建服务器 (主机: {hostname}, IP: {local_ip})")
        
        # 创建WebSocket服务器
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=model_config.host,
            port=model_config.port,
            metadata=policy_metadata,
        )
        
        logger.info("=" * 60)
        logger.info(f"🚀 OpenPI推理服务器已启动!")
        logger.info(f"📍 地址: {model_config.host}:{model_config.port}")
        logger.info(f"🤖 模型: {model_config.config_name}")
        logger.info(f"📂 检查点: {model_config.checkpoint_dir}")
        logger.info(f"💬 默认提示: {model_config.default_prompt}")
        logger.info("=" * 60)
        logger.info("服务器正在运行，按 Ctrl+C 停止...")
        
        # 启动服务器
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
