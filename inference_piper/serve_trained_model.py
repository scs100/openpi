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
import argparse
from pathlib import Path

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
    checkpoint_dir: str = "checkpoints/sgd_swap_manager/sgd_swap_manager_norm/16000"
    config_name: str = "sgd_swap_manager"
    default_prompt: str = "pick and place purple long eggplant"
    port: int = 8000
    host: str = "0.0.0.0"

def create_trained_model_config(config_name: str = "sgd_swap_manager"):
    """创建与训练时相同的模型配置"""
    # 导入数据配置
    from pick_and_place_egglant.eggplant_train_config import EggplantDataConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    config = _config.TrainConfig(
        name=config_name,

        # 模型配置 - 与训练时保持一致
        model=pi0.Pi0Config(
            action_dim=32,          # 必须保持32，与预训练模型匹配
            action_horizon=50,
            max_token_len=48,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ),

        # 数据配置 - 使用训练时的norm stats
        data=EggplantDataConfig(
            data_path="/home/agilex/data/pick_and_place_eggplant/openpi",
            default_prompt="pick and place purple long eggplant",
        ),

        # 权重加载器
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    )
    return config

def create_policy(model_config: TrainedModelConfig):
    """创建训练好的策略"""
    logger.info(f"正在加载训练好的模型...")
    logger.info(f"检查点目录: {model_config.checkpoint_dir}")
    
    # 检查检查点是否存在
    checkpoint_path = Path(model_config.checkpoint_dir)
    if not checkpoint_path.exists():
        logger.error(f"检查点目录不存在: {model_config.checkpoint_dir}")
        
        # 列出可用的检查点
        base_dir = Path("checkpoints/sgd_swap_manager/sgd_swap_manager_norm")
        if base_dir.exists():
            logger.info("可用的检查点:")
            for item in sorted(base_dir.iterdir()):
                if item.is_dir() and item.name.isdigit():
                    logger.info(f"  - {item}")
        raise FileNotFoundError(f"检查点目录不存在: {model_config.checkpoint_dir}")
    
    # 创建配置
    config = create_trained_model_config(model_config.config_name)
    
    # 创建训练好的策略
    policy = _policy_config.create_trained_policy(
        config, 
        model_config.checkpoint_dir,
        default_prompt=model_config.default_prompt
    )
    
    logger.info("模型加载成功!")
    return policy
"""
conda activate openpi &&  python inference_piper/serve_trained_model.py --checkpoint_dir checkpoints/sgd_swap_manager/sgd_swap_manager_norm/16000 --config_name sgd_swap_manager

"""

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动训练好的OpenPI模型推理服务器")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="检查点目录路径"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="sgd_swap_manager",
        help="配置名称"
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default="pick and place purple long eggplant",
        help="默认提示词"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址"
    )
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 配置
    model_config = TrainedModelConfig()

    # 使用命令行参数更新配置
    if args.checkpoint_dir:
        model_config.checkpoint_dir = args.checkpoint_dir
        logger.info(f"使用指定的检查点: {args.checkpoint_dir}")
    else:
        # 只有在没有指定检查点时才自动查找最新的
        logger.info("未指定检查点，查找最新的检查点...")
        base_dir = Path("checkpoints/sgd_swap_manager/sgd_swap_manager_norm")
        if base_dir.exists():
            checkpoints = [item for item in sorted(base_dir.iterdir())
                          if item.is_dir() and item.name.isdigit()]
            if checkpoints:
                # 使用最新的检查点
                latest_checkpoint = checkpoints[-1]
                model_config.checkpoint_dir = str(latest_checkpoint)
                logger.info(f"使用最新检查点: {latest_checkpoint}")
            else:
                logger.error("未找到有效的检查点")
                return
        else:
            logger.error(f"检查点基础目录不存在: {base_dir}")
            return

    # 更新其他配置
    model_config.config_name = args.config_name
    model_config.default_prompt = args.default_prompt
    model_config.port = args.port
    model_config.host = args.host
    
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
