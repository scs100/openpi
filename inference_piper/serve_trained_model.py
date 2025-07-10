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
    checkpoint_dir: str = "checkpoints/lora_training/rgb_lora_sgd_lr1e-4/16000"
    config_name: str = "lora_training"
    default_prompt: str = "pick the long eggplant and place on the plant"
    port: int = 8000
    host: str = "0.0.0.0"

def create_trained_model_config(config_name: str = "lora_training"):
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
            data_path="/home/agilex/data/pick_and_place_eggplant/openpi_33fps",
            default_prompt="pick then long eggplant and place on the plant",
        ),

        # 权重加载器
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    )
    return config

def create_policy(model_config: TrainedModelConfig):
    """创建训练好的策略"""
    logger.info(f"正在加载训练好的模型...")
    logger.info(f"检查点目录: {model_config.checkpoint_dir}")

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

使用示例:
conda activate openpi;
python inference_piper/serve_trained_model.py \
     --checkpoint_dir checkpoints/lora_training/lora_sgd_bat6_10w_lr1e-41e5/40000 \
     --config_name lora_training \
     --default_prompt "pick the long eggplant and place on the plant" \
     --port 8000 \
     --host 0.0.0.0
"""
def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动训练好的OpenPI模型推理服务器")
    parser.add_argument("--checkpoint_dir",
                        default="checkpoints/lora_training/rgb_lora_sgd_lr1e-4/16000",
                        help="检查点目录路径")
    parser.add_argument("--config_name",
                        default="lora_training",
                        help="配置名称")
    parser.add_argument("--default_prompt",
                        default="pick then long eggplant and place on the plant",
                        help="默认提示词")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")

    args = parser.parse_args()

    # 使用命令行参数创建配置
    model_config = TrainedModelConfig(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        default_prompt=args.default_prompt,
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
