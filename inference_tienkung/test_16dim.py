#!/usr/bin/env python3
"""
测试16维数据处理的简单脚本
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_dimensions():
    """测试数据维度检测"""
    # 模拟16维数据
    test_data = {
        'action': [
            np.random.randn(16) for _ in range(10)  # 16维动作
        ],
        'observation.state': [
            np.random.randn(16) for _ in range(10)  # 16维状态
        ],
        'base_0_rgb': [
            {'bytes': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).tobytes()} 
            for _ in range(10)
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    # 检查维度
    if len(df) > 0:
        first_action = np.array(df['action'].iloc[0])
        first_state = np.array(df['observation.state'].iloc[0])
        actual_action_dim = len(first_action)
        actual_state_dim = len(first_state)
        
        logger.info(f"实际动作维度: {actual_action_dim}")
        logger.info(f"实际状态维度: {actual_state_dim}")
        
        # 使用实际维度，但不超过16维
        use_action_dim = min(actual_action_dim, 16)
        use_state_dim = min(actual_state_dim, 16)
        
        logger.info(f"使用动作维度: {use_action_dim}")
        logger.info(f"使用状态维度: {use_state_dim}")
        
        # 加载数据
        actions = np.array([np.array(action)[:use_action_dim] for action in df['action']])
        states = np.array([np.array(state)[:use_state_dim] for state in df['observation.state']])
        
        logger.info(f"动作数组形状: {actions.shape}")
        logger.info(f"状态数组形状: {states.shape}")
        
        # 测试关节名称
        joint_names = [
            'Left_J1', 'Left_J2', 'Left_J3', 'Left_J4', 'Left_J5', 'Left_J6', 'Left_J7', 'Left_Gripper',
            'Right_J1', 'Right_J2', 'Right_J3', 'Right_J4', 'Right_J5', 'Right_J6', 'Right_J7', 'Right_Gripper'
        ]
        
        logger.info(f"关节名称数量: {len(joint_names)}")
        for i, name in enumerate(joint_names):
            if i < use_action_dim:
                logger.info(f"关节 {i}: {name} = {actions[0, i]:.3f}")
        
        # 测试状态结构分析
        if use_state_dim >= 16:
            current_state = states[0]
            logger.info(f"状态结构分析:")
            logger.info(f"  左臂[0:7] = {current_state[:7]}")
            logger.info(f"  左夹爪[7] = {current_state[7]}")
            logger.info(f"  右臂[8:15] = {current_state[8:15]}")
            logger.info(f"  右夹爪[15] = {current_state[15]}")
        
        return True
    
    return False

def test_real_data():
    """测试真实数据文件"""
    data_dir = Path("/home/q/data/pick_up_parts_from_belt_conveyor_place_on_plate_fast_250702/openpi")
    
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return False
    
    # 查找parquet文件
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"未找到parquet文件: {data_dir}")
        return False
    
    # 读取第一个文件
    file_path = parquet_files[0]
    logger.info(f"读取文件: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"文件列名: {list(df.columns)}")
        
        if 'action' in df.columns and len(df) > 0:
            first_action = np.array(df['action'].iloc[0])
            logger.info(f"真实动作维度: {len(first_action)}")
            logger.info(f"动作前8维: {first_action[:8]}")
            if len(first_action) >= 16:
                logger.info(f"动作后8维: {first_action[8:16]}")
        
        if 'observation.state' in df.columns and len(df) > 0:
            first_state = np.array(df['observation.state'].iloc[0])
            logger.info(f"真实状态维度: {len(first_state)}")
            logger.info(f"状态前8维: {first_state[:8]}")
            if len(first_state) >= 16:
                logger.info(f"状态后8维: {first_state[8:16]}")
        
        return True
        
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("测试16维数据处理")
    logger.info("=" * 60)
    
    # 测试模拟数据
    logger.info("\n1. 测试模拟16维数据:")
    test_data_dimensions()
    
    # 测试真实数据
    logger.info("\n2. 测试真实数据文件:")
    test_real_data()
    
    logger.info("\n测试完成!")
