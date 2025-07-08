#!/usr/bin/env python3
"""
OpenPI Full Episode Inference Visualization
Complete episode inference with all timesteps
"""

import os
import time
import sys
from PIL import Image
import io

# 检查依赖项
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import logging
    from openpi_client import websocket_client_policy
    import argparse
    import traceback
except ImportError as e:
    print(f"缺少必要的依赖项: {e}")
    print("请确保已安装所有依赖项，可通过以下命令安装：")
    print("pip install numpy pandas matplotlib seaborn pillow")
    sys.exit(1)

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array * 2.0 - 1.0  # 归一化到[-1,1]

class FullEpisodeInferenceVisualizer:
    """Full episode inference visualizer"""
    
    def __init__(self, host="localhost", port=8000, data_path="/home/q/data/pick_and_place_eggplant/openpi", debug=False, prediction_step=0):
        self.host = host
        self.port = port
        self.data_path = Path(data_path)
        self.policy = None
        self.gt_data = None
        self.debug = debug
        # 统一控制选择哪一步预测进行对比和可视化
        # 0: 第一步, -1: 最后一步, 其他: 指定步数
        self.prediction_step = prediction_step
        # Joint names for 16-dimensional actions (左臂7个+左夹爪1个+右臂7个+右夹爪1个)
        self.joint_names = [
            'Left_J1', 'Left_J2', 'Left_J3', 'Left_J4', 'Left_J5', 'Left_J6', 'Left_J7', 'Left_Gripper',
            'Right_J1', 'Right_J2', 'Right_J3', 'Right_J4', 'Right_J5', 'Right_J6', 'Right_J7', 'Right_Gripper'
        ]
        
    def connect_to_server(self):
        """Connect to inference server"""
        try:
            self.policy = websocket_client_policy.WebsocketClientPolicy(
                host=self.host, port=self.port
            )
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def load_gt_data(self, target_episode=None, max_episodes=3):
        """Load ground truth data"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_files:
            # 尝试直接在数据路径下查找
            parquet_files = list(self.data_path.glob("episode_*.parquet"))
            if not parquet_files:
                return False

        parquet_files = sorted(parquet_files)

        # 如果指定了目标episode，只加载该episode
        if target_episode is not None:
            target_file = None
            for file_path in parquet_files:
                # 从文件名提取episode编号
                filename = file_path.name
                if filename.startswith('episode_') and filename.endswith('.parquet'):
                    try:
                        episode_num = int(filename.split('_')[1].split('.')[0])
                        if episode_num == target_episode:
                            target_file = file_path
                            break
                    except (ValueError, IndexError):
                        continue

            if target_file is None:
                logger.error(f"找不到episode {target_episode}的parquet文件")
                logger.info(f"可用的episode文件: {[f.name for f in parquet_files[:10]]}")  # 显示前10个
                return False

            parquet_files = [target_file]
        else:
            parquet_files = parquet_files[:max_episodes]
        all_data = []
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                
                # 检查实际可用的相机和数据结构
                if self.debug and len(df) > 0:
                    logger.info(f"Parquet文件结构分析 ({file_path.name}):")
                    logger.info(f"列名: {list(df.columns)}")
                    
                    # 检查相机列
                    camera_cols = [col for col in df.columns if 'image' in col.lower()]
                    logger.info(f"可能的相机列: {camera_cols}")
                    
                    # 检查状态向量维度
                    if 'observation.state' in df.columns and len(df['observation.state']) > 0:
                        first_state = df['observation.state'].iloc[0]
                        actual_state_dim = len(first_state) if isinstance(first_state, (list, tuple)) else len(np.array(first_state))
                        logger.info(f"状态向量维度: {actual_state_dim}")

                    # 检查动作向量维度
                    if 'action' in df.columns and len(df['action']) > 0:
                        first_action = df['action'].iloc[0]
                        actual_action_dim = len(first_action) if isinstance(first_action, (list, tuple)) else len(np.array(first_action))
                        logger.info(f"动作向量维度: {actual_action_dim}")

                # 自动检测数据维度并加载动作和状态数据
                # 先检查第一个样本来确定实际维度
                if len(df) > 0:
                    first_action = np.array(df['action'].iloc[0])
                    first_state = np.array(df['observation.state'].iloc[0])
                    actual_action_dim = len(first_action)
                    actual_state_dim = len(first_state)

                    # 使用实际维度，但不超过16维（左臂7+左夹爪1+右臂7+右夹爪1）
                    use_action_dim = min(actual_action_dim, 16)
                    use_state_dim = min(actual_state_dim, 16)

                    logger.info(f"实际动作维度: {actual_action_dim}, 使用维度: {use_action_dim}")
                    logger.info(f"实际状态维度: {actual_state_dim}, 使用维度: {use_state_dim}")

                    actions = np.array([np.array(action)[:use_action_dim] for action in df['action']])
                    states = np.array([np.array(state)[:use_state_dim] for state in df['observation.state']])
                else:
                    # 如果没有数据，使用默认的16维
                    actions = np.array([])
                    states = np.array([])
                
                # 收集图像数据（如果存在）
                image_columns = [col for col in df.columns if 'image' in col.lower()]
                
                # 存储列名到数据字典中，用于后续格式确定
                column_names = list(df.columns)
                
                # 直接存储原始数据帧引用，这样我们在推理时可以直接访问
                episode_data = {
                    'episode_file': file_path.name,
                    'actions': actions,
                    'states': states,
                    'df': df,  # 存储整个数据帧
                    'image_columns': image_columns,  # 存储图像列名
                    'column_names': column_names,  # 存储所有列名
                    'length': len(actions)
                }
                all_data.append(episode_data)
                logger.info(f"已加载 {file_path.name}：{len(actions)}步，状态形状{states.shape}，动作形状{actions.shape}，图像列数量{len(image_columns)}")
                
            except Exception as e:
                logger.error(f"无法加载 {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        self.gt_data = all_data
        return len(all_data) > 0
    
    def run_full_episode_inference(self, episode_idx=0, step_limit=None, max_consecutive_errors=10, future_steps=30):
        """执行完整剧集推理，预测未来30个时刻的状态"""
        if not self.gt_data or episode_idx >= len(self.gt_data):
            return None
            
        episode = self.gt_data[episode_idx]
        total_steps = episode['length']
        
        if step_limit is not None:
            max_steps = min(step_limit, total_steps - future_steps)  # 确保有足够的步骤进行未来预测
        else:
            max_steps = total_steps - future_steps
            
        logger.info(f"执行完整剧集推理：{max_steps}/{total_steps}步，每步预测未来{future_steps}个时刻")
        
        gt_states = []
        pred_states = []
        inference_times = []
        
        success_count = 0
        consecutive_errors = 0
        
        # 使用OpenPI的tokenizer
        try:
            # 尝试导入tokenizer
            from openpi.models import tokenizer
            logger.info("成功导入OpenPI tokenizer模块")
            have_tokenizer = True
        except ImportError:
            logger.warning("无法导入OpenPI tokenizer模块，将使用占位符")
            have_tokenizer = False
        
        prompt_text = "pick and place purple long eggplant"
        
        # 创建一个通用的tokenize函数
        def simple_tokenize(text, max_length=48):
            """简单的tokenize函数，将每个字符转为ASCII码"""
            tokens = [ord(c) % 1000 for c in text]  # 使用ASCII值模1000作为简单token
            # 添加批次维度并填充到指定长度
            result = np.zeros((1, max_length), dtype=np.int32)
            mask = np.zeros((1, max_length), dtype=bool)
            length = min(len(tokens), max_length)
            result[0, :length] = tokens[:length]
            mask[0, :length] = True
            return result, mask
        
        # 定义映射关系
        image_key_map = {
            "exterior_image_1_left": "base_0_rgb",
            "wrist_image_left": "left_wrist_0_rgb",
            "wrist_image_right": "right_wrist_0_rgb"
        }
        
        for step_idx in range(max_steps):
            try:
                # 获取当前步骤的真实状态
                current_state = episode['states'][step_idx]
                # 将状态向量扩展到32维以匹配服务器期望
                padded_state = np.zeros(32, dtype=np.float32)
                padded_state[:len(current_state)] = current_state

                if self.debug and step_idx == 0:
                    logger.info(f"当前状态维度: {len(current_state)}, 填充后维度: {len(padded_state)}")
                    logger.info(f"状态前8维: {current_state[:8] if len(current_state) >= 8 else current_state}")
                    if len(current_state) >= 16:
                        logger.info(f"状态结构: 左臂[0:7]={current_state[:7]}, 左夹爪[7]={current_state[7]}, 右臂[8:15]={current_state[8:15]}, 右夹爪[15]={current_state[15]}")
                # 获取图像数据 - 直接从数据帧获取
                df = episode['df']
                image_columns = episode['image_columns']
                # 准备图像数据字典 - 与训练时保持一致
                images = {}
                if image_columns:
                    row = df.iloc[step_idx]
                    for col in image_columns:
                        if '.' in col:
                            cam_name = col.split('.')[-1]
                        else:
                            cam_name = col
                        img_data = row[col]
                        if img_data is not None:
                            mapped_key = image_key_map.get(cam_name, cam_name)
                            if isinstance(img_data, dict) and 'bytes' in img_data:
                                images[mapped_key] = decode_image(img_data['bytes'])
                                # # debug，用PIL保存图片查看是否正常
                                if self.debug and step_idx < 3:  # 只保存前3步的图片
                                    # 将[-1,1]的float32转换回[0,255]的uint8格式用于保存
                                    debug_img_array = ((images[mapped_key] + 1.0) / 2.0 * 255.0).astype(np.uint8)
                                    debug_img = Image.fromarray(debug_img_array)
                                    debug_path = f"debug_step_{step_idx}_{mapped_key}.png"
                                    debug_img.save(debug_path)
                                    logger.info(f"Debug: 保存图片到 {debug_path}, 形状: {images[mapped_key].shape}")
                            elif isinstance(img_data, bytes):
                                images[mapped_key] = decode_image(img_data)
                                # # debug，用PIL保存图片查看是否正常
                                if self.debug and step_idx < 3:  # 只保存前3步的图片
                                    # 将[-1,1]的float32转换回[0,255]的uint8格式用于保存
                                    debug_img_array = ((images[mapped_key] + 1.0) / 2.0 * 255.0).astype(np.uint8)
                                    debug_img = Image.fromarray(debug_img_array)
                                    debug_path = f"debug_step_{step_idx}_{mapped_key}.png"
                                    debug_img.save(debug_path)
                                    logger.info(f"Debug: 保存图片到 {debug_path}, 形状: {images[mapped_key].shape}")
                            else:
                                images[mapped_key] = img_data

                # 确保所有必需的相机都存在 - 与训练时保持一致
                required_cameras = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

                # 如果有base_0_rgb，获取其形状用于创建黑色图像
                if "base_0_rgb" in images:
                    base_shape = images["base_0_rgb"].shape
                    black_image = np.zeros(base_shape, dtype=images["base_0_rgb"].dtype)
                else:
                    # 默认形状，如果没有任何图像
                    black_image = np.zeros((224, 224, 3), dtype=np.uint8)

                # 为缺失的相机添加黑色图像
                for cam in required_cameras:
                    if cam not in images:
                        images[cam] = black_image.copy()
                        if self.debug:
                            logger.info(f"Step {step_idx}: 为缺失的相机 {cam} 添加黑色图像")

                # 创建image_mask - 与训练时保持一致
                image_masks = {}
                for cam in required_cameras:
                    if cam == "base_0_rgb":
                        # base_0_rgb 总是可用的（即使是黑色图像）
                        image_masks[cam] = True
                    else:
                        # 其他相机只有在原始数据中存在时才标记为可用
                        original_available = any(
                            image_key_map.get(col.split('.')[-1] if '.' in col else col, col) == cam
                            for col in image_columns
                        )
                        image_masks[cam] = original_available

                # --- 新增详细日志 ---
                if self.debug:
                    logger.info(f"Step {step_idx}: 输入state前5维: {padded_state[:5]}")
                    for k, v in images.items():
                        logger.info(f"Step {step_idx}: image[{k}] shape: {getattr(v, 'shape', None)} dtype: {getattr(v, 'dtype', None)} mask: {image_masks.get(k, 'N/A')}")

                # 使用OpenPI tokenizer处理提示词
                if have_tokenizer:
                    tokens, tokenized_prompt_mask = simple_tokenize(prompt_text)
                else:
                    tokens = np.zeros((1, 48), dtype=np.int32)
                    tokenized_prompt_mask = np.ones((1, 48), dtype=bool)

                obs = {
                    "state": padded_state,
                    "image": images,
                    "image_mask": image_masks,  # 添加image_mask
                    "prompt": prompt_text,
                    "tokenized_prompt": tokens,
                    "tokenized_prompt_mask": tokenized_prompt_mask
                }
                # 执行推理
                start_time = time.time()
                if self.policy is None:
                    raise ValueError("策略未初始化 - 服务器连接失败")
                result = self.policy.infer(obs)
                inference_time = time.time() - start_time
                # --- 新增详细日志 ---
                if self.debug:
                    logger.info(f"Step {step_idx}: 推理返回keys: {list(result.keys())}")
                # 使用动作预测而不是状态（因为state键只是返回输入状态）
                if "actions" in result:
                    pred_state_key = "actions"
                elif "future_states" in result:
                    pred_state_key = "future_states"
                elif "predicted_state" in result:
                    pred_state_key = "predicted_state"
                elif "states" in result:
                    pred_state_key = "states"
                else:
                    logger.warning("在结果中找不到动作键，尝试使用状态")
                    pred_state_key = "state"
                # 记录首次返回的数据结构
                if step_idx == 0 or self.debug:
                    logger.info(f"Step {step_idx}: 使用键 '{pred_state_key}' 获取预测动作")
                    if pred_state_key in result:
                        pred_data = result[pred_state_key]
                        logger.info(f"Step {step_idx}: 预测数据类型: {type(pred_data)}, 形状: {getattr(pred_data, 'shape', 'N/A')}, 内容前5: {np.array(pred_data).flatten()[:5]}")
                # 获取当前和未来步骤的真实状态
                gt_state_current = current_state
                gt_states_future = []
                for i in range(future_steps):
                    if step_idx + i < len(episode['states']):
                        gt_states_future.append(episode['states'][step_idx + i])
                    else:
                        gt_states_future.append(episode['states'][-1])
                # 从结果中获取预测状态
                if pred_state_key in result:
                    pred_states_data = result[pred_state_key]
                    # 如果只返回了当前状态
                    if not isinstance(pred_states_data, (list, tuple)) or len(pred_states_data) == 1:
                        pred_states_future = [pred_states_data] * future_steps
                    else:
                        pred_states_future = pred_states_data[:future_steps]
                        if len(pred_states_future) < future_steps:
                            last_state = pred_states_future[-1]
                            if isinstance(pred_states_future, tuple):
                                pred_states_future = list(pred_states_future)
                            for _ in range(future_steps - len(pred_states_future)):
                                pred_states_future.append(last_state)
                else:
                    logger.warning(f"在结果中找不到键 '{pred_state_key}'，使用持久性预测")
                    pred_states_future = [current_state] * future_steps
                # --- 检查预测是否合理（动作预测应该与当前状态不同）---
                if pred_state_key == "actions":
                    # 对于动作预测，使用统一的prediction_step选择哪一步进行比较
                    selected_action = np.array(pred_states_future[0])  # 这是(50,32)
                    # 使用实际的状态维度进行比较
                    actual_dim = len(current_state)
                    if len(selected_action.shape) == 2:  # 如果是(50,32)形状
                        pred_action = selected_action[self.prediction_step, :actual_dim]  # 使用统一参数选择步数
                    else:  # 如果已经是(32,)形状
                        pred_action = selected_action[:actual_dim]
                    current_state_actual = np.array(current_state)[:actual_dim]
                    if np.allclose(pred_action, current_state_actual, atol=1e-6):
                        logger.warning(f"Step {step_idx}: 预测动作与当前状态过于相似，可能未真正推理！")
                    else:
                        logger.info(f"Step {step_idx}: 预测动作与当前状态差异正常，推理成功 (使用第{self.prediction_step}步预测, 维度: {actual_dim})")
                else:
                    # 对于状态预测，检查是否与GT状态相同
                    # 使用实际的状态维度进行比较
                    pred_state_to_check = np.array(pred_states_future[0])[:len(current_state)]
                    gt_state_to_check = np.array(gt_states_future[0])[:len(current_state)]
                    if np.allclose(pred_state_to_check, gt_state_to_check):
                        logger.warning(f"Step {step_idx}: 预测和GT完全一致，可能未真正推理！")
                # 存储结果
                gt_states.append(gt_states_future)
                pred_states.append(pred_states_future)
                inference_times.append(inference_time)
                success_count += 1
                consecutive_errors = 0  # 重置连续错误计数
                if step_idx % 20 == 0 or step_idx == max_steps - 1:
                    logger.info(f"进度: {step_idx+1}/{max_steps}步, "
                              f"平均时间: {np.mean(inference_times):.3f}秒, "
                              f"成功率: {success_count/(step_idx+1)*100:.1f}%")
                
            except Exception as e:
                logger.error(f"步骤 {step_idx} 失败: {e}")
                # 记录详细的错误信息和堆栈跟踪
                logger.error(f"详细错误: {traceback.format_exc()}")
                
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"连续出现{consecutive_errors}个错误，中止处理")
                    break
                
                # 尝试重新连接服务器
                if "connection" in str(e).lower() or "websocket" in str(e).lower():
                    logger.info("尝试重新连接服务器...")
                    try:
                        self.policy = websocket_client_policy.WebsocketClientPolicy(
                            host=self.host, port=self.port
                        )
                        logger.info("服务器重连成功")
                    except Exception as reconnect_error:
                        logger.error(f"服务器重连失败: {reconnect_error}")
                        
                continue
        
        # 检查是否有成功的推理结果
        if not gt_states or not pred_states:
            logger.error("没有成功的推理步骤，无法生成结果")
            return None
            
        # 转换为numpy数组
        gt_states_array = np.array(gt_states)
        pred_states_array = np.array(pred_states)
        
        # 验证数组维度
        logger.info(f"完成推理: {len(gt_states)}/{max_steps}步骤成功, "
                   f"gt_shape={gt_states_array.shape}, pred_shape={pred_states_array.shape}")
        
        return {
            'gt_states': gt_states_array,
            'pred_states': pred_states_array,
            'inference_times': np.array(inference_times),
            'episode_idx': episode_idx,
            'episode_file': episode['episode_file'],
            'total_steps': len(gt_states),
            'future_steps': future_steps
        }
    
    def create_full_episode_plot(self, results, save_path="full_episode_inference.png"):
        """Create multi-step prediction visualization for the full episode (English annotation, only plot t+1 prediction for all timesteps)"""
        gt_states = results['gt_states']
        pred_states_raw = results['pred_states']
        future_steps = results['future_steps']

        # 动态确定关节数量，基于实际数据维度
        actual_joint_count = min(gt_states.shape[-1], len(self.joint_names))

        # 处理预测数据形状：从(200, 30, 50, 32)选择指定步的前N维 -> (200, 30, N)
        if len(pred_states_raw.shape) == 4:  # (timesteps, future_steps, action_horizon, action_dim)
            # 使用统一的prediction_step选择哪一步动作的前N维
            pred_states = pred_states_raw[:, :, self.prediction_step, :actual_joint_count]
            logger.info(f"可视化使用第{self.prediction_step}步预测 (共{pred_states_raw.shape[2]}步), 关节数量: {actual_joint_count}")
        else:
            # 如果已经是正确形状，直接使用前N维
            pred_states = pred_states_raw[..., :actual_joint_count]

        # Check array shapes
        if len(gt_states.shape) < 3 or len(pred_states.shape) < 3:
            logger.error(f"Shape error: gt_states shape {gt_states.shape}, pred_states shape {pred_states.shape}")
            logger.error("Cannot create multi-step prediction plot, please ensure future steps are returned.")
            return None

        steps = np.arange(len(gt_states))

        # 动态创建子图，每个关节一行
        fig, axes = plt.subplots(actual_joint_count, 1, figsize=(24, 3.5*actual_joint_count), sharex=True)
        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(
            f'Episode {results["episode_idx"]} t+1 Prediction (All Timesteps) ({results["total_steps"]} steps, Avg: {np.mean(results["inference_times"]):.3f}s/step, {1/np.mean(results["inference_times"]):.1f} Hz)',
            fontsize=22, fontweight='bold'
        )

        for joint_idx in range(actual_joint_count):
            ax = axes[joint_idx]
            joint_name = self.joint_names[joint_idx] if joint_idx < len(self.joint_names) else f'Joint_{joint_idx}'
            try:
                gt_values = gt_states[:, 0, joint_idx]
                pred_values = pred_states[:, 0, joint_idx]

                ax.plot(steps, gt_values, color='royalblue', linewidth=2.5, label='Ground Truth')
                ax.plot(steps, pred_values, color='orangered', linewidth=2.5, label='Prediction', linestyle='--')

                mse = np.mean((pred_values - gt_values) ** 2)
                mae = np.mean(np.abs(pred_values - gt_values))
                if np.std(gt_values) > 1e-6:
                    corr = np.corrcoef(gt_values, pred_values)[0, 1]
                    corr_text = f'{corr:.3f}'
                else:
                    corr_text = 'N/A'

                ax.set_title(f'{joint_name}   MAE:{mae:.3f}   Corr:{corr_text}', fontsize=16)
                ax.set_ylabel('Value', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=12)
                if joint_idx == 0:
                    ax.legend(fontsize=14, loc='upper right')
            except IndexError:
                ax.set_title(f'{joint_name}\nData unavailable', fontsize=16)
                ax.text(0.5, 0.5, 'Index error', ha='center', va='center', transform=ax.transAxes)
                continue

        axes[-1].set_xlabel('Timestep', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Full episode t+1 prediction plot saved to: {save_path}")

        self.print_multi_step_prediction_stats(results)
        return save_path

    def create_multi_step_comparison_plot(self, results, save_path="multi_step_comparison.png", max_timesteps=10, prediction_horizon=20):
        """创建多步预测对比图，每个时刻的预测轨迹用一条线段表示"""
        gt_states = results['gt_states']
        pred_states_raw = results['pred_states']

        # 处理预测数据形状：从(200, 30, 50, 32)获取多步预测
        if len(pred_states_raw.shape) != 4:
            logger.error(f"预测数据形状不正确: {pred_states_raw.shape}, 期望4维")
            return None

        total_timesteps = pred_states_raw.shape[0]
        available_horizon = pred_states_raw.shape[2]  # 50步

        # 限制显示的时间步数和预测步长
        max_timesteps = min(max_timesteps, total_timesteps)
        prediction_horizon = min(prediction_horizon, available_horizon)

        # 生成颜色映射 - 每个时刻一种颜色
        colors = plt.cm.tab20(np.linspace(0, 1, max_timesteps))

        # 动态确定关节数量
        actual_joint_count = min(gt_states.shape[-1], len(self.joint_names))

        # 创建子图：N个关节，每个关节一行
        fig, axes = plt.subplots(actual_joint_count, 1, figsize=(24, 3.5*actual_joint_count), sharex=True)
        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(
            f'Multi-Timestep Prediction Trajectories ({actual_joint_count} joints)\n'
            f'Showing {max_timesteps} timesteps, each predicting {prediction_horizon} steps ahead',
            fontsize=16, fontweight='bold', y=0.995
        )

        for joint_idx in range(actual_joint_count):
            ax = axes[joint_idx]

            # 绘制GT轨迹
            gt_full = gt_states[:, 0, joint_idx]  # 完整的GT轨迹
            ax.plot(range(len(gt_full)), gt_full, color='black', linewidth=3, label='Ground Truth', alpha=0.8)

            # 绘制每个时刻的预测轨迹
            for t_idx in range(0, max_timesteps, 3):  # 每10步画一条轨迹
                color = colors[t_idx % len(colors)]

                # 获取在时刻t_idx的预测轨迹（未来prediction_horizon步）
                pred_trajectory = pred_states_raw[t_idx, 0, :prediction_horizon, joint_idx]

                # 时间轴：从t_idx+1开始的prediction_horizon步
                time_axis = range(t_idx + 1, t_idx + 1 + prediction_horizon)

                # 确保不超出总时间范围
                if time_axis[-1] < len(gt_full):
                    ax.plot(time_axis, pred_trajectory,
                           color=color,
                           linewidth=2,
                           alpha=0.7,
                           label=f'Pred from t={t_idx}' if joint_idx == 0 else "",
                           linestyle='-')

                    # 在起始点添加标记
                    ax.scatter(t_idx + 1, pred_trajectory[0], color=color, s=30, alpha=0.8, zorder=5)

            # 设置子图属性
            joint_name = self.joint_names[joint_idx] if joint_idx < len(self.joint_names) else f'Joint_{joint_idx}'
            ax.set_ylabel(f'{joint_name}\nValue', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(len(gt_full), max_timesteps + prediction_horizon))

            # 只在第一个子图显示图例
            if joint_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

            # 添加说明文本
            actual_trajectories = len(range(0, max_timesteps, 10))
            ax.text(0.02, 0.98, f'Showing {actual_trajectories} prediction trajectories (every 10 steps)',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"多时刻预测轨迹图已保存到: {save_path} (显示{max_timesteps}个时刻，每个预测{prediction_horizon}步)")
        return save_path

    def print_multi_step_prediction_stats(self, results):
        """打印多步预测的综合统计信息"""

        gt_states = results['gt_states']
        pred_states_raw = results['pred_states']
        future_steps = results['future_steps']

        # 动态确定关节数量
        actual_joint_count = min(gt_states.shape[-1], len(self.joint_names))

        # 处理预测数据形状：从(200, 30, 50, 32)选择指定步的前N维 -> (200, 30, N)
        if len(pred_states_raw.shape) == 4:  # (timesteps, future_steps, action_horizon, action_dim)
            # 使用统一的prediction_step选择哪一步动作的前N维
            pred_states = pred_states_raw[:, :, self.prediction_step, :actual_joint_count]
        else:
            # 如果已经是正确形状，直接使用前N维
            pred_states = pred_states_raw[..., :actual_joint_count]
        
        print("\n" + "="*90)
        print(f"📊 完整剧集 {results['episode_idx']} 未来{future_steps}步预测分析")
        print("="*90)
        print(f"数据文件: {results['episode_file']}")
        print(f"分析步骤总数: {results['total_steps']}")
        print(f"平均推理时间: {np.mean(results['inference_times']):.3f}秒")
        print(f"推理频率: {1/np.mean(results['inference_times']):.1f} Hz")
        print(f"总推理时间: {np.sum(results['inference_times']):.1f}秒")
        print("-"*90)
        
        # 按预测步长计算统计数据
        print(f"按预测步长的性能统计:")
        print(f"{'步长':<8} {'MSE':<12} {'MAE':<12} {'相关系数':<12}")
        print("-"*50)
        
        max_steps_to_analyze = min(5, gt_states.shape[1], pred_states.shape[1])
        
        for step in range(max_steps_to_analyze):
            # 获取特定预测步长的所有数据点
            gt_horizon = gt_states[:, step, :].reshape(-1)
            pred_horizon = pred_states[:, step, :].reshape(-1)
            
            # 计算统计数据
            mse = np.mean((pred_horizon - gt_horizon) ** 2)
            mae = np.mean(np.abs(pred_horizon - gt_horizon))
            
            if np.std(gt_horizon) > 1e-6 and np.std(pred_horizon) > 1e-6:
                corr = np.corrcoef(gt_horizon, pred_horizon)[0, 1]
                corr_text = f"{corr:.4f}"
            else:
                corr_text = "N/A"
                
            print(f"{step:<8} {mse:<12.6f} {mae:<12.6f} {corr_text:<12}")
            
        print("="*90)


def main():
    """Main function"""
    import argparse
    """
conda activate openpi && python inference_tienkung/inference_with_dis_openloop.py \
  --host localhost --port 8000 \
  --data_path /home/q/data/pick_up_parts_from_belt_conveyor_place_on_plate_fast_250702/openpi \
  --episode 0 --step_limit 200 --output my_infer.png \
  --future_steps 50 --prediction_step 29    \
  --multi_step_plot  --max_timesteps 200 --prediction_horizon 50
      """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OpenPI完整剧集推理可视化（带扰动）")
    parser.add_argument("--host", default="localhost", help="推理服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="推理服务器端口")
    parser.add_argument("--data_path", default="/home/q/data/pick_and_place_eggplant/openpi", 
                        help="数据路径")
    parser.add_argument("--episode", type=int, default=0, help="要分析的剧集索引")
    parser.add_argument("--step_limit", type=int, default=20, 
                        help="步骤限制（设置为0表示不限制）")
    parser.add_argument("--output", default="test_inference.png", 
                        help="输出图像文件")
    parser.add_argument("--full_run", action="store_true",
                        help="完整运行（不推荐，除非确认小批量测试成功）")
    parser.add_argument("--max_errors", type=int, default=1,
                        help="允许的最大连续错误数，超过此数值将中止处理")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式，打印更多信息")
    parser.add_argument("--future_steps", type=int, default=30,
                        help="要预测的未来步骤数量")
    parser.add_argument("--prediction_step", type=int, default=0,
                        help="选择哪一步预测进行对比和可视化 (0:第一步, -1:最后一步)")
    parser.add_argument("--multi_step_plot", action="store_true",
                        help="生成多时刻预测轨迹对比图")
    parser.add_argument("--max_timesteps", type=int, default=10,
                        help="显示多少个时刻的预测轨迹")
    parser.add_argument("--prediction_horizon", type=int, default=20,
                        help="每个时刻预测多少步的轨迹")
    args = parser.parse_args()
    # 如果设置为0表示不限制
    step_limit = None if args.step_limit == 0 else args.step_limit
    try:
        # 初始化可视化器
        visualizer = FullEpisodeInferenceVisualizer(
            host=args.host,
            port=args.port,
            data_path=args.data_path,
            debug=args.debug,
            prediction_step=args.prediction_step
        )
        # 连接服务器
        if not visualizer.connect_to_server():
            logger.error("无法连接到推理服务器，请确保服务器正在运行")
            return 1
        # 加载数据
        if not visualizer.load_gt_data(target_episode=args.episode):
            logger.error("无法加载地面真值数据，请检查数据路径")
            return 1
        logger.info(f"开始为剧集 {args.episode} 进行完整分析...")
        # 运行完整剧集推理（现在只加载了目标episode，所以索引是0）
        results = visualizer.run_full_episode_inference(
            episode_idx=0,
            step_limit=step_limit,
            max_consecutive_errors=args.max_errors,
            future_steps=args.future_steps
        )
        if results is None:
            logger.error("推理失败，无法生成结果")
            return 1
        # 创建可视化
        visualizer.create_full_episode_plot(results, args.output)
        logger.info(f"完整剧集可视化已完成！结果保存到: {args.output}")

        # 如果需要，创建多时刻预测轨迹对比图
        if args.multi_step_plot:
            multi_step_output = args.output.replace('.png', '_multi_step.png')
            visualizer.create_multi_step_comparison_plot(results, multi_step_output, args.max_timesteps, args.prediction_horizon)
            logger.info(f"多时刻预测轨迹图已完成！结果保存到: {multi_step_output}")

        return 0
    except KeyboardInterrupt:
        logger.info("用户中断了操作")
        return 130
    except Exception as e:
        logger.exception(f"发生意外错误: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
