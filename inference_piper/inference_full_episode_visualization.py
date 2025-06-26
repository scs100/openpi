#!/usr/bin/env python3
"""
OpenPI Full Episode Inference Visualization
Complete episode inference with all timesteps
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from openpi_client import websocket_client_policy

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullEpisodeInferenceVisualizer:
    """Full episode inference visualizer"""
    
    def __init__(self, host="localhost", port=8000, data_path="/home/testuser/data/pick_and_place_eggplant/openpi"):
        self.host = host
        self.port = port
        self.data_path = Path(data_path)
        self.policy = None
        self.gt_data = None
        
        # Joint names for 14-dimensional actions
        self.joint_names = [
            'Left_J1', 'Left_J2', 'Left_J3', 'Left_J4', 'Left_J5', 'Left_J6', 'Left_Gripper',
            'Right_J1', 'Right_J2', 'Right_J3', 'Right_J4', 'Right_J5', 'Right_J6', 'Right_Gripper'
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
    
    def load_gt_data(self, max_episodes=3):
        """Load ground truth data"""
        parquet_files = list(self.data_path.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_files:
            return False
            
        parquet_files = sorted(parquet_files)[:max_episodes]
        all_data = []
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                actions = np.array([np.array(action)[:14] for action in df['action']])
                states = np.array([np.array(state)[:14] for state in df['observation.state']])
                
                episode_data = {
                    'episode_file': file_path.name,
                    'actions': actions,
                    'states': states,
                    'length': len(actions)
                }
                all_data.append(episode_data)
                logger.info(f"Loaded {file_path.name}: {len(actions)} steps")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        self.gt_data = all_data
        return len(all_data) > 0
    
    def run_full_episode_inference(self, episode_idx=0, step_limit=None):
        """Run inference for full episode"""
        if not self.gt_data or episode_idx >= len(self.gt_data):
            return None
            
        episode = self.gt_data[episode_idx]
        total_steps = episode['length']
        
        if step_limit is not None:
            max_steps = min(step_limit, total_steps)
        else:
            max_steps = total_steps
            
        logger.info(f"Running full episode inference: {max_steps}/{total_steps} steps")
        
        gt_actions = []
        pred_actions = []
        inference_times = []
        
        for step_idx in range(max_steps):
            try:
                # Create observation
                state = episode['states'][step_idx]
                obs = {
                    "state": state.astype(np.float32),
                    "images": {
                        "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                        "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                        "cam_left_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                        "cam_right_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                    },
                    "prompt": "pick and place purple long eggplant"
                }
                
                # Execute inference
                start_time = time.time()
                result = self.policy.infer(obs)
                inference_time = time.time() - start_time
                
                # Store results
                gt_action = episode['actions'][step_idx]
                pred_action = result["actions"][0]  # Current timestep prediction
                
                gt_actions.append(gt_action)
                pred_actions.append(pred_action)
                inference_times.append(inference_time)
                
                # Progress logging
                if step_idx % 20 == 0 or step_idx == max_steps - 1:
                    logger.info(f"Progress: {step_idx+1}/{max_steps} steps, "
                              f"avg time: {np.mean(inference_times):.3f}s")
                    
            except Exception as e:
                logger.error(f"Step {step_idx} failed: {e}")
                continue
        
        return {
            'gt_actions': np.array(gt_actions),
            'pred_actions': np.array(pred_actions),
            'inference_times': np.array(inference_times),
            'episode_idx': episode_idx,
            'episode_file': episode['episode_file'],
            'total_steps': len(gt_actions)
        }
    
    def create_full_episode_plot(self, results, save_path="full_episode_inference.png"):
        """Create comprehensive full episode plot"""
        
        gt_actions = results['gt_actions']
        pred_actions = results['pred_actions']
        steps = np.arange(len(gt_actions))
        
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        episode_idx = results['episode_idx']
        total_steps = results['total_steps']
        avg_time = np.mean(results['inference_times'])
        
        fig.suptitle(f'Complete Episode {episode_idx} Inference Analysis ({total_steps} steps, '
                    f'avg: {avg_time:.3f}s/step, {1/avg_time:.1f} Hz)', 
                    fontsize=16, fontweight='bold')
        
        # Create grid layout: 4 rows x 4 columns for 14 joints + 2 summary plots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Plot each joint (14 joints in 4x4 grid, leaving 2 spaces for summary)
        for joint_idx in range(14):
            row = joint_idx // 4
            col = joint_idx % 4
            
            ax = fig.add_subplot(gs[row, col])
            
            joint_name = self.joint_names[joint_idx]
            gt_values = gt_actions[:, joint_idx]
            pred_values = pred_actions[:, joint_idx]
            
            # Plot ground truth and prediction
            ax.plot(steps, gt_values, 'b-', linewidth=1.5, alpha=0.8, label='GT')
            ax.plot(steps, pred_values, 'r-', linewidth=1.5, alpha=0.8, label='Pred')
            
            # Calculate and display statistics
            mse = np.mean((pred_values - gt_values) ** 2)
            mae = np.mean(np.abs(pred_values - gt_values))
            
            if np.std(gt_values) > 1e-6:
                corr = np.corrcoef(gt_values, pred_values)[0, 1]
                corr_text = f'{corr:.3f}'
            else:
                corr_text = 'N/A'
            
            ax.set_title(f'{joint_name}\nMAE:{mae:.3f} Corr:{corr_text}', fontsize=10)
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if joint_idx == 0:  # Only show legend on first plot
                ax.legend(fontsize=8)
        
        # Summary plot 1: Overall error trends (bottom left)
        ax_error = fig.add_subplot(gs[3, 2])
        
        # Calculate overall error metrics
        errors = np.abs(pred_actions - gt_actions)
        avg_error_per_step = np.mean(errors, axis=1)
        max_error_per_step = np.max(errors, axis=1)
        
        ax_error.plot(steps, avg_error_per_step, 'orange', linewidth=2, label='Avg Error')
        ax_error.fill_between(steps, 0, avg_error_per_step, alpha=0.3, color='orange')
        ax_error.plot(steps, max_error_per_step, 'red', linewidth=1, alpha=0.7, label='Max Error')
        
        ax_error.set_title('Error Evolution Over Time', fontsize=12, fontweight='bold')
        ax_error.set_xlabel('Time Step')
        ax_error.set_ylabel('Absolute Error')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        
        # Summary plot 2: Performance metrics (bottom right)
        ax_metrics = fig.add_subplot(gs[3, 3])
        
        # Calculate rolling performance metrics
        window_size = max(5, len(steps) // 20)
        if len(steps) >= window_size:
            rolling_mse = []
            rolling_mae = []
            rolling_corr = []
            
            for i in range(window_size-1, len(steps)):
                start_idx = i - window_size + 1
                end_idx = i + 1
                
                window_gt = gt_actions[start_idx:end_idx].flatten()
                window_pred = pred_actions[start_idx:end_idx].flatten()
                
                mse = np.mean((window_pred - window_gt) ** 2)
                mae = np.mean(np.abs(window_pred - window_gt))
                
                if np.std(window_gt) > 1e-6:
                    corr = np.corrcoef(window_gt, window_pred)[0, 1]
                else:
                    corr = 0
                
                rolling_mse.append(mse)
                rolling_mae.append(mae)
                rolling_corr.append(corr)
            
            rolling_steps = steps[window_size-1:]
            
            ax_metrics_twin = ax_metrics.twinx()
            
            line1 = ax_metrics.plot(rolling_steps, rolling_mse, 'r-', label='MSE', linewidth=2)
            line2 = ax_metrics.plot(rolling_steps, rolling_mae, 'b-', label='MAE', linewidth=2)
            line3 = ax_metrics_twin.plot(rolling_steps, rolling_corr, 'g-', label='Correlation', linewidth=2)
            
            ax_metrics.set_xlabel('Time Step')
            ax_metrics.set_ylabel('MSE / MAE', color='black')
            ax_metrics_twin.set_ylabel('Correlation', color='green')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax_metrics.legend(lines, labels, loc='upper right')
        
        ax_metrics.set_title(f'Rolling Metrics (window={window_size})', fontsize=12, fontweight='bold')
        ax_metrics.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Full episode plot saved to: {save_path}")
        
        # Print comprehensive statistics
        self.print_full_episode_stats(results)
        
        return save_path
    
    def print_full_episode_stats(self, results):
        """Print comprehensive episode statistics"""
        
        gt_actions = results['gt_actions']
        pred_actions = results['pred_actions']
        
        print("\n" + "="*90)
        print(f"📊 COMPLETE EPISODE {results['episode_idx']} INFERENCE ANALYSIS")
        print("="*90)
        print(f"Episode File: {results['episode_file']}")
        print(f"Total Steps Analyzed: {results['total_steps']}")
        print(f"Average Inference Time: {np.mean(results['inference_times']):.3f}s")
        print(f"Inference Frequency: {1/np.mean(results['inference_times']):.1f} Hz")
        print(f"Total Inference Time: {np.sum(results['inference_times']):.1f}s")
        print("-"*90)
        
        # Overall statistics
        overall_mse = np.mean((pred_actions - gt_actions) ** 2)
        overall_mae = np.mean(np.abs(pred_actions - gt_actions))
        overall_corr = np.corrcoef(gt_actions.flatten(), pred_actions.flatten())[0, 1]
        
        print(f"OVERALL PERFORMANCE:")
        print(f"  MSE: {overall_mse:.6f}")
        print(f"  MAE: {overall_mae:.6f}")
        print(f"  Correlation: {overall_corr:.6f}")
        print(f"  Max Error: {np.max(np.abs(pred_actions - gt_actions)):.6f}")
        print(f"  Min Error: {np.min(np.abs(pred_actions - gt_actions)):.6f}")
        print("-"*90)
        
        # Per-joint statistics
        print(f"PER-JOINT PERFORMANCE:")
        print(f"{'Joint':<15} {'MSE':<10} {'MAE':<10} {'Corr':<8} {'Max_Err':<10} {'Range_GT':<12}")
        print("-"*90)
        
        for joint_idx in range(14):
            joint_name = self.joint_names[joint_idx]
            gt_values = gt_actions[:, joint_idx]
            pred_values = pred_actions[:, joint_idx]
            
            mse = np.mean((pred_values - gt_values) ** 2)
            mae = np.mean(np.abs(pred_values - gt_values))
            max_error = np.max(np.abs(pred_values - gt_values))
            gt_range = np.max(gt_values) - np.min(gt_values)
            
            if np.std(gt_values) > 1e-6:
                corr = np.corrcoef(gt_values, pred_values)[0, 1]
                corr_str = f"{corr:.3f}"
            else:
                corr_str = "N/A"
            
            print(f"{joint_name:<15} {mse:<10.6f} {mae:<10.6f} {corr_str:<8} "
                  f"{max_error:<10.6f} {gt_range:<12.6f}")
        
        print("="*90)

def main():
    """Main function"""
    visualizer = FullEpisodeInferenceVisualizer()
    
    if not visualizer.connect_to_server():
        logger.error("Cannot connect to inference server")
        return
    
    if not visualizer.load_gt_data():
        logger.error("Cannot load ground truth data")
        return
    
    # Choose episode and step limit
    episode_to_analyze = 0  # Change this to analyze different episodes
    step_limit = 6000  # Set to None for complete episode, or limit for faster processing
    
    logger.info(f"Starting full episode analysis for episode {episode_to_analyze}...")
    
    # Run full episode inference
    results = visualizer.run_full_episode_inference(
        episode_idx=episode_to_analyze, 
        step_limit=step_limit
    )
    
    if results is None:
        logger.error("Failed to run inference")
        return
    
    # Create comprehensive visualization
    save_path = f"full_episode_{episode_to_analyze}_inference.png"
    visualizer.create_full_episode_plot(results, save_path)
    
    logger.info("Full episode visualization completed!")

if __name__ == "__main__":
    main()
