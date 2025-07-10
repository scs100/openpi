#!/usr/bin/env python3
"""
安全配置管理工具
用于动态调整机械臂控制的安全参数
"""

import json
import numpy as np
from pathlib import Path

class SafetyConfig:
    """安全配置管理器"""
    
    def __init__(self, config_file="safety_config.json"):
        self.config_file = Path(config_file)
        self.load_config()
    
    def load_config(self):
        """加载安全配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.max_single_step_change = config.get("max_single_step_change", 0.03)
                self.emergency_stop_threshold = config.get("emergency_stop_threshold", 0.3)
                self.max_velocity_estimate = config.get("max_velocity_estimate", 0.2)
                self.enable_safety_limits = config.get("enable_safety_limits", True)
                print(f"✅ 从 {self.config_file} 加载安全配置")
            except Exception as e:
                print(f"⚠️  加载配置失败: {e}，使用默认值")
                self.set_defaults()
        else:
            print("📁 配置文件不存在，使用默认安全配置")
            self.set_defaults()
    
    def set_defaults(self):
        """设置默认安全配置"""
        self.max_single_step_change = 0.03      # 约1.7度
        self.emergency_stop_threshold = 0.3     # 约17度
        self.max_velocity_estimate = 0.2        # 0.2 rad/s
        self.enable_safety_limits = True
    
    def save_config(self):
        """保存安全配置"""
        config = {
            "max_single_step_change": self.max_single_step_change,
            "emergency_stop_threshold": self.emergency_stop_threshold,
            "max_velocity_estimate": self.max_velocity_estimate,
            "enable_safety_limits": self.enable_safety_limits
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ 安全配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
    def get_safety_mode_configs(self):
        """获取预定义的安全模式配置"""
        return {
            "ultra_safe": {
                "max_single_step_change": 0.017,  # 1度
                "emergency_stop_threshold": 0.175, # 10度
                "max_velocity_estimate": 0.1,      # 0.1 rad/s
                "description": "超安全模式 - 最小运动，最高安全性"
            },
            "safe": {
                "max_single_step_change": 0.035,  # 2度
                "emergency_stop_threshold": 0.26,  # 15度
                "max_velocity_estimate": 0.2,      # 0.2 rad/s
                "description": "安全模式 - 平衡安全性和响应性"
            },
            "normal": {
                "max_single_step_change": 0.052,  # 3度
                "emergency_stop_threshold": 0.35,  # 20度
                "max_velocity_estimate": 0.3,      # 0.3 rad/s
                "description": "正常模式 - 标准安全限制"
            },
            "aggressive": {
                "max_single_step_change": 0.087,  # 5度
                "emergency_stop_threshold": 0.52,  # 30度
                "max_velocity_estimate": 0.5,      # 0.5 rad/s
                "description": "激进模式 - 高响应性，较低安全限制"
            }
        }
    
    def set_safety_mode(self, mode):
        """设置安全模式"""
        configs = self.get_safety_mode_configs()
        if mode not in configs:
            print(f"❌ 未知安全模式: {mode}")
            print(f"可用模式: {list(configs.keys())}")
            return False
        
        config = configs[mode]
        self.max_single_step_change = config["max_single_step_change"]
        self.emergency_stop_threshold = config["emergency_stop_threshold"]
        self.max_velocity_estimate = config["max_velocity_estimate"]
        
        print(f"✅ 设置安全模式: {mode}")
        print(f"   {config['description']}")
        self.print_current_config()
        return True
    
    def print_current_config(self):
        """打印当前配置"""
        print(f"\n🛡️  当前安全配置:")
        print(f"   📏 单步限制: {self.max_single_step_change:.3f}rad ({np.degrees(self.max_single_step_change):.1f}°)")
        print(f"   🚨 紧急阈值: {self.emergency_stop_threshold:.3f}rad ({np.degrees(self.emergency_stop_threshold):.1f}°)")
        print(f"   ⚡ 最大速度: {self.max_velocity_estimate:.2f}rad/s")
        print(f"   🔒 安全限制: {'启用' if self.enable_safety_limits else '禁用'}")
    
    def interactive_config(self):
        """交互式配置"""
        print("🔧 交互式安全配置")
        print("=" * 50)
        
        while True:
            print("\n选择操作:")
            print("1. 查看当前配置")
            print("2. 设置安全模式")
            print("3. 自定义配置")
            print("4. 保存配置")
            print("5. 退出")
            
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == "1":
                self.print_current_config()
            
            elif choice == "2":
                configs = self.get_safety_mode_configs()
                print("\n可用安全模式:")
                for i, (mode, config) in enumerate(configs.items(), 1):
                    print(f"{i}. {mode}: {config['description']}")
                
                try:
                    mode_choice = int(input("\n选择模式 (1-4): ")) - 1
                    mode_names = list(configs.keys())
                    if 0 <= mode_choice < len(mode_names):
                        self.set_safety_mode(mode_names[mode_choice])
                    else:
                        print("❌ 无效选择")
                except ValueError:
                    print("❌ 请输入数字")
            
            elif choice == "3":
                print("\n自定义安全配置:")
                try:
                    step = float(input(f"单步限制 (当前: {self.max_single_step_change:.3f}rad): ") or self.max_single_step_change)
                    emergency = float(input(f"紧急阈值 (当前: {self.emergency_stop_threshold:.3f}rad): ") or self.emergency_stop_threshold)
                    velocity = float(input(f"最大速度 (当前: {self.max_velocity_estimate:.2f}rad/s): ") or self.max_velocity_estimate)
                    
                    self.max_single_step_change = step
                    self.emergency_stop_threshold = emergency
                    self.max_velocity_estimate = velocity
                    
                    print("✅ 自定义配置已设置")
                    self.print_current_config()
                    
                except ValueError:
                    print("❌ 请输入有效数字")
            
            elif choice == "4":
                self.save_config()
            
            elif choice == "5":
                break
            
            else:
                print("❌ 无效选择")

def main():
    """主函数"""
    print("🛡️  机械臂安全配置工具")
    print("用于调整xROCS控制服务的安全参数")
    print()
    
    config = SafetyConfig()
    config.interactive_config()
    
    print("\n👋 配置完成")

if __name__ == "__main__":
    main()
