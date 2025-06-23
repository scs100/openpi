#!/usr/bin/env python3
"""
内存监控脚本
实时监控系统内存和swap使用情况
"""

import time
import psutil
import subprocess
import argparse
from datetime import datetime

def get_memory_info():
    """获取内存信息"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # GPU内存
    gpu_info = "N/A"
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, total = result.stdout.strip().split(', ')
            usage_pct = float(used) / float(total) * 100
            gpu_info = f"{used}MB/{total}MB ({usage_pct:.1f}%)"
    except:
        pass
    
    return {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'memory_total_gb': memory.total / (1024**3),
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'swap_total_gb': swap.total / (1024**3),
        'swap_used_gb': swap.used / (1024**3),
        'swap_percent': swap.percent,
        'gpu_info': gpu_info
    }

def print_memory_status(info, show_header=False):
    """打印内存状态"""
    if show_header:
        print("=" * 80)
        print(f"{'时间':<8} {'系统内存':<20} {'Swap':<15} {'GPU内存':<20}")
        print("=" * 80)
    
    memory_str = f"{info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f}GB ({info['memory_percent']:.1f}%)"
    swap_str = f"{info['swap_used_gb']:.1f}GB ({info['swap_percent']:.1f}%)"
    
    # 颜色编码
    memory_color = ""
    swap_color = ""
    
    if info['memory_percent'] > 90:
        memory_color = "🔴"
    elif info['memory_percent'] > 80:
        memory_color = "🟡"
    else:
        memory_color = "🟢"
    
    if info['swap_percent'] > 50:
        swap_color = "🔴"
    elif info['swap_percent'] > 20:
        swap_color = "🟡"
    else:
        swap_color = "🟢"
    
    print(f"{info['timestamp']:<8} {memory_color}{memory_str:<19} {swap_color}{swap_str:<14} {info['gpu_info']:<20}")

def monitor_memory(interval=5, duration=None, log_file=None):
    """监控内存使用"""
    print(f"🔍 开始监控内存使用 (间隔: {interval}秒)")
    if duration:
        print(f"⏱️ 监控时长: {duration}秒")
    if log_file:
        print(f"📝 日志文件: {log_file}")
    
    start_time = time.time()
    iteration = 0
    
    # 打开日志文件
    log_handle = None
    if log_file:
        log_handle = open(log_file, 'w', encoding='utf-8')
        log_handle.write("timestamp,memory_used_gb,memory_total_gb,memory_percent,swap_used_gb,swap_percent,gpu_info\n")
    
    try:
        while True:
            info = get_memory_info()
            
            # 显示表头
            if iteration % 20 == 0:
                print_memory_status(info, show_header=True)
            else:
                print_memory_status(info)
            
            # 写入日志
            if log_handle:
                log_handle.write(f"{info['timestamp']},{info['memory_used_gb']:.2f},{info['memory_total_gb']:.2f},"
                               f"{info['memory_percent']:.1f},{info['swap_used_gb']:.2f},{info['swap_percent']:.1f},"
                               f"{info['gpu_info']}\n")
                log_handle.flush()
            
            # 检查警告条件
            if info['memory_percent'] > 95:
                print(f"⚠️ 警告: 系统内存使用率过高 {info['memory_percent']:.1f}%")
            
            if info['swap_percent'] > 80:
                print(f"🚨 严重警告: Swap使用率过高 {info['swap_percent']:.1f}%")
            
            # 检查是否达到监控时长
            if duration and (time.time() - start_time) >= duration:
                break
            
            iteration += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")
    
    finally:
        if log_handle:
            log_handle.close()
            print(f"📝 日志已保存到: {log_file}")

def analyze_log(log_file):
    """分析日志文件"""
    try:
        import pandas as pd
        
        df = pd.read_csv(log_file)
        
        print(f"📊 内存使用分析 - {log_file}")
        print("=" * 50)
        
        print(f"📈 系统内存:")
        print(f"  - 平均使用率: {df['memory_percent'].mean():.1f}%")
        print(f"  - 最高使用率: {df['memory_percent'].max():.1f}%")
        print(f"  - 最低使用率: {df['memory_percent'].min():.1f}%")
        
        print(f"💾 Swap内存:")
        print(f"  - 平均使用率: {df['swap_percent'].mean():.1f}%")
        print(f"  - 最高使用率: {df['swap_percent'].max():.1f}%")
        print(f"  - 最低使用率: {df['swap_percent'].min():.1f}%")
        
        # 警告统计
        high_memory = (df['memory_percent'] > 90).sum()
        high_swap = (df['swap_percent'] > 50).sum()
        
        print(f"⚠️ 警告统计:")
        print(f"  - 高内存使用(>90%): {high_memory} 次")
        print(f"  - 高Swap使用(>50%): {high_swap} 次")
        
    except ImportError:
        print("❌ 需要安装pandas来分析日志: pip install pandas")
    except Exception as e:
        print(f"❌ 分析日志失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="内存监控工具")
    parser.add_argument('-i', '--interval', type=int, default=5, help='监控间隔(秒)')
    parser.add_argument('-d', '--duration', type=int, help='监控时长(秒)')
    parser.add_argument('-l', '--log', type=str, help='日志文件路径')
    parser.add_argument('-a', '--analyze', type=str, help='分析日志文件')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_log(args.analyze)
    else:
        monitor_memory(args.interval, args.duration, args.log)

if __name__ == "__main__":
    main()
