#!/bin/bash

# 配置sudo免密执行训练脚本所需的所有系统命令
# 使用方法: sudo bash setup_swap_sudo.sh

set -e

echo "正在配置sudo免密执行训练相关的系统命令..."

# 获取当前用户名
CURRENT_USER=${SUDO_USER:-$(whoami)}

# 创建sudoers配置文件
SUDOERS_FILE="/etc/sudoers.d/openpi-training-nopasswd"

# 写入配置
cat > "$SUDOERS_FILE" << EOF
# 允许用户免密执行OpenPI训练脚本所需的系统命令
# 由 setup_swap_sudo.sh 自动生成

# Swap管理命令 (内存管理)
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/swapoff
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/swapoff
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/swapon
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/swapon

# 系统缓存清理命令 (内存优化)
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/sysctl vm.drop_caches=1
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/sysctl vm.drop_caches=3
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/sysctl vm.drop_caches=1
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/sysctl vm.drop_caches=3

# GPU管理命令 (GPU重置，训练恢复时使用)
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi --gpu-reset
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/nvidia-smi --gpu-reset
EOF

# 设置正确的权限
chmod 440 "$SUDOERS_FILE"

# 验证sudoers文件语法
if visudo -c -f "$SUDOERS_FILE"; then
    echo "✅ sudoers配置成功！"
    echo "用户 $CURRENT_USER 现在可以免密执行以下命令："
    echo ""
    echo "📋 内存管理命令："
    echo "  - sudo swapoff -a          # 关闭所有swap"
    echo "  - sudo swapon -a           # 启用所有swap"
    echo ""
    echo "🧹 系统缓存清理命令："
    echo "  - sudo sysctl vm.drop_caches=1  # 清理页面缓存"
    echo "  - sudo sysctl vm.drop_caches=3  # 清理所有缓存"
    echo ""
    echo "🎮 GPU管理命令："
    echo "  - sudo nvidia-smi --gpu-reset   # GPU硬件重置"
    echo ""
    echo "🧪 测试命令："
    echo "  sudo -n swapoff --help     # 应该不提示密码"
    echo "  sudo -n sysctl --help      # 应该不提示密码"
    echo "  sudo -n nvidia-smi --help  # 应该不提示密码"
else
    echo "❌ sudoers配置文件语法错误，正在删除..."
    rm -f "$SUDOERS_FILE"
    exit 1
fi

echo ""
echo "🎉 配置完成！"
echo ""
echo "📝 说明："
echo "1. 这些配置在系统重启后依然有效，不需要重新运行"
echo "2. 配置文件位置: $SUDOERS_FILE"
echo "3. 如需删除配置: sudo rm $SUDOERS_FILE"
echo ""
echo "现在可以运行OpenPI训练脚本，所有内存管理操作都将自动执行，无需密码！"
