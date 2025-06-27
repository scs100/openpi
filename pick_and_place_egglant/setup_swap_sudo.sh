#!/bin/bash

# 配置sudo免密执行训练相关命令的脚本
# 使用方法: sudo bash setup_swap_sudo.sh

set -e

echo "正在配置sudo免密执行训练相关命令..."

# 获取当前用户名
CURRENT_USER=${SUDO_USER:-$(whoami)}

# 创建sudoers配置文件
SUDOERS_FILE="/etc/sudoers.d/training-nopasswd"

# 写入配置
cat > "$SUDOERS_FILE" << EOF
# 允许用户免密执行训练相关命令
# 由 setup_swap_sudo.sh 自动生成

# Swap管理命令
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/swapoff
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/swapoff
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/swapon
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/swapon

# 系统内存管理命令
$CURRENT_USER ALL=(ALL) NOPASSWD: /sbin/sysctl vm.drop_caches=*
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/sysctl vm.drop_caches=*

# 文件系统同步命令
$CURRENT_USER ALL=(ALL) NOPASSWD: /bin/sync
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/sync
EOF

# 设置正确的权限
chmod 440 "$SUDOERS_FILE"

# 验证sudoers文件语法
if visudo -c -f "$SUDOERS_FILE"; then
    echo "✅ sudoers配置成功！"
    echo "用户 $CURRENT_USER 现在可以免密执行以下命令："
    echo "  - sudo swapoff -a"
    echo "  - sudo swapon -a"
    echo "  - sudo sysctl vm.drop_caches=1"
    echo "  - sudo sysctl vm.drop_caches=3"
    echo "  - sudo sync"
    echo ""
    echo "测试命令："
    echo "  sudo -n swapoff --help  # 应该不提示密码"
    echo "  sudo -n sysctl vm.drop_caches=0  # 应该不提示密码"
else
    echo "❌ sudoers配置文件语法错误，正在删除..."
    rm -f "$SUDOERS_FILE"
    exit 1
fi

echo ""
echo "配置完成！现在可以运行训练脚本而无需手动输入密码。"
