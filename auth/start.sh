#!/bin/bash

# API网关服务启动脚本

set -e

echo "开始启动API网关服务..."

# 检查必要的目录
mkdir -p /app/shared /app/logs

# 设置权限
chmod -R 755 /app/shared /app/logs

# 等待后端服务启动
echo "等待后端服务启动..."
sleep 15

echo "启动API网关服务..."

# 移除了有问题的pickle初始化代码
echo "跳过HTTP客户端预初始化（在应用启动时动态创建）..."

# 启动服务 - 使用python api.py以启用SSL配置
exec python app.py 