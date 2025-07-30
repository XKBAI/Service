#!/bin/bash

# 设置错误处理
set -e

echo "🚀 启动CosyVoice统一服务..."

# 检查模型文件（使用绝对容器路径）
if [ ! -d "/app/models/CosyVoice2-0.5B" ]; then
    echo "❌ 模型文件不存在: /app/models/CosyVoice2-0.5B"
    exit 1
fi

# 进入工作目录（使用绝对路径）
cd /app/async_cosyvoice/runtime/async_grpc

# 启动gRPC服务（后台运行，使用绝对路径）
echo "🎯 启动gRPC服务..."
python /app/async_cosyvoice/runtime/async_grpc/server.py --load_jit --load_trt --fp16 --model_dir /app/models/CosyVoice2-0.5B &
GRPC_PID=$!

# 等待gRPC服务启动
echo "⏳ 等待gRPC服务启动..."
sleep 30

# 检查gRPC服务是否正常运行
if ! kill -0 $GRPC_PID 2>/dev/null; then
    echo "❌ gRPC服务启动失败"
    exit 1
fi

echo "✅ gRPC服务启动成功 (PID: $GRPC_PID)"

# 启动FastAPI服务（前台运行，使用绝对路径）
echo "🌐 启动FastAPI服务..."
python /app/async_cosyvoice/runtime/async_grpc/client_fastapi.py

# 如果FastAPI退出，也清理gRPC进程
kill $GRPC_PID 2>/dev/null || true