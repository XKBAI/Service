# start_cosyvoice.sh（单容器版本）
#!/bin/bash

set -e

echo "🚀 启动CosyVoice统一服务..."

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA not available. Please install NVIDIA drivers."
    exit 1
fi

# 检查模型文件是否存在（修正路径）
MODEL_DIR="/home/xkb2/Desktop/QY/CosyVoice/pretrained_models/CosyVoice2-0.5B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    echo "Please make sure the model files are in the correct location."
    exit 1
fi

# 特别检查关键配置文件
CONFIG_FILE="$MODEL_DIR/cosyvoice2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Model config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✅ 模型文件检查通过: $MODEL_DIR"

# 创建必要的目录（使用绝对路径）
mkdir -p /home/xkb2/Desktop/QY/CosyVoice/async_cosyvoice/runtime/async_grpc/audio_output
mkdir -p /home/xkb2/Desktop/QY/CosyVoice/async_cosyvoice/runtime/async_grpc/shared-data

# 检查docker-compose是否存在
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# 进入正确的工作目录（使用绝对路径）
cd /home/xkb2/Desktop/QY/CosyVoice/async_cosyvoice/runtime/async_grpc

# 构建并启动服务
echo "📦 构建Docker镜像..."
docker-compose build

echo "🎯 启动统一服务..."
docker-compose up -d

echo "⏳ 等待服务启动完成..."
sleep 60

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 检查服务日志
echo "🔍 检查服务日志..."
docker-compose logs cosyvoice-all-in-one | tail -20

echo "✅ 服务启动完成！"
echo "📊 FastAPI文档: http://localhost:55032/docs"
echo "💡 健康检查: http://localhost:55032/health"
echo "🎤 直接gRPC: localhost:50000"
echo "🎤 测试API: curl -X POST http://localhost:55032/synthesize -H 'Content-Type: application/json' -d '{\"text\":\"你好世界\",\"spk_id\":\"001\"}'"