#!/bin/bash

# OLMOCR API 服务启动脚本

set -e

echo "开始启动OLMOCR API服务..."

# 设置默认环境变量
export OCR_PORT=${OCR_PORT:-9000}
export OCR_HOST=${OCR_HOST:-"0.0.0.0"}

# 检查并创建必要的目录
echo "检查必要目录..."
mkdir -p /app/uploads /app/results /app/model /app/shared
mkdir -p /root/.cache/huggingface /root/.cache/torch

# 设置权限
chmod -R 755 /app/uploads /app/results /app/model /app/shared

# 等待其他依赖服务启动（如果需要）
echo "等待依赖服务启动..."
sleep 10

# 检查Python环境
echo "检查Python环境..."
python3 --version
python3 -c "import fastapi, uvicorn, pydantic; print('✅ FastAPI环境正常')"

# 检查OLMOCR模块
echo "检查OLMOCR模块..."
python3 -c "from olmocr.pipeline import check_poppler_version; print('✅ OLMOCR模块导入正常')" || echo "⚠️ OLMOCR模块导入可能有问题"

# 启动API服务
echo "启动OLMOCR API服务..."
echo "监听地址: ${OCR_HOST}:${OCR_PORT}"

# 使用exec确保信号能正确传递到Python进程
exec python3 olmocr_api.py 