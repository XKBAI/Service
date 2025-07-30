#!/bin/bash

# MD2PDF服务启动脚本

set -e

echo "开始启动MD2PDF服务..."

# 检查必要的目录
mkdir -p /app/storage/input_md /app/storage/output_pdf /app/shared /app/logs

# 设置权限
chmod -R 755 /app/storage /app/shared /app/logs

# 检查pandoc是否可用
if ! command -v pandoc &> /dev/null; then
    echo "错误: pandoc未找到"
    exit 1
fi

echo "Pandoc版本: $(pandoc --version | head -1)"

echo "启动MD2PDF API服务..."

# 启动服务
exec python app.py