# Dockerfile (针对OLMOCR API优化 - 使用NVIDIA CUDA基础镜像)

# 1. 使用NVIDIA官方CUDA镜像，预装CUDA工具链
FROM ubuntu:22.04

# ARG HTTP_PROXY
# ARG HTTPS_PROXY
ARG NO_PROXY

# 2. 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app:$PYTHONPATH
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
# 确保Hugging Face缓存路径一致
ENV HF_HOME=/root/.cache/huggingface
# 确保Transformers缓存路径一致
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
# CUDA相关环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 安装系统基础依赖和工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    wget \
    bzip2 \
    build-essential \
    git \
    curl \
    poppler-utils \
    libpoppler-dev \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 4. 升级pip和安装核心依赖
RUN python3 -m pip install --upgrade pip setuptools wheel

# 5. 复制并安装项目所需的 Python 依赖
# 优化缓存：只有当 requirements.txt 变化时，才会重新执行 pip install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8. 设置工作目录
WORKDIR /app

# 9. 复制所有应用程序文件
COPY pyproject.toml .
COPY olmocr/ ./olmocr/
COPY olmocr_api.py .
COPY start.sh .

# 10. 安装项目本身（可编辑安装）
RUN python3 -m pip install -e . --no-deps

# 11. 创建必要的目录
RUN mkdir -p /app/uploads /app/results /app/model /app/shared /tmp/melotts_audio \
    /root/.cache/huggingface /root/.cache/torch /app/logs

# 12. 设置权限
RUN chmod -R 755 /app
RUN chmod +x /app/start.sh

# 13. 暴露端口 - 根据olmocr_api.py中的实际端口配置
EXPOSE 9000

# 14. 健康检查 - 检查实际运行的端点
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:9000/olmocr/ || exit 1

# 15. 环境变量 - 设置OCR API服务端口
ENV OCR_PORT=9000
ENV OCR_HOST=0.0.0.0

# 16. 启动命令 - 直接运行olmocr_api.py
CMD ["python3", "olmocr_api.py"]