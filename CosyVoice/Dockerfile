# Dockerfile（修正版）
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 设置Python别名
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 3. 先复制requirements.txt（使用绝对路径）
COPY async_cosyvoice/requirements.txt /app/requirements.txt

ENV http_proxy=http://192.168.2.222:1080
ENV https_proxy=http://192.168.2.222:1080

# 4. 安装Python依赖
RUN pip install -r /app/requirements.txt

# 5. 安装torch和torchaudio
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 6. 创建必要目录
RUN mkdir -p /app/models /app/audio_output

# 7. 复制原始cosyvoice包（重要！）
COPY cosyvoice /app/cosyvoice

# 8. 复制async_cosyvoice包（保持目录结构，使用绝对容器路径）
COPY async_cosyvoice /app/async_cosyvoice

# 9. 复制third_party目录（如果需要的话）
COPY third_party /app/third_party

# 10. 处理启动脚本（使用绝对容器路径）
RUN chmod +x /app/async_cosyvoice/runtime/async_grpc/start_services.sh

# 11. 设置工作目录（绝对路径）
WORKDIR /app/async_cosyvoice/runtime/async_grpc

ENV http_proxy=
ENV https_proxy=

EXPOSE 50000 9000

# 12. 启动服务（使用绝对容器路径）
CMD ["/app/async_cosyvoice/runtime/async_grpc/start_services.sh"]