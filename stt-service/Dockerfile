# 使用PyTorch官方CUDA镜像
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# 设置工作目录
WORKDIR /app

# 从构建上下文复制requirements.txt并安装Python依赖
# 优化：先复制requirements.txt，利用Docker层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# RUN pip install nvidia-cudnn-cu12==9.1.0.70

# 复制应用程序代码
# 确保finaltest-api.py和所有其他Python文件被复制
# 如果你的程序入口文件名称固定，建议显式复制
# COPY finaltest-api.py .
# COPY your_other_files.py .
# 或者如果你确定所有 .py 文件都在根目录，可以继续使用：
COPY *.py .

# 创建模型缓存目录和共享数据目录，并设置正确权限
RUN mkdir -p /tmp/faster_whisper /app/shared && \
    chmod -R 777 /tmp/faster_whisper /app/shared # 确保 appuser 可以写入

# 修复Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/ctranslate2.libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH


# 创建非root用户（安全最佳实践）
# 确保工作目录及其内容归新用户所有
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser && \
    chown -R appuser:appuser /app /tmp/faster_whisper
USER appuser

# 暴露应用程序端口
EXPOSE 57001

# 健康检查：确保应用程序在容器内运行并响应HTTP请求
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:57001/stt/health || exit 1

# 启动命令：指定Python解释器和入口文件
CMD ["python", "finaltest-api.py"]