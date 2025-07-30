FROM condaforge/miniforge3:latest

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    g++ \
    gcc \
    cmake && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# 设置工作目录
WORKDIR /app

# 配置 pip 镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 复制 requirements 文件
COPY requirements.txt .

# 升级 pip 并安装依赖（使用 --no-build-isolation）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 复制应用程序代码
COPY . .

# 创建必要目录
RUN mkdir -p /app/shared /app/logs && \
    chmod -R 755 /app 
# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动命令
CMD ["python", "/app/new_tools/query_api.py"]