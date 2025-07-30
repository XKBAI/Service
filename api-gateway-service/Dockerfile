# 使用本地已有的miniforge3镜像
FROM condaforge/miniforge3:latest

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# 安装系统依赖（使用mamba/conda）
RUN mamba install -y \
    curl \
    wget \
    && mamba clean -a

# 设置工作目录
WORKDIR /app

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 复制应用文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/shared /app/logs

# 设置权限
RUN chmod -R 755 /app
RUN chmod +x /app/start.sh

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动命令
CMD ["./start.sh"] 