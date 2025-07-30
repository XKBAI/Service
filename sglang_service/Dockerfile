# 使用和其他成功运行服务相同的基础镜像
FROM ubuntu:22.04

# 设置时区，避免安装过程中提示
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 更新包列表并安装必要的工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    build-essential \
    git \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装CUDA开发工具链
# 添加NVIDIA官方CUDA仓库
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb

# 更新包列表并安装CUDA Toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-4 \
    cuda-compiler-12-4 \
    cuda-libraries-dev-12-4 \
    cuda-driver-dev-12-4 \
    && rm -rf /var/lib/apt/lists/*

# 设置CUDA_HOME环境变量 (安装CUDA后设置)
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# 创建符号链接以保持兼容性
RUN ln -sf /usr/local/cuda-12.4 /usr/local/cuda

# 安装 miniconda
WORKDIR /root # 将 miniconda 安装在 /root 下
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# 将 conda 添加到 PATH
ENV PATH /opt/conda/bin:$PATH

# 创建并激活 conda 环境
# 我们会在这个环境中安装 PyTorch 和 sglang
RUN conda create -n sglang python=3.10 -y
# 激活 conda 环境到后续 RUN 命令中，需要特殊的语法
SHELL ["conda", "run", "-n", "sglang", "/bin/bash", "-c"]

# 安装 PyTorch 和 sglang
# 注意：这里的 PyTorch 版本和 CUDA 版本要匹配你上面安装的 CUDA Toolkit
# PyTorch 2.7.0+cu124 (假设有这个版本，或用 cu121 替换，因为 PyTorch 的 whl 包可能没有精确的 cu124)
# 如果 PyTorch 官方没有 cu124 的 whl，你可能需要回退到 cu121 或编译。
# 再次提醒，`--index-url` 的 URL 必须与你的 CUDA 版本相符。
RUN pip install torch --index-url https://download.pytorch.org/whl/nightly/cu121 \
    && pip install sglang[all]==0.4.6.post4 \
    && pip install transformers accelerate pydantic orjson uvicorn uvloop fastapi aiohttp requests numpy # SGlang 完整依赖

# 切换回默认 shell，以便 ENTRYPOINT/CMD 正确执行
SHELL ["/bin/bash", "-c"]

# 设置工作目录回 /app (如果你希望你的应用代码放在这里)
WORKDIR /app

# 暴露端口
EXPOSE 30024

# 启动命令
ENTRYPOINT ["conda", "run", "-n", "sglang", "python3", "-m", "sglang.launch_server"]
CMD ["--served-model-name", "vlm", \
     "--host", "0.0.0.0", \
     "--model-path", "/app/model/olmOCR-7B-0225-preview", \
     "--chat-template", "qwen2-vl", \
     "--port", "30024"]