FROM gcr.io/ai2-beaker-core/public/cqgl31u2ba5vrtuc91og:latest

# Update the package list and install libaio-dev and gnupg2
RUN apt update && apt-get install -y libaio-dev gnupg2

# Add NVIDIA package repository keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get -y update

# Set up the NVIDIA CUDA repository
RUN apt-get install -y software-properties-common \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
    && apt-get update

# Install CUDA toolkit and nvcc 12.1
RUN apt-get install -y cuda-nvcc-12-1

# Get flash attention setup
RUN pip install flash-attn --no-build-isolation

# Install PDF utilities
RUN apt-get install -y poppler-utils
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get install -y ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

