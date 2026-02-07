# 使用 RunPod 官方的基础镜像，包含 CUDA 12.4.1 (支持 vLLM)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 1. 安装系统依赖 (FFmpeg 对音频处理至关重要)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件并安装 Python 包
COPY requirements.txt .
# 先安装基础依赖，再安装 Flash Attention (编译需要时间)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install flash-attn --no-build-isolation

# 3. 预下载模型 (这一步会显著增加构建时间，但减少冷启动时间)
COPY builder/ builder/
RUN python builder/fetch_models.py

# 4. 复制源代码
COPY src/ .

# 5. 设置启动命令
CMD ["python", "-u", "handler.py"]