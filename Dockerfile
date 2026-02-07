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

# 安装依赖：
# - Qwen3-ASR 0.6B 模型很小，通常不需要 Flash Attention 也能高效运行
# - Flash Attention 编译时间很长（5-15分钟），会显著增加镜像构建时间
# - 如果你的 GPU 需要 Flash Attention 来支持更大的 batch，可以取消下面这行的注释
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt
# 如果需要 Flash Attention，请取消下面这行的注释：
# RUN pip install flash-attn --no-build-isolation

# 3. 预下载模型 (这一步会显著增加构建时间，但减少冷启动时间)
COPY builder/ builder/
RUN python builder/fetch_models.py

# 4. 复制源代码
COPY src/ .
COPY test_input.json .

# 5. 设置启动命令
# 使用 RunPod 推荐的方式启动 worker
# 可以通过环境变量控制并发度：
# - RP_CONCURRENCY: 并发处理数，默认 1
# - RP_THREADS: 线程数，默认 1
CMD ["python", "-u", "handler.py"]