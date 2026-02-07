import os
from huggingface_hub import snapshot_download

# 定义模型保存路径 (必须与 Dockerfile 和 handler.py 一致)
ASR_MODEL_DIR = "/models/asr"
ALIGNER_MODEL_DIR = "/models/aligner"

# 定义 HuggingFace 模型 ID
# 如果你想用 1.7B，请修改这里为 "Qwen/Qwen3-ASR-1.7B"
ASR_REPO_ID = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_REPO_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

def fetch_models():
    print(f"⏳ Downloading ASR Model: {ASR_REPO_ID}...")
    snapshot_download(
        repo_id=ASR_REPO_ID,
        local_dir=ASR_MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略非 PyTorch 权重
    )
    
    print(f"⏳ Downloading Aligner Model: {ALIGNER_REPO_ID}...")
    snapshot_download(
        repo_id=ALIGNER_REPO_ID,
        local_dir=ALIGNER_MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    
    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    fetch_models()