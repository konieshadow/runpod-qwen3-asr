from huggingface_hub import snapshot_download

# Model download directory (consistent with Dockerfile)
ASR_MODEL_DIR = "/models/asr"
ALIGNER_MODEL_DIR = "/models/aligner"

# Define HuggingFace model IDs
# If you want to use 1.7B, change this to "Qwen/Qwen3-ASR-1.7B"
ASR_REPO_ID = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_REPO_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

def fetch_models():
    print(f"⏳ Downloading ASR Model: {ASR_REPO_ID}...")
    snapshot_download(
        repo_id=ASR_REPO_ID,
        local_dir=ASR_MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Ignore non-PyTorch weights
    )
    
    print(f"⏳ Downloading Aligner Model: {ALIGNER_REPO_ID}...")
    snapshot_download(
        repo_id=ALIGNER_REPO_ID,
        local_dir=ALIGNER_MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Ignore non-PyTorch weights
    )
    
    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    fetch_models()