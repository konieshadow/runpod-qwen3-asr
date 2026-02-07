import runpod
import torch
import os
from qwen_asr import Qwen3ASRModel
from utils import download_audio, split_audio, cleanup_files

# --- 配置 ---
# 可以根据需要修改为 "Qwen/Qwen3-ASR-1.7B"
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B" 
ALIGNER_NAME = "Qwen/Qwen3-ForcedAligner-0.6B"
# 模型下载目录 (与 Dockerfile 中一致)
MODEL_DIR = "/models/asr"
ALIGNER_DIR = "/models/aligner"

# 全局变量缓存模型
model = None

def init_model():
    global model
    if model is None:
        print("🚀 Loading Qwen3-ASR Model with vLLM backend...")
        try:
            # 使用 vLLM 后端加载，通过 GPU 加速
            model = Qwen3ASRModel.LLM(
                model=MODEL_DIR,
                gpu_memory_utilization=0.7, # 根据显卡显存调整，0.7 适合 24GB 显卡同时跑其他任务
                max_new_tokens=4096,
                forced_aligner=ALIGNER_DIR,
                forced_aligner_kwargs={
                    "dtype": torch.bfloat16,
                    "device_map": "cuda:0",
                },
            )
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

def handler(job):
    """
    RunPod 处理函数
    输入格式: {"input": {"audio_url": "https://...", "language": "auto"}}
    """
    job_input = job["input"]
    audio_url = job_input.get("audio_url")
    language = job_input.get("language", None) # None 为自动检测

    if not audio_url:
        return {"error": "Missing 'audio_url' in input."}

    # 1. 准备环境
    temp_id = job["id"]
    local_audio_path = f"/tmp/{temp_id}_raw.mp3"
    chunk_dir = f"/tmp/{temp_id}_chunks"
    
    try:
        # 2. 下载音频
        print(f"⬇️ Downloading audio from {audio_url}...")
        download_audio(audio_url, local_audio_path)

        # 3. 切片音频 (规避 ForcedAligner 的 5 分钟限制)
        # 将音频切分为 4.5 分钟 (270秒) 的片段
        print("✂️ Splitting audio into chunks...")
        chunks_info = split_audio(local_audio_path, chunk_dir, chunk_length_ms=270000)
        
        full_transcript = []
        full_text = ""
        
        # 4. 逐个片段转录
        # 注意：这里是串行处理片段。如果追求极致速度，可以使用 ThreadPoolExecutor 并行提交给 vLLM
        # 但考虑到时间戳合并的顺序性，串行更容易维护。
        print(f"🔄 Processing {len(chunks_info)} chunks...")
        
        for idx, chunk in enumerate(chunks_info):
            chunk_path = chunk["path"]
            time_offset = chunk["start_time_sec"]
            
            # 调用模型转录
            # Qwen3-ASR 的 transcribe 支持直接传入文件路径
            results = model.transcribe(
                audio=chunk_path,
                language=language,
                return_time_stamps=True
            )
            
            res = results[0] # 单文件处理
            
            # 合并文本
            full_text += res.text + " "
            
            # 调整时间戳 (加上当前切片的偏移量)
            # ForcedAligner 返回的 timestamps 结构通常是List[List[float]] 或者 List[Dict]
            # Qwen3 返回的是对象，我们提取 raw data
            if res.time_stamps:
                for segment in res.time_stamps:
                    # 假设 segment 是 [start, end, text] 或类似结构，根据实际输出调整
                    # 打印一下结构以防万一
                    # 调整时间
                    adjusted_segment = {
                        "start": segment[0] + time_offset,
                        "end": segment[1] + time_offset,
                        "text": segment[2] if len(segment) > 2 else ""
                    }
                    full_transcript.append(adjusted_segment)

        return {
            "status": "success",
            "text": full_text.strip(),
            "segments": full_transcript,
            "language_detected": results[0].language # 返回最后一个片段检测到的语言作为参考
        }

    except Exception as e:
        print(f"❌ Error processing job: {e}")
        return {"error": str(e)}
        
    finally:
        # 5. 清理临时文件
        cleanup_files([local_audio_path, chunk_dir])

# 初始化模型
init_model()

# 启动 Serverless 监听
runpod.serverless.start({"handler": handler})