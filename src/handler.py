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

def _clear_kv_cache():
    """清理 vLLM 的 KV cache，防止长音频处理时显存泄漏"""
    global model
    if model is not None and hasattr(model, 'llm'):
        try:
            # vLLM 0.3.0+ 支持 reset_prefix_cache
            if hasattr(model.llm, 'reset_prefix_cache'):
                model.llm.reset_prefix_cache()
        except Exception as e:
            print(f"⚠️ Warning: Failed to clear KV cache: {e}")


def _parse_timestamp_segment(segment, time_offset):
    """
    解析时间戳段，安全处理多种可能的返回格式
    
    Args:
        segment: 可能是 list [start, end, text] 或 dict {"start": x, "end": y, "text": z}
        time_offset: 时间偏移量（秒）
        
    Returns:
        dict: {"start": float, "end": float, "text": str}
    """
    try:
        if isinstance(segment, dict):
            # 字典格式
            start = segment.get("start", segment.get(0, 0))
            end = segment.get("end", segment.get(1, 0))
            text = segment.get("text", segment.get(2, ""))
        elif isinstance(segment, (list, tuple)) and len(segment) >= 2:
            # 列表格式 [start, end, text?]
            start = segment[0]
            end = segment[1]
            text = segment[2] if len(segment) > 2 else ""
        else:
            # 未知格式，尝试解析
            print(f"⚠️ Warning: Unknown timestamp segment format: {type(segment)} - {repr(segment)}")
            start = end = 0
            text = str(segment) if segment is not None else ""
        
        # 验证数值有效性
        start = float(start) if start is not None else 0.0
        end = float(end) if end is not None else 0.0
        
        return {
            "start": start + time_offset,
            "end": end + time_offset,
            "text": str(text) if text is not None else ""
        }
    except Exception as e:
        # 任何解析错误都返回默认值，不中断处理
        print(f"⚠️ Warning: Failed to parse timestamp segment: {e}")
        return {"start": time_offset, "end": time_offset, "text": ""}


def _sanitize_job_id(job_id):
    """清理 job ID，确保适合用作文件名"""
    import re
    # 移除非字母数字字符，限制长度
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(job_id))
    return sanitized[:64]  # 限制长度


def handler(job):
    """
    RunPod 处理函数
    输入格式: {"input": {"audio_url": "https://...", "language": "auto"}}
    """
    job_input = job["input"]
    audio_url = job_input.get("audio_url")
    language = job_input.get("language", None)  # None 为自动检测

    if not audio_url:
        return {"error": "Missing 'audio_url' in input."}

    # 1. 准备环境
    safe_job_id = _sanitize_job_id(job.get("id", "unknown"))
    local_audio_path = f"/tmp/{safe_job_id}_raw.mp3"
    chunk_dir = f"/tmp/{safe_job_id}_chunks"
    
    try:
        # 2. 下载音频（带超时）
        print(f"⬇️ Downloading audio from {audio_url}...")
        download_audio(audio_url, local_audio_path, timeout=300)

        # 3. 使用 VAD 智能切分音频
        print("✂️ Splitting audio into chunks using VAD...")
        chunks_info = split_audio_smart(
            local_audio_path, 
            chunk_dir, 
            max_chunk_ms=270000,
            min_silence_ms=300
        )
        print(f"📦 Audio split into {len(chunks_info)} chunks")
        
        full_transcript = []
        full_text = ""
        last_detected_language = None
        
        # 4. 逐个片段转录
        print(f"🔄 Processing {len(chunks_info)} chunks...")
        
        for idx, chunk in enumerate(chunks_info):
            chunk_path = chunk["path"]
            time_offset = chunk["start_time_sec"]
            
            print(f"  📝 Processing chunk {idx + 1}/{len(chunks_info)} ({chunk['start_time_sec']:.1f}s - {chunk['end_time_sec']:.1f}s)...")
            
            # 调用模型转录
            results = model.transcribe(
                audio=chunk_path,
                language=language,
                return_time_stamps=True
            )
            
            res = results[0]
            
            # 合并文本
            if hasattr(res, 'text'):
                full_text += res.text + " "
            
            # 记录检测到的语言
            if hasattr(res, 'language'):
                last_detected_language = res.language
            
            # 调整时间戳并合并
            if hasattr(res, 'time_stamps') and res.time_stamps:
                for segment in res.time_stamps:
                    adjusted_segment = _parse_timestamp_segment(segment, time_offset)
                    full_transcript.append(adjusted_segment)
            
            # 每处理 3 个片段清理一次 KV cache，防止 OOM
            if (idx + 1) % 3 == 0:
                _clear_kv_cache()

        return {
            "status": "success",
            "text": full_text.strip(),
            "segments": full_transcript,
            "language_detected": last_detected_language
        }

    except Exception as e:
        import traceback
        import os
        
        # 打印详细错误日志到服务端
        error_trace = traceback.format_exc()
        print(f"❌ Error processing job: {e}")
        print(error_trace)
        
        # 返回给客户端的错误信息（隐藏敏感细节）
        error_type = type(e).__name__
        
        # 根据错误类型返回用户友好的消息
        if "Download timeout" in str(e):
            user_message = "Audio download timed out. Please check the URL and try again."
        elif "Failed to download" in str(e):
            user_message = "Failed to download audio from the provided URL."
        elif "CUDA" in str(e) or "cuda" in str(e):
            user_message = "GPU processing error. The service may be temporarily overloaded."
        elif "OutOfMemory" in error_type or "No memory" in str(e):
            user_message = "Audio too long or complex to process. Please try a shorter audio file."
        else:
            # 通用错误，不暴露内部细节
            user_message = f"Processing error ({error_type}). Please try again later."
        
        return {
            "error": user_message,
            "error_type": error_type,
            "job_id": safe_job_id if 'safe_job_id' in locals() else "unknown"
        }
        
    finally:
        # 5. 清理临时文件
        cleanup_files([local_audio_path, chunk_dir])

# 初始化模型
init_model()

# 启动 Serverless 监听
runpod.serverless.start({"handler": handler})