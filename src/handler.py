import runpod
import torch
import os
from qwen_asr import Qwen3ASRModel
from utils import download_audio, split_audio_smart, cleanup_files

# --- Configuration ---
# Can be modified to "Qwen/Qwen3-ASR-1.7B" as needed
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B" 
ALIGNER_NAME = "Qwen/Qwen3-ForcedAligner-0.6B"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保这里的路径与 fetch_models.py 中的一致
MODEL_DIR = os.path.join(BASE_DIR, "models", "asr")
ALIGNER_DIR = os.path.join(BASE_DIR, "models", "aligner")

# Global variable to cache model
model = None

def init_model():
    global model
    if model is None:
        print("🚀 Loading Qwen3-ASR Model with vLLM backend...")
        try:
            # Load using vLLM backend for GPU acceleration
            model = Qwen3ASRModel.LLM(
                model=MODEL_DIR,
                gpu_memory_utilization=0.8, # Adjust based on GPU memory, 0.8 is suitable for 24GB GPUs running other tasks
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
    """Clear vLLM's KV cache to prevent memory leaks during long audio processing"""
    global model
    if model is not None and hasattr(model, 'llm'):
        try:
            # vLLM 0.3.0+ supports reset_prefix_cache
            if hasattr(model.llm, 'reset_prefix_cache'):
                model.llm.reset_prefix_cache()
        except Exception as e:
            print(f"⚠️ Warning: Failed to clear KV cache: {e}")


def _parse_timestamp_segment(segment, time_offset):
    """
    Parse timestamp segment, safely handling multiple possible return formats
    
    Args:
        segment: May be list [start, end, text] or dict {"start": x, "end": y, "text": z}
                 or ForcedAlignItem(text=..., start_time=..., end_time=...)
        time_offset: Time offset in seconds
        
    Returns:
        dict: {"start": float, "end": float, "text": str}
    """
    try:
        if isinstance(segment, dict):
            # Dict format
            start = segment.get("start", segment.get(0, 0))
            end = segment.get("end", segment.get(1, 0))
            text = segment.get("text", segment.get(2, ""))
        elif isinstance(segment, (list, tuple)) and len(segment) >= 2:
            # List format [start, end, text?]
            start = segment[0]
            end = segment[1]
            text = segment[2] if len(segment) > 2 else ""
        elif hasattr(segment, 'start_time') and hasattr(segment, 'end_time') and hasattr(segment, 'text'):
            # ForcedAlignItem object format
            start = segment.start_time
            end = segment.end_time
            text = segment.text
        else:
            # Unknown format, try to parse
            print(f"⚠️ Warning: Unknown timestamp segment format: {type(segment)} - {repr(segment)}")
            start = end = 0
            text = str(segment) if segment is not None else ""
        
        # Validate numeric values
        start = float(start) if start is not None else 0.0
        end = float(end) if end is not None else 0.0
        
        return {
            "start": start + time_offset,
            "end": end + time_offset,
            "text": str(text) if text is not None else ""
        }
    except Exception as e:
        # Return default values for any parsing errors, don't interrupt processing
        print(f"⚠️ Warning: Failed to parse timestamp segment: {e}")
        return {"start": time_offset, "end": time_offset, "text": ""}


def _sanitize_job_id(job_id):
    """Sanitize job ID to make it suitable for use as filename"""
    import re
    # Remove non-alphanumeric characters, limit length
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(job_id))
    return sanitized[:64]  # Limit length


def handler(job):
    """
    RunPod handler function
    Input format: {"input": {"audio_url": "https://...", "language": "auto"}}
    """
    job_input = job["input"]
    audio_url = job_input.get("audio_url")
    language = job_input.get("language", None)
    initial_context = job_input.get("initial_context", None)
    use_previous_context = job_input.get("use_previous_context", False)

    # Convert "auto" to None because model doesn't support "auto" string, use None for auto-detection
    if isinstance(language, str) and language.lower() == "auto":
        language = None

    if not audio_url:
        return {"error": "Missing 'audio_url' in input."}

    # 1. Prepare environment
    safe_job_id = _sanitize_job_id(job.get("id", "unknown"))
    local_audio_path = f"/tmp/{safe_job_id}_raw.mp3"
    chunk_dir = f"/tmp/{safe_job_id}_chunks"
    
    try:
        # 2. Download audio (with timeout)
        print(f"⬇️ Downloading audio from {audio_url}...")
        download_audio(audio_url, local_audio_path, timeout=300)

        # 2.5 Print file info and validate audio file
        print("📊 Checking audio file info...")
        try:
            from pydub import AudioSegment
            import os

            # Get file size
            file_size = os.path.getsize(local_audio_path)
            print(f"  📁 File size: {file_size / 1024 / 1024:.2f} MB ({file_size} bytes)")

            # Load and check audio info
            audio = AudioSegment.from_file(local_audio_path)
            duration_sec = len(audio) / 1000.0
            sample_rate = audio.frame_rate
            channels = audio.channels
            bit_depth = audio.sample_width * 8

            print(f"  ⏱️  Duration: {duration_sec:.2f} seconds ({duration_sec / 60:.2f} minutes)")
            print(f"  🔊 Sample rate: {sample_rate} Hz")
            print(f"  🎚️  Channels: {channels} ({'mono' if channels == 1 else 'stereo'})")
            print(f"  🎵 Bit depth: {bit_depth} bits")

            # Basic validation
            if file_size == 0:
                raise ValueError("Downloaded audio file is empty (0 bytes)")

            if duration_sec < 0.1:
                raise ValueError(f"Audio duration too short: {duration_sec:.2f}s (minimum 0.1s)")

            if duration_sec > 3600 * 4:  # 4 hours
                raise ValueError(f"Audio duration too long: {duration_sec / 3600:.2f} hours (maximum 4 hours)")

            print("✅ Audio file validation passed")

        except Exception as e:
            print(f"❌ Audio file validation failed: {e}")
            raise RuntimeError(f"Audio file validation failed: {e}")

        # 3. Split audio intelligently using VAD
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

        # Initialize context: use initial_context or None
        current_context = initial_context if initial_context else None

        # 4. Transcribe chunks one by one
        print(f"🔄 Processing {len(chunks_info)} chunks...")

        for idx, chunk in enumerate(chunks_info):
            chunk_path = chunk["path"]
            time_offset = chunk["start_time_sec"]
            chunk_duration = chunk['end_time_sec'] - chunk['start_time_sec']
            
            # Skip audio chunks that are too short (less than 0.5 seconds) to avoid forced alignment errors
            if chunk_duration < 0.5:
                print(f"  ⏭️ Skipping chunk {idx + 1}/{len(chunks_info)} (too short: {chunk_duration:.2f}s)")
                continue
            
            print(f"  📝 Processing chunk {idx + 1}/{len(chunks_info)} ({chunk['start_time_sec']:.1f}s - {chunk['end_time_sec']:.1f}s)...")
            
            try:
                # Call model for transcription with current context
                results = model.transcribe(
                    audio=chunk_path,
                    language=language,
                    return_time_stamps=True,
                    context=current_context
                )
            except Exception as e:
                # If transcription fails (e.g., empty audio causing forced alignment error), log warning and skip this chunk
                print(f"  ⚠️ Warning: Failed to transcribe chunk {idx + 1}: {e}")
                continue
            
            res = results[0]
            
            # Handle different return formats (object or dict)
            # Get text
            text = res.text if hasattr(res, 'text') else res.get('text', '') if isinstance(res, dict) else ''
            if text:
                full_text += text + " "
            
            # Record detected language
            language_val = None
            if hasattr(res, 'language'):
                language_val = res.language
            elif isinstance(res, dict) and 'language' in res:
                language_val = res['language']
            if language_val:
                last_detected_language = language_val
            
            # Get timestamp data (supports time_stamps or chunks fields)
            timestamps_data = None
            if hasattr(res, 'time_stamps') and res.time_stamps:
                timestamps_data = res.time_stamps
            elif hasattr(res, 'chunks') and res.chunks:
                timestamps_data = res.chunks
            elif isinstance(res, dict):
                if res.get('time_stamps'):
                    timestamps_data = res['time_stamps']
                elif res.get('chunks'):
                    timestamps_data = res['chunks']
            
            # Adjust timestamps and merge
            if timestamps_data:
                for segment in timestamps_data:
                    adjusted_segment = _parse_timestamp_segment(segment, time_offset)
                    full_transcript.append(adjusted_segment)

            # If use_previous_context is enabled, use current chunk's text as context for next chunk
            if use_previous_context and text:
                # Take last 200 characters of current chunk text as context for next chunk
                # This length can be adjusted based on model's maximum supported context length
                current_context = text.strip()[-200:] if len(text) > 200 else text.strip()

            # Clear KV cache every 3 chunks processed to prevent OOM
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
        
        # Print detailed error log to server
        error_trace = traceback.format_exc()
        print(f"❌ Error processing job: {e}")
        print(error_trace)
        
        # Error message returned to client (hiding sensitive details)
        error_type = type(e).__name__
        
        # Return user-friendly messages based on error type
        if "Download timeout" in str(e):
            user_message = "Audio download timed out. Please check the URL and try again."
        elif "Failed to download" in str(e):
            user_message = "Failed to download audio from the provided URL."
        elif "CUDA" in str(e) or "cuda" in str(e):
            user_message = "GPU processing error. The service may be temporarily overloaded."
        elif "OutOfMemory" in error_type or "No memory" in str(e):
            user_message = "Audio too long or complex to process. Please try a shorter audio file."
        else:
            # Generic error, don't expose internal details
            user_message = f"Processing error ({error_type}). Please try again later."
        
        return {
            "error": user_message,
            "error_type": error_type,
            "job_id": safe_job_id if 'safe_job_id' in locals() else "unknown"
        }
        
    finally:
        # 5. Clean up temporary files
        cleanup_files([local_audio_path, chunk_dir])

if __name__ == "__main__":
    # 初始化模型
    init_model()

    # 启动 Serverless 监听
    runpod.serverless.start({"handler": handler})