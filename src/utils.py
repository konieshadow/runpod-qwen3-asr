import os
import requests
import shutil
from pydub import AudioSegment

def download_audio(url, save_path):
    """从 URL 下载音频到本地路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}")

def split_audio(file_path, output_dir, chunk_length_ms=270000):
    """
    将音频切分为指定长度的片段。
    
    Args:
        file_path: 源文件路径
        output_dir: 输出目录
        chunk_length_ms: 切片长度 (毫秒), 默认 270s (4.5分钟)
        
    Returns:
        List[Dict]: 包含切片路径和时间偏移量的列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 加载音频 (pydub 支持 mp3, wav, m4a 等，依赖系统安装 ffmpeg)
    audio = AudioSegment.from_file(file_path)
    
    chunks_info = []
    duration_ms = len(audio)
    
    for i, start_ms in enumerate(range(0, duration_ms, chunk_length_ms)):
        end_ms = min(start_ms + chunk_length_ms, duration_ms)
        chunk = audio[start_ms:end_ms]
        
        # 导出切片
        chunk_filename = f"chunk_{i}.wav" # 统一转为 wav 处理以提高速度
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk.export(chunk_path, format="wav")
        
        chunks_info.append({
            "path": chunk_path,
            "start_time_sec": start_ms / 1000.0,
            "end_time_sec": end_ms / 1000.0
        })
        
    return chunks_info

def cleanup_files(paths):
    """清理临时文件和目录"""
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)