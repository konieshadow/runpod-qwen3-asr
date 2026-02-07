import os
import requests
import shutil
import webrtcvad
import collections
import contextlib
from pydub import AudioSegment

def download_audio(url, save_path, timeout=300):
    """从 URL 下载音频到本地路径
    
    Args:
        url: 音频文件 URL
        save_path: 本地保存路径
        timeout: 下载超时时间（秒），默认 5 分钟
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.Timeout:
        raise RuntimeError(f"Download timeout after {timeout}s: {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}")

class WebRTCVADHelper:
    """WebRTC VAD 辅助类
    
    注意：WebRTC VAD 只支持 8kHz、16kHz、32kHz 采样率。
    对于原始音频采样率高于 32kHz 的情况（如 48kHz），重采样到 16kHz
    会丢失 8kHz 以上的高频信息。这可能会影响 VAD 对高频辅音的检测，
    但对于播客语音场景通常影响不大。
    
    如果需要保留高频信息，可以考虑：
    1. 使用 Silero VAD（支持任意采样率）
    2. 将音频先降采样到 16kHz 用于 VAD，但保留原始音频用于 ASR
    """
    
    def __init__(self, aggressiveness=2):
        """
        Args:
            aggressiveness: 0-3，越高越激进（把更多内容标记为语音）
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000  # WebRTC VAD 只支持 8k, 16k, 32k
        self.frame_duration = 30  # ms，可选 10, 20, 30
        
    def _read_frames(self, audio_data):
        """将音频数据切分为 VAD 帧"""
        frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2  # 16-bit = 2 bytes
        offset = 0
        while offset + frame_size <= len(audio_data):
            yield audio_data[offset:offset + frame_size]
            offset += frame_size
            
    def detect_speech_regions(self, audio_segment):
        """
        检测音频中的语音区域
        
        Args:
            audio_segment: pydub AudioSegment（任意采样率，会被重采样到 16kHz）
            
        Returns:
            List[Tuple[int, int]]: 语音区域列表 [(start_ms, end_ms), ...]，时间戳对应原始音频
        """
        # 记录原始音频的帧率，用于后续时间戳转换
        original_frame_rate = audio_segment.frame_rate
        
        # 转换为 16kHz mono PCM 用于 VAD
        # 注意：对于高于 16kHz 的音频，这会丢失高频信息
        audio = audio_segment.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data
        
        # 逐帧检测
        regions = []
        current_region_start = None
        frame_idx = 0
        
        for frame in self._read_frames(raw_data):
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            timestamp_ms = frame_idx * self.frame_duration
            
            if is_speech and current_region_start is None:
                # 语音开始
                current_region_start = timestamp_ms
            elif not is_speech and current_region_start is not None:
                # 语音结束
                regions.append((current_region_start, timestamp_ms))
                current_region_start = None
                
            frame_idx += 1
        
        # 处理最后一个区域（如果音频结束时还在说话）
        if current_region_start is not None:
            regions.append((current_region_start, frame_idx * self.frame_duration))
            
        return regions


def _find_best_split_in_range(silence_regions, start_ms, end_ms, target_ms):
    """
    在指定范围内找到最佳的切分点
    
    优先选择最接近 target_ms 的停顿处切分，以尽量保持语义完整性
    
    Args:
        silence_regions: 所有停顿区域列表 [(start, end), ...]
        start_ms: 当前片段起始位置
        end_ms: 当前片段结束位置（不能超过的最大位置）
        target_ms: 理想切分位置（通常是 start_ms + max_chunk_ms）
        
    Returns:
        int: 最佳切分点位置（毫秒）
    """
    # 筛选在当前范围内的停顿
    valid_silences = [
        (s, e) for s, e in silence_regions
        if s >= start_ms and e <= end_ms and s > start_ms  # 停顿必须在当前范围内且不在起点
    ]
    
    if not valid_silences:
        # 没有合适的停顿，被迫在最大长度处切分
        return min(start_ms + 270000, end_ms)
    
    # 找最接近 target 的停顿中点
    best_split = None
    best_distance = float('inf')
    
    for s, e in valid_silences:
        # 使用停顿的中点作为切分点
        midpoint = (s + e) // 2
        distance = abs(midpoint - target_ms)
        
        if distance < best_distance:
            best_distance = distance
            best_split = midpoint
    
    return best_split if best_split is not None else min(start_ms + 270000, end_ms)


def find_split_points(audio_segment, max_chunk_ms=270000, min_silence_ms=300):
    """
    基于 VAD 的智能切分点查找
    
    策略：
    1. 使用 VAD 检测语音区域
    2. 优先在语音区域之间的停顿处切分，保持语义完整性
    3. 如果当前段落超过 max_chunk_ms，在最近的合适停顿处强制切分
    4. 只有在没有合适停顿时才在句子中间切分
    
    Args:
        audio_segment: pydub AudioSegment
        max_chunk_ms: 最大片段长度（毫秒）
        min_silence_ms: 最小停顿长度（毫秒），低于此值的停顿不视为切分点
        
    Returns:
        List[int]: 切分点列表（毫秒），包含起点 0 和终点 duration
    """
    duration_ms = len(audio_segment)
    
    # 音频过短，无需切分
    if duration_ms <= max_chunk_ms:
        return [0, duration_ms]
    
    # 使用 VAD 检测语音区域
    vad_helper = WebRTCVADHelper(aggressiveness=2)
    speech_regions = vad_helper.detect_speech_regions(audio_segment)
    
    # 如果没有检测到语音，整个音频作为一段（但受最大长度限制）
    if not speech_regions:
        if duration_ms <= max_chunk_ms:
            return [0, duration_ms]
        # 没有语音但过长，在最大长度处切分
        split_points = [0]
        current = max_chunk_ms
        while current < duration_ms:
            split_points.append(min(current, duration_ms))
            current += max_chunk_ms
        if split_points[-1] != duration_ms:
            split_points.append(duration_ms)
        return split_points
    
    # 从语音区域推导出停顿区域
    silence_regions = []
    
    # 开头的静音
    if speech_regions[0][0] > 0:
        silence_regions.append((0, speech_regions[0][0]))
    
    # 语音之间的静音
    for i in range(len(speech_regions) - 1):
        silence_start = speech_regions[i][1]
        silence_end = speech_regions[i + 1][0]
        if silence_end > silence_start:
            silence_regions.append((silence_start, silence_end))
    
    # 结尾的静音
    if speech_regions[-1][1] < duration_ms:
        silence_regions.append((speech_regions[-1][1], duration_ms))
    
    # 筛选出满足最小停顿长度的静音区域
    valid_silences = [
        (start, end) for start, end in silence_regions 
        if (end - start) >= min_silence_ms
    ]
    
    # 构建切分点
    split_points = [0]
    current_start = 0
    
    for silence_start, silence_end in valid_silences:
        # 计算如果在当前停顿处切分，当前段落的长度
        potential_end = silence_start
        segment_length = potential_end - current_start
        
        if segment_length >= max_chunk_ms:
            # 当前段落已经超过最大长度，需要在最近的合适停顿处强制切分
            best_split = _find_best_split_in_range(
                silence_regions, current_start, silence_start, current_start + max_chunk_ms
            )
            split_points.append(best_split)
            current_start = best_split
        
        # 在当前停顿处切分
        split_points.append(silence_start)
        current_start = silence_end
    
    # 确保最后一个切分点是音频结尾
    if split_points[-1] != duration_ms:
        # 检查最后一段是否过长
        last_segment_length = duration_ms - current_start
        if last_segment_length > max_chunk_ms:
            # 在最后一段内找最佳切分点
            best_split = _find_best_split_in_range(
                silence_regions, current_start, duration_ms, current_start + max_chunk_ms
            )
            split_points.append(best_split)
        split_points.append(duration_ms)
    
    return split_points


def split_audio_smart(file_path, output_dir, max_chunk_ms=270000, min_silence_ms=300):
    """
    基于 VAD 的智能音频切分
    
    与旧版 split_audio 保持接口兼容，但内部使用 VAD 检测停顿位置，
    尽量在句子边界处切分，保持语义完整性。
    
    Args:
        file_path: 源文件路径
        output_dir: 输出目录
        max_chunk_ms: 最大片段长度（毫秒），默认 270s
        min_silence_ms: 最小停顿长度（毫秒），默认 300ms
        
    Returns:
        List[Dict]: 包含切片路径和时间偏移量的列表
        [
            {
                "path": "/tmp/chunks/chunk_0.wav",
                "start_time_sec": 0.0,
                "end_time_sec": 125.5
            },
            ...
        ]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载音频
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    
    # 音频过短，直接返回
    if duration_ms <= max_chunk_ms:
        chunk_path = os.path.join(output_dir, "chunk_0.wav")
        audio.export(chunk_path, format="wav")
        return [{
            "path": chunk_path,
            "start_time_sec": 0.0,
            "end_time_sec": duration_ms / 1000.0
        }]
    
    # 使用 VAD 查找切分点
    split_points = find_split_points(audio, max_chunk_ms, min_silence_ms)
    
    # 导出切片
    chunks_info = []
    for i in range(len(split_points) - 1):
        start_ms = split_points[i]
        end_ms = split_points[i + 1]
        
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        
        chunks_info.append({
            "path": chunk_path,
            "start_time_sec": start_ms / 1000.0,
            "end_time_sec": end_ms / 1000.0
        })
    
    return chunks_info


# 保留旧函数名作为别名，保持向后兼容
split_audio = split_audio_smart

def cleanup_files(paths):
    """清理临时文件和目录"""
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)