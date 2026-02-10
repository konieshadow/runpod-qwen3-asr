import os
import requests
import shutil
import webrtcvad
import collections
import contextlib
from pydub import AudioSegment

def download_audio(url, save_path, timeout=300, convert_to_wav=True):
    """Download audio from URL to local path with optional format conversion

    Args:
        url: Audio file URL
        save_path: Local save path (if convert_to_wav is True, this should have .wav extension)
        timeout: Download timeout in seconds, default 5 minutes
        convert_to_wav: Whether to convert downloaded audio to WAV format for better compatibility

    Returns:
        Dict: Audio file info containing file_size, duration_sec, sample_rate, channels, bit_depth
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "audio/*,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity;q=1, *;q=0",
        "Referer": url,
    }

    # Download to a temporary file first
    temp_path = save_path + ".tmp"

    try:
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.Timeout:
        raise RuntimeError(f"Download timeout after {timeout}s: {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}")

    # Validate file size
    file_size = os.path.getsize(temp_path)
    if file_size == 0:
        os.remove(temp_path)
        raise RuntimeError("Downloaded audio file is empty (0 bytes)")

    # Load and validate audio format
    try:
        audio = AudioSegment.from_file(temp_path)
    except Exception as e:
        os.remove(temp_path)
        raise RuntimeError(f"Failed to parse audio file: {e}")

    duration_sec = len(audio) / 1000.0
    sample_rate = audio.frame_rate
    channels = audio.channels
    bit_depth = audio.sample_width * 8

    # Validate audio duration
    if duration_sec < 0.1:
        os.remove(temp_path)
        raise RuntimeError(f"Audio duration too short: {duration_sec:.2f}s (minimum 0.1s)")

    if duration_sec > 3600 * 4:  # 4 hours
        os.remove(temp_path)
        raise RuntimeError(f"Audio duration too long: {duration_sec / 3600:.2f} hours (maximum 4 hours)")

    # Print audio info
    print("📊 Audio file info:")
    print(f"  📁 File size: {file_size / 1024 / 1024:.2f} MB ({file_size} bytes)")
    print(f"  ⏱️  Duration: {duration_sec:.2f} seconds ({duration_sec / 60:.2f} minutes)")
    print(f"  🔊 Sample rate: {sample_rate} Hz")
    print(f"  🎚️  Channels: {channels} ({'mono' if channels == 1 else 'stereo'})")
    print(f"  🎵 Bit depth: {bit_depth} bits")

    # Convert to WAV format if requested
    if convert_to_wav:
        try:
            # Export as WAV with standard parameters for better compatibility
            audio.export(save_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        except Exception as e:
            # If conversion fails, fall back to the original file
            shutil.move(temp_path, save_path)
            print(f"⚠️ Warning: Audio conversion failed, using original format: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # Just rename the temp file to final path
        shutil.move(temp_path, save_path)

    # Return audio info for caller to use
    return {
        "file_size": file_size,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": bit_depth
    }

class WebRTCVADHelper:
    """WebRTC VAD helper class
    
    Note: WebRTC VAD only supports 8kHz, 16kHz, 32kHz sample rates.
    For original audio with sample rate higher than 32kHz (e.g., 48kHz), resampling to 16kHz
    will lose high-frequency information above 8kHz. This may affect VAD detection of
    high-frequency consonants, but usually has little impact for podcast speech scenarios.
    
    To preserve high-frequency information, consider:
    1. Use Silero VAD (supports any sample rate)
    2. Downsample audio to 16kHz for VAD, but keep original audio for ASR
    """
    
    def __init__(self, aggressiveness=2):
        """
        Args:
            aggressiveness: 0-3, higher is more aggressive (marks more content as speech)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000  # WebRTC VAD only supports 8k, 16k, 32k
        self.frame_duration = 30  # ms, options: 10, 20, 30
        
    def _read_frames(self, audio_data):
        """Split audio data into VAD frames"""
        frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2  # 16-bit = 2 bytes
        offset = 0
        while offset + frame_size <= len(audio_data):
            yield audio_data[offset:offset + frame_size]
            offset += frame_size
            
    def detect_speech_regions(self, audio_segment):
        """
        Detect speech regions in audio
        
        Args:
            audio_segment: pydub AudioSegment (any sample rate, will be resampled to 16kHz)
            
        Returns:
            List[Tuple[int, int]]: List of speech regions [(start_ms, end_ms), ...], timestamps correspond to original audio
        """
        # Record original audio frame rate for timestamp conversion
        original_frame_rate = audio_segment.frame_rate
        
        # Convert to 16kHz mono PCM for VAD
        # Note: For audio higher than 16kHz, this will lose high-frequency information
        audio = audio_segment.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data
        
        # Frame-by-frame detection
        regions = []
        current_region_start = None
        frame_idx = 0
        
        for frame in self._read_frames(raw_data):
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            timestamp_ms = frame_idx * self.frame_duration
            
            if is_speech and current_region_start is None:
                # Speech start
                current_region_start = timestamp_ms
            elif not is_speech and current_region_start is not None:
                # Speech end
                regions.append((current_region_start, timestamp_ms))
                current_region_start = None
                
            frame_idx += 1
        
        # Handle the last region (if still speaking at the end of audio)
        if current_region_start is not None:
            regions.append((current_region_start, frame_idx * self.frame_duration))
            
        return regions


def _find_best_split_in_range(silence_regions, start_ms, end_ms, target_ms):
    """
    Find the best split point within the specified range
    
    Prioritize splitting at the pause closest to target_ms to maintain semantic integrity
    
    Args:
        silence_regions: List of all silence regions [(start, end), ...]
        start_ms: Start position of current segment
        end_ms: End position of current segment (maximum position that cannot be exceeded)
        target_ms: Ideal split position (usually start_ms + max_chunk_ms)
        
    Returns:
        int: Best split point position (milliseconds)
    """
    # Filter silences within current range
    valid_silences = [
        (s, e) for s, e in silence_regions
        if s >= start_ms and e <= end_ms and s > start_ms  # Pause must be within current range and not at start
    ]
    
    if not valid_silences:
        # No suitable pause, forced to split at maximum length
        return min(start_ms + 270000, end_ms)
    
    # Find the pause midpoint closest to target
    best_split = None
    best_distance = float('inf')
    
    for s, e in valid_silences:
        # Use the midpoint of the pause as the split point
        midpoint = (s + e) // 2
        distance = abs(midpoint - target_ms)
        
        if distance < best_distance:
            best_distance = distance
            best_split = midpoint
    
    return best_split if best_split is not None else min(start_ms + 270000, end_ms)


def find_split_points(audio_segment, max_chunk_ms=270000, min_silence_ms=300):
    """
    Find split points based on VAD (Voice Activity Detection)
    
    Strategy:
    1. Use VAD to detect speech regions
    2. Prioritize splitting at pauses between speech regions to maintain semantic integrity
    3. If current segment exceeds max_chunk_ms, force split at the nearest suitable pause
    4. Only split mid-sentence when no suitable pause exists
    
    Args:
        audio_segment: pydub AudioSegment
        max_chunk_ms: Maximum segment length (milliseconds)
        min_silence_ms: Minimum silence length (milliseconds), pauses below this value are not considered split points
        
    Returns:
        List[int]: List of split points (milliseconds), including start 0 and end duration
    """
    duration_ms = len(audio_segment)
    
    # Audio too short, no need to split
    if duration_ms <= max_chunk_ms:
        return [0, duration_ms]
    
    # Use VAD to detect speech regions
    vad_helper = WebRTCVADHelper(aggressiveness=2)
    speech_regions = vad_helper.detect_speech_regions(audio_segment)
    
    # If no speech detected, treat entire audio as one segment (but limited by max length)
    if not speech_regions:
        if duration_ms <= max_chunk_ms:
            return [0, duration_ms]
        # No speech but too long, split at max length
        split_points = [0]
        current = max_chunk_ms
        while current < duration_ms:
            split_points.append(min(current, duration_ms))
            current += max_chunk_ms
        if split_points[-1] != duration_ms:
            split_points.append(duration_ms)
        return split_points
    
    # Derive silence regions from speech regions
    silence_regions = []
    
    # Silence at the beginning
    if speech_regions[0][0] > 0:
        silence_regions.append((0, speech_regions[0][0]))
    
    # Silence between speech regions
    for i in range(len(speech_regions) - 1):
        silence_start = speech_regions[i][1]
        silence_end = speech_regions[i + 1][0]
        if silence_end > silence_start:
            silence_regions.append((silence_start, silence_end))
    
    # Silence at the end
    if speech_regions[-1][1] < duration_ms:
        silence_regions.append((speech_regions[-1][1], duration_ms))
    
    # Filter silence regions that meet minimum silence length
    valid_silences = [
        (start, end) for start, end in silence_regions 
        if (end - start) >= min_silence_ms
    ]
    
    # Build split points
    split_points = [0]
    current_start = 0
    
    for silence_start, silence_end in valid_silences:
        # Calculate current segment length if split at current pause
        potential_end = silence_start
        segment_length = potential_end - current_start
        
        if segment_length >= max_chunk_ms:
            # Current segment exceeds max length, need to force split at nearest suitable pause
            best_split = _find_best_split_in_range(
                silence_regions, current_start, silence_start, current_start + max_chunk_ms
            )
            # Avoid adding duplicate split points
            if best_split != split_points[-1]:
                split_points.append(best_split)
            current_start = best_split
            
            # Recalculate distance to current pause after forced split
            segment_length = silence_start - current_start
            if segment_length >= max_chunk_ms:
                # Still too long, skip current pause and continue to next
                continue
        
        # Split at current pause (avoid duplicates)
        if silence_start != split_points[-1]:
            split_points.append(silence_start)
        current_start = silence_end
    
    # Ensure last split point is at audio end
    if split_points[-1] != duration_ms:
        # Check if last segment is too long
        last_segment_length = duration_ms - current_start
        if last_segment_length > max_chunk_ms:
            # Find best split point within last segment
            best_split = _find_best_split_in_range(
                silence_regions, current_start, duration_ms, current_start + max_chunk_ms
            )
            if best_split != split_points[-1]:
                split_points.append(best_split)
        if duration_ms != split_points[-1]:
            split_points.append(duration_ms)
    
    return split_points


def split_audio_smart(file_path, output_dir, max_chunk_ms=270000, min_silence_ms=300):
    """
    Smart audio splitting based on VAD (Voice Activity Detection)
    
    Maintains interface compatibility with the old split_audio but uses VAD internally
    to detect pause positions, trying to split at sentence boundaries to maintain semantic integrity.
    
    Args:
        file_path: Source file path
        output_dir: Output directory
        max_chunk_ms: Maximum segment length (milliseconds), default 270s
        min_silence_ms: Minimum silence length (milliseconds), default 300ms
        
    Returns:
        List[Dict]: List containing segment paths and time offsets
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
    
    # Load audio
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    
    # Audio too short, return directly
    if duration_ms <= max_chunk_ms:
        chunk_path = os.path.join(output_dir, "chunk_0.wav")
        audio.export(chunk_path, format="wav")
        return [{
            "path": chunk_path,
            "start_time_sec": 0.0,
            "end_time_sec": duration_ms / 1000.0
        }]
    
    # Use VAD to find split points
    split_points = find_split_points(audio, max_chunk_ms, min_silence_ms)
    
    # Export chunks
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


# Keep old function name as alias for backward compatibility
split_audio = split_audio_smart

def cleanup_files(paths):
    """Clean up temporary files and directories"""
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to cleanup {path}: {e}")