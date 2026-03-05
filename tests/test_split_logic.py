import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import find_split_points, download_audio
from pydub import AudioSegment


def test_split_logic():
    audio_url = "https://dts-api.xiaoyuzhoufm.com/track/60de7c003dd577b40d5a40f3/6981fe42e3b98bb2deff888b/media.xyzcdn.net/60de7c003dd577b40d5a40f3/llSpWJz2-qdy9PcYJxqpUmG0nFLw.m4a"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "test_audio.wav")
        
        print(f"下载音频: {audio_url}")
        download_audio(audio_url, audio_path, timeout=600)
        
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000
        
        print(f"\n音频总时长: {duration_sec:.2f} 秒 ({duration_sec/60:.2f} 分钟)")
        
        max_chunk_ms = 270000
        min_silence_ms = 300
        
        print(f"\n测试参数: max_chunk_ms={max_chunk_ms}, min_silence_ms={min_silence_ms}")
        
        split_points = find_split_points(audio, max_chunk_ms, min_silence_ms)
        
        print(f"\n分段结果: {len(split_points)-1} 个片段")
        print("=" * 60)
        
        segment_lengths = []
        for i in range(len(split_points) - 1):
            start_ms = split_points[i]
            end_ms = split_points[i + 1]
            length_sec = (end_ms - start_ms) / 1000
            segment_lengths.append(length_sec)
            
            print(f"片段 {i+1}: {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s (长度: {length_sec:.2f}s)")
        
        print("=" * 60)
        
        short_segments = [l for l in segment_lengths if l < 30]
        medium_segments = [l for l in segment_lengths if 30 <= l < 60]
        long_segments = [l for l in segment_lengths if l >= 60]
        
        print(f"\n统计:")
        print(f"  < 30s: {len(short_segments)} 个")
        print(f"  30-60s: {len(medium_segments)} 个")
        print(f"  >= 60s: {len(long_segments)} 个")
        print(f"  总计: {len(segment_lengths)} 个")
        
        if segment_lengths:
            avg_length = sum(segment_lengths) / len(segment_lengths)
            print(f"  平均长度: {avg_length:.2f}s")
            print(f"  最短: {min(segment_lengths):.2f}s")
            print(f"  最长: {max(segment_lengths):.2f}s")
        
        if short_segments:
            print(f"\n⚠️  警告: 发现 {len(short_segments)} 个短于30秒的片段")
            for i, l in enumerate(segment_lengths):
                if l < 30:
                    print(f"    短片段 {i+1}: {l:.2f}s")
        
        # 验证约束
        print("\n✅ 约束验证:")
        all_valid = True
        
        # 检查1: 任意分段 <= max_chunk_ms
        for i, l in enumerate(segment_lengths):
            if l > max_chunk_ms / 1000:
                print(f"  ❌ 分段 {i+1} 超过最大时长: {l:.2f}s > {max_chunk_ms/1000}s")
                all_valid = False
        
        # 检查2: 不存在连续短分段（连续总和 < max_chunk_ms）
        for i in range(len(segment_lengths) - 1):
            combined = segment_lengths[i] + segment_lengths[i+1]
            if combined < max_chunk_ms / 1000:
                print(f"  ❌ 分段 {i+1} 和 {i+2} 连续总时长过短: {combined:.2f}s < {max_chunk_ms/1000}s")
                all_valid = False
        
        if all_valid:
            print("  ✅ 所有约束满足")


if __name__ == "__main__":
    test_split_logic()
