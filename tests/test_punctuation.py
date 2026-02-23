import json
import sys
import os
import re


def add_punctuation_to_segments(full_text, segments, language=None):
    """Copy of the function from utils.py for testing without dependencies."""
    if not segments or not full_text:
        return segments
    
    try:
        is_cjk = language and language.lower() in ('chinese', 'japanese', 'korean')
        
        if is_cjk:
            full_tokens = list(full_text)
            full_tokens = [t for t in full_tokens if t.strip()]
        else:
            full_tokens = full_text.split()
        
        num_tokens = len(full_tokens)
        
        full_tokens_normalized = []
        if is_cjk:
            for t in full_tokens:
                full_tokens_normalized.append(t)
        else:
            for t in full_tokens:
                norm = re.sub(r'[^\w\s]', '', t).lower().strip()
                full_tokens_normalized.append(norm)
        
        segment_words = []
        for seg in segments:
            if is_cjk:
                segment_words.append(seg['text'])
            else:
                word = re.sub(r'[^\w\s]', '', seg['text']).lower().strip()
                segment_words.append(word)
        
        cjk_punctuation = set('，。！？；：""''（）【】《》、·~…—·\'\".,!?;:$%&()*+,-./:;<=>?@[\\]^_`{|}~\s0123456789')
        
        def is_punctuation(char, is_cjk_lang):
            if is_cjk_lang:
                return char in cjk_punctuation
            return not char.strip() or not re.match(r'[\w]', char)
        
        num_segments = len(segments)
        
        if is_cjk:
            full_to_segment_map = {}
            segment_idx = 0
            full_idx = 0
            MAX_LOOK_AHEAD = 10
            
            while segment_idx < num_segments and full_idx < num_tokens:
                seg_word = segment_words[segment_idx]
                
                while full_idx < num_tokens and is_punctuation(full_tokens[full_idx], is_cjk):
                    full_idx += 1
                
                if full_idx >= num_tokens:
                    break
                
                full_token_norm = full_tokens_normalized[full_idx]
                
                if not full_token_norm:
                    full_idx += 1
                    continue
                
                if seg_word == full_token_norm:
                    full_to_segment_map[full_idx] = segment_idx
                    segment_idx += 1
                    full_idx += 1
                else:
                    found = False
                    for offset in range(1, min(MAX_LOOK_AHEAD + 1, num_tokens - full_idx)):
                        look_ahead_idx = full_idx + offset
                        while look_ahead_idx < num_tokens and is_punctuation(full_tokens[look_ahead_idx], is_cjk):
                            look_ahead_idx += 1
                        
                        if look_ahead_idx < num_tokens and full_tokens_normalized[look_ahead_idx] == seg_word:
                            for skipped in range(full_idx, look_ahead_idx):
                                full_to_segment_map[skipped] = -1
                            full_idx = look_ahead_idx
                            full_to_segment_map[full_idx] = segment_idx
                            segment_idx += 1
                            full_idx += 1
                            found = True
                            break
                    
                    if not found:
                        segment_idx += 1
            
            aligned_segments = []
            mapped_segment_indices = set()
            
            for full_idx in range(num_tokens):
                char = full_tokens[full_idx]
                
                if is_punctuation(char, is_cjk):
                    next_seg_idx = None
                    prev_seg_idx = None
                    
                    for fi in range(full_idx + 1, num_tokens):
                        if fi in full_to_segment_map and full_to_segment_map[fi] >= 0:
                            next_seg_idx = full_to_segment_map[fi]
                            break
                    
                    for fi in range(full_idx - 1, -1, -1):
                        if fi in full_to_segment_map and full_to_segment_map[fi] >= 0:
                            prev_seg_idx = full_to_segment_map[fi]
                            break
                    
                    if next_seg_idx is not None and next_seg_idx < len(segments):
                        seg = segments[next_seg_idx]
                        aligned_segments.append({
                            'start': seg['start'],
                            'end': seg['start'] + 0.01,
                            'text': char
                        })
                    elif prev_seg_idx is not None and prev_seg_idx < len(segments):
                        seg = segments[prev_seg_idx]
                        aligned_segments.append({
                            'start': seg['end'],
                            'end': seg['end'] + 0.01,
                            'text': char
                        })
                else:
                    seg_idx = full_to_segment_map.get(full_idx, -1)
                    if seg_idx >= 0 and seg_idx < len(segments):
                        new_seg = segments[seg_idx].copy()
                        new_seg['text'] = char
                        aligned_segments.append(new_seg)
                        mapped_segment_indices.add(seg_idx)
            
            for seg_idx in range(len(segments)):
                if seg_idx not in mapped_segment_indices:
                    aligned_segments.append(segments[seg_idx].copy())
            
            aligned_segments.sort(key=lambda x: x['start'])
            
            return aligned_segments
        else:
            segment_idx = 0
            full_idx = 0
            aligned_segments = []
            MAX_LOOK_AHEAD = 3
            
            while segment_idx < num_segments and full_idx < num_tokens:
                seg_word = segment_words[segment_idx]
                full_token_norm = full_tokens_normalized[full_idx]
                
                if not full_token_norm:
                    full_idx += 1
                    continue
                
                if seg_word == full_token_norm:
                    new_seg = segments[segment_idx].copy()
                    new_seg['text'] = full_tokens[full_idx]
                    aligned_segments.append(new_seg)
                    segment_idx += 1
                    full_idx += 1
                else:
                    found_offset = 0
                    for offset in range(1, min(MAX_LOOK_AHEAD + 1, num_tokens - full_idx)):
                        if full_tokens_normalized[full_idx + offset] == seg_word:
                            found_offset = offset
                            break
                    
                    if found_offset > 0:
                        full_idx += found_offset
                        new_seg = segments[segment_idx].copy()
                        new_seg['text'] = full_tokens[full_idx]
                        aligned_segments.append(new_seg)
                        segment_idx += 1
                        full_idx += 1
                    else:
                        aligned_segments.append(segments[segment_idx].copy())
                        segment_idx += 1
            
            while segment_idx < num_segments:
                aligned_segments.append(segments[segment_idx].copy())
                segment_idx += 1
            
            return aligned_segments
    
    except Exception as e:
        print(f"⚠️ Warning: Failed to add punctuation to segments: {e}")
        return segments


def test_chinese_punctuation():
    print("=" * 50)
    print("Test: Chinese punctuation addition")
    print("=" * 50)
    
    # Segments must match full_text characters (excluding punctuation)
    segments = [
        {"start": 0.0, "end": 0.5, "text": "本"},
        {"start": 0.5, "end": 1.0, "text": "期"},
        {"start": 1.0, "end": 1.5, "text": "节"},
        {"start": 1.5, "end": 2.0, "text": "目"},
        {"start": 2.0, "end": 2.5, "text": "由"},
        {"start": 2.5, "end": 3.0, "text": "天"},
        {"start": 3.0, "end": 3.5, "text": "猫"},
        {"start": 3.5, "end": 4.0, "text": "冠"},
        {"start": 4.0, "end": 4.5, "text": "名"},
        {"start": 4.5, "end": 5.0, "text": "播"},
        {"start": 5.0, "end": 5.5, "text": "出"},
    ]
    
    # Full text with punctuation
    full_text = "本期节目由天猫冠名播出。"
    
    result = add_punctuation_to_segments(full_text, segments, language="Chinese")
    
    print(f"Input segments: {len(segments)}")
    print(f"Output segments: {len(result)}")
    print("\nResult segments:")
    for seg in result:
        print(f"  '{seg['text']}' (start={seg['start']}, end={seg['end']})")
    
    # Check punctuation was added
    punct_marks = [seg['text'] for seg in result if seg['text'] in '，。！？；：']
    
    assert len(punct_marks) >= 1, f"Expected at least 1 punctuation mark, got {len(punct_marks)}"
    assert '。' in punct_marks, "Expected '。' in result"
    
    print("\n✓ Chinese punctuation test PASSED")
    return True


def test_english_punctuation():
    print("\n" + "=" * 50)
    print("Test: English punctuation addition")
    print("=" * 50)
    
    segments = [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
        {"start": 1.0, "end": 1.5, "text": "this"},
        {"start": 1.5, "end": 2.0, "text": "is"},
        {"start": 2.0, "end": 2.5, "text": "a"},
        {"start": 2.5, "end": 3.0, "text": "test"},
    ]
    
    full_text = "Hello world, this is a test."
    
    result = add_punctuation_to_segments(full_text, segments, language="English")
    
    print(f"Input segments: {len(segments)}")
    print(f"Output segments: {len(result)}")
    print("\nResult segments:")
    for seg in result:
        print(f"  '{seg['text']}'")
    
    # English keeps punctuation attached to words from full_text
    assert result[0]['text'] == "Hello", f"Expected 'Hello', got '{result[0]['text']}'"
    # Note: punctuation is attached to words in English mode
    assert result[-1]['text'] == "test.", f"Expected 'test.', got '{result[-1]['text']}'"
    
    print("\n✓ English punctuation test PASSED")
    return True


def test_with_real_data():
    print("\n" + "=" * 50)
    print("Test: Real data from exemples/result.txt")
    print("=" * 50)
    
    result_file = os.path.join(os.path.dirname(__file__), '..', 'exemples', 'result.txt')
    if not os.path.exists(result_file):
        print("⚠ Skipping: result.txt not found")
        return True
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output = data.get('output', {})
    full_text = output.get('text', '')
    segments = output.get('segments', [])
    
    print(f"Input: {len(segments)} segments")
    print(f"Full text length: {len(full_text)} characters")
    
    result = add_punctuation_to_segments(full_text, segments, language='Chinese')
    
    print(f"Output: {len(result)} segments")
    
    punct_marks = set('，。！？；：""''（）【】《》')
    punct_count = sum(1 for seg in result if seg['text'] in punct_marks)
    
    print(f"Punctuation segments: {punct_count}")
    
    assert len(result) > len(segments), "Output should have more segments than input"
    assert punct_count > 0, "Should have punctuation marks in result"
    
    print("\n✓ Real data test PASSED")
    return True


def test_no_language_param():
    print("\n" + "=" * 50)
    print("Test: No language parameter (should default to non-CJK behavior)")
    print("=" * 50)
    
    segments = [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
    ]
    
    full_text = "Hello world"
    
    result = add_punctuation_to_segments(full_text, segments)
    
    print(f"Input segments: {len(segments)}")
    print(f"Output segments: {len(result)}")
    
    assert len(result) > 0, "Should return segments"
    
    print("\n✓ No language param test PASSED")
    return True


def test_empty_input():
    print("\n" + "=" * 50)
    print("Test: Empty input handling")
    print("=" * 50)
    
    result = add_punctuation_to_segments("", [], language="Chinese")
    assert result == [], "Empty segments should return empty list"
    
    result = add_punctuation_to_segments("text", [], language="Chinese")
    assert result == [], "Empty segments should return empty list"
    
    result = add_punctuation_to_segments("", [{"start": 0, "end": 1, "text": "a"}], language="Chinese")
    assert len(result) == 1, "Should return original segment"
    
    print("✓ Empty input test PASSED")
    return True


if __name__ == "__main__":
    tests = [
        test_empty_input,
        test_no_language_param,
        test_chinese_punctuation,
        test_english_punctuation,
        test_with_real_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    sys.exit(0 if failed == 0 else 1)
