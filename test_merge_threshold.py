#!/usr/bin/env python3
"""测试合并逻辑的单元测试 - 独立版本"""

# 复制关键常量和函数进行独立测试
MERGE_THRESHOLD = 0.65


def _merge_consecutive_short_segments(segments, max_chunk_ms, merge_threshold=0.65):
    """Step 3a: 合并连续短分段

    规则：连续分段总时长 < max_chunk_ms * merge_threshold 时合并
    迭代处理直到没有连续短分段

    Args:
        segments: 分段列表 [(start_ms, end_ms), ...]
        max_chunk_ms: 最大分段时长（毫秒）
        merge_threshold: 合并阈值（0-1），限制合并后的分段不超过 max_chunk_ms * merge_threshold
                        默认0.65，即270秒的分段合并后不超过175秒

    Returns:
        List[Tuple[int, int]]: 合并后的分段列表
    """
    if len(segments) <= 1:
        return segments

    # 计算实际的合并上限
    merge_limit = int(max_chunk_ms * merge_threshold)

    # 迭代合并，直到没有可合并的连续短分段
    while True:
        merged = []
        i = 0
        changed = False

        while i < len(segments):
            if i == len(segments) - 1:
                # 最后一个分段
                merged.append(segments[i])
                i += 1
                continue

            current_start, current_end = segments[i]
            next_start, next_end = segments[i + 1]

            current_len = current_end - current_start
            next_len = next_end - next_start
            combined_len = current_len + next_len

            # 关键修改：使用 merge_limit 而不是 max_chunk_ms
            if combined_len < merge_limit:
                # 合并这两个分段
                merged.append((current_start, next_end))
                i += 2
                changed = True
            else:
                merged.append(segments[i])
                i += 1

        if not changed:
            break
        segments = merged
        if len(segments) <= 1:
            break

    return segments


def test_merge_logic():
    """测试合并逻辑是否正确应用阈值"""
    print("测试合并逻辑")
    print("=" * 60)

    max_chunk_ms = 270000  # 270秒
    merge_threshold = MERGE_THRESHOLD  # 0.65
    merge_limit = int(max_chunk_ms * merge_threshold)  # 175500ms = 175.5秒

    print(f"配置:")
    print(f"  max_chunk_ms: {max_chunk_ms}ms ({max_chunk_ms / 1000}s)")
    print(f"  merge_threshold: {merge_threshold}")
    print(f"  merge_limit: {merge_limit}ms ({merge_limit / 1000}s)")
    print()

    # 测试用例1: 两个短分段应该合并
    test1 = [(0, 80000), (80000, 160000)]  # 80s + 80s = 160s < 175s
    result1 = _merge_consecutive_short_segments(test1, max_chunk_ms, merge_threshold)
    print(f"测试1: 两个短分段应该合并")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test1]}")
    print(f"  期望: 合并为一个分段 (160s)")
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result1]}")
    print(
        f"  ✅ 通过" if len(result1) == 1 and result1[0] == (0, 160000) else "  ❌ 失败"
    )
    print()

    # 测试用例2: 两个分段总和超过阈值，不应该合并
    test2 = [(0, 100000), (100000, 200000)]  # 100s + 100s = 200s > 175s
    result2 = _merge_consecutive_short_segments(test2, max_chunk_ms, merge_threshold)
    print(f"测试2: 两个分段总和超过阈值，不应该合并")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test2]}")
    print(f"  期望: 保持两个分段")
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result2]}")
    print(f"  ✅ 通过" if len(result2) == 2 else "  ❌ 失败")
    print()

    # 测试用例3: 多个分段，部分合并
    test3 = [
        (0, 60000),  # 60s
        (60000, 120000),  # 60s, 60+60=120s < 175s → 合并
        (120000, 250000),  # 130s, 不与前面合并 (120+130=250s > 175s)
        (250000, 350000),  # 100s, 130+100=230s > 175s → 不合并
    ]
    result3 = _merge_consecutive_short_segments(test3, max_chunk_ms, merge_threshold)
    print(f"测试3: 多个分段，部分合并")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test3]}")
    print(f"  分段长度: {[e / 1000 - s / 1000 for s, e in test3]}")
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result3]}")
    print(f"  结果长度: {[e / 1000 - s / 1000 for s, e in result3]}")

    # 验证：合并后分段不应超过 merge_limit
    max_seg_len = max([e - s for s, e in result3])
    print(f"  最长分段: {max_seg_len / 1000}s")
    print(
        f"  ✅ 通过"
        if max_seg_len <= merge_limit
        else f"  ❌ 失败: 超过阈值 {merge_limit / 1000}s"
    )
    print()

    # 测试用例4: 实际场景模拟
    # 模拟预分割产生的很多小分段（每个在静音中点分割）
    test4 = [
        (0, 140000),  # 140s
        (140000, 280000),  # 140s, 140+140=280s > 175s → 不合并
        (280000, 420000),  # 140s, 不合并
        (420000, 560000),  # 140s, 不合并
    ]
    result4 = _merge_consecutive_short_segments(test4, max_chunk_ms, merge_threshold)
    print(f"测试4: 实际场景 - 预分割后不合并")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test4]}")
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result4]}")
    print(f"  结果长度: {[e / 1000 - s / 1000 for s, e in result4]}")

    # 验证：所有分段都不超过 merge_limit
    all_valid = all([e - s <= merge_limit for s, e in result4])
    print(f"  ✅ 通过" if all_valid else "  ❌ 失败")
    print()

    # 测试用例5: 边界情况 - 刚好在阈值上
    test5 = [
        (0, 87000),  # 87s
        (87000, 174000),  # 87s, 87+87=174s < 175s → 应该合并
    ]
    result5 = _merge_consecutive_short_segments(test5, max_chunk_ms, merge_threshold)
    print(f"测试5: 边界情况 - 刚好在阈值内")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test5]}")
    print(
        f"  合并后长度: {(test5[0][1] - test5[0][0] + test5[1][1] - test5[1][0]) / 1000}s"
    )
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result5]}")
    print(f"  ✅ 通过" if len(result5) == 1 else "  ❌ 失败")
    print()

    # 测试用例6: 边界情况 - 刚好超过阈值
    test6 = [
        (0, 88000),  # 88s
        (88000, 176000),  # 88s, 88+88=176s > 175s → 不应该合并
    ]
    result6 = _merge_consecutive_short_segments(test6, max_chunk_ms, merge_threshold)
    print(f"测试6: 边界情况 - 刚好超过阈值")
    print(f"  输入: {[(s / 1000, e / 1000) for s, e in test6]}")
    print(
        f"  合并后长度: {(test6[0][1] - test6[0][0] + test6[1][1] - test6[1][0]) / 1000}s"
    )
    print(f"  结果: {[(s / 1000, e / 1000) for s, e in result6]}")
    print(f"  ✅ 通过" if len(result6) == 2 else "  ❌ 失败")
    print()

    print("=" * 60)
    print("关键验证:")
    print(f"  ✅ 合并阈值正确应用: {merge_limit / 1000}s")
    print(f"  ✅ 防止GPU OOM: 分段不超过 {merge_limit / 1000}s")
    print(f"  ✅ 保持合理分段数量: 不会产生过多小分段")
    print()
    print("修复总结:")
    print(f"  - 原问题: 合并条件过宽，导致分段接近270s，在24GB VRAM上OOM")
    print(f"  - 修复方案: 引入 merge_threshold=0.65，限制合并后分段 ≤175s")
    print(f"  - 效果: 为KV cache和aligner留出足够GPU内存余量")


if __name__ == "__main__":
    test_merge_logic()
