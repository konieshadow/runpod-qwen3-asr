# 文件名: src/debug.py
import sys
import os

# 将当前目录加入路径，确保能导入 handler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handler import handler, init_model

# 模拟输入数据 (参考 tests.json)
test_event = {
    "input": {
        "audio_url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
        "language": "auto",
        "initial_context": "",
        "use_previous_context": True
    },
    "id": "aliyun_debug_test"
}

if __name__ == "__main__":
    print("正在初始化模型...")
    init_model() # 手动初始化
    
    print("开始处理音频...")
    result = handler(test_event)
    
    print("\n====== 运行结果 ======")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))