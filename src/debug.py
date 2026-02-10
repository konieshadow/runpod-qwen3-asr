# 文件名: src/debug.py
import sys
import os

# 将当前目录加入路径，确保能导入 handler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handler import handler, init_model

# 模拟输入数据 (参考 tests.json)
test_event = {
    "input": {
        "audio_url": "http://alioss.gcores.com/uploads/audio/d72b2f75-8b21-46b9-903a-88cefee5dd65.mp3",
        "language": "auto",
        "initial_context": "",
        "use_previous_context": True
    },
    "id": "aliyun_debug_test"
}

if __name__ == "__main__":
    print("开始处理音频...")
    result = handler(test_event)
    
    print("\n====== 运行结果 ======")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))