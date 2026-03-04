# Qwen3-ASR Serverless Worker - 项目上下文

## 项目概述

这是一个 **Qwen3-ASR** 音频转录服务的 **RunPod Serverless** 部署包。它提供 AI 驱动的语音转文字 API，具有以下功能：

- **模型**：Qwen3-ASR-0.6B（主 ASR 模型）+ Qwen3-ForcedAligner-0.6B（时间戳对齐）
- **长音频支持**：使用 WebRTC VAD（语音活动检测）智能分段，在自然停顿处分割长音频
- **超快推理**：vLLM 后端实现 GPU 加速转录
- **上下文感知**：支持 `initial_context` 和 `use_previous_context`，提高连续音频的准确性
- **多语言**：支持 30+ 种语言，自动检测

## 技术栈

| 组件 | 技术 |
|------|------|
| 运行时 | Python 3.11 |
| 框架 | RunPod Serverless SDK |
| ML 后端 | vLLM, PyTorch, Qwen-ASR |
| 音频处理 | pydub, webrtcvad, ffmpeg |
| 容器 | Docker (CUDA 12.4.1, Ubuntu 22.04) |
| 模型中心 | Hugging Face |

## 项目结构

```
/root/runpod-qwen3-asr/
├── src/
│   ├── handler.py      # RunPod 主入口，处理任务
│   ├── utils.py        # 音频下载、VAD 分段、清理工具
│   └── debug.py        # 本地测试脚本（模拟 RunPod 任务）
├── builder/
│   └── fetch_models.py # Docker 构建时预下载模型
├── tests/
│   └── test_download_redirect.py  # 测试音频 URL 重定向处理
├── .runpod/
│   ├── hub.json        # RunPod Hub 元数据（GPU 需求、配置）
│   └── tests.json      # RunPod 自动化测试配置
├── Dockerfile          # 多阶段构建，预下载模型
├── requirements.txt    # Python 依赖
├── test_input.json     # 本地测试样例输入
└── README.md           # 用户文档
```

## 核心组件

### `src/handler.py`
- **入口点**：`handler(job)` 函数，RunPod 每次请求时调用
- **模型加载**：通过 `init_model()` 延迟初始化，使用 vLLM 后端
- **处理流程**：
  1. 从 URL 下载音频
  2. 使用 VAD 分割音频（最大 270 秒/块，最小 300ms 静音）
  3. 带上下文支持转录每个片段
  4. 合并时间戳，每 3 个片段清理 KV 缓存
  5. 清理临时文件

### `src/utils.py`
- **`download_audio()`**：下载音频并显示进度，验证格式，转换为 16kHz 单声道 WAV
- **`split_audio_smart()`**：基于 VAD 在自然停顿处智能分割
- **`WebRTCVADHelper`**：使用 WebRTC VAD 检测语音区域（16kHz）
- **`add_punctuation_to_segments()`**：将词级片段与带标点的完整文本对齐

### `builder/fetch_models.py`
- 预下载 ASR 和 Aligner 模型到 `/models/asr` 和 `/models/aligner`
- 使用 `snapshot_download()` 并过滤权重格式

## 构建与运行

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 预下载模型（可选，用于本地测试）
python builder/fetch_models.py

# 运行本地测试（模拟 RunPod 任务）
python src/debug.py

# 运行特定测试
python tests/test_download_redirect.py
```

### Docker 构建

```bash
# 登录 Docker Hub
docker login

# 构建镜像（下载模型，约 5-15 分钟）
docker build -t your-username/qwen3-asr-serverless:v1 .

# 推送到仓库
docker push your-username/qwen3-asr-serverless:v1
```

### RunPod 部署

1. **创建模板**：使用推送的 Docker 镜像，设置容器磁盘 ≥20GB
2. **创建端点**：选择 GPU（推荐 24GB 显存：L4, RTX 3090/4090）
3. **环境变量**：可选设置 `HF_TOKEN` 用于私有模型

### API 请求格式

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "language": "auto",
    "initial_context": "",
    "use_previous_context": false
  }
}
```

## 配置选项

### 模型选择
默认使用 0.6B 模型。如需使用 1.7B：
- 修改 `src/handler.py` 中的 `MODEL_NAME`
- 修改 `builder/fetch_models.py` 中的 `ASR_REPO_ID`
- 重新构建 Docker 镜像

### VAD 参数（在 `handler.py` 中）
```python
max_chunk_ms=270000    # 每块最大 270 秒
min_silence_ms=300     # 分割点最小 300ms 静音
```

### GPU 内存
```python
gpu_memory_utilization=0.8  # 在 handler.py 的 init_model() 中
```

## 开发规范

### 代码风格
- **类型提示**：在工具函数中使用
- **错误处理**：Try-except 块，提供用户友好的错误信息
- **日志**：使用 emoji 前缀的 print 语句（🚀, ✅, ⚠️, ❌）
- **清理**：始终在 `finally` 块中清理临时文件

### 测试实践
- **单元测试**：`tests/` 目录，用于隔离功能测试
- **集成测试**：`src/debug.py`，用于完整流程测试
- **RunPod 测试**：`.runpod/tests.json`，用于自动化部署验证

### 文件命名
- **模块**：`snake_case.py`
- **目录**：`snake_case`
- **临时文件**：`{job_id}_{type}.ext` 模式

### 错误处理模式
```python
# 根据错误类型提供用户友好的错误信息
if "Download timeout" in error_str:
    user_message = "音频下载超时..."
elif "OutOfMemory" in error_type:
    user_message = "音频太长，无法处理..."
```

## 重要说明

### KV 缓存管理
vLLM KV 缓存每 3 个片段清理一次，防止 OOM：
```python
if (idx + 1) % 3 == 0:
    _clear_kv_cache()
```

### 上下文传递
当 `use_previous_context=True` 时，每个片段的最后 200 个字符传递给下一个片段：
```python
current_context = text.strip()[-200:] if len(text) > 200 else text.strip()
```

### 时间戳解析
支持模型返回的多种格式：
- 列表：`[start, end, text]`
- 字典：`{"start": x, "end": y, "text": z}`
- 对象：`ForcedAlignItem(text=..., start_time=..., end_time=...)`

### 音频验证
- **最小时长**：0.1 秒
- **最大时长**：4 小时
- **采样率**：转换为 16kHz 单声道进行处理
- **支持格式**：MP3, WAV, M4A 等（通过 ffmpeg）

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 冷启动延迟（15-30秒） | 正常现象 - 模型加载时间 |
| OOM 错误 | 降低 `gpu_memory_utilization` 或使用更大 GPU |
| 音频下载超时 | 检查 URL 有效性，增加 `download_audio()` 中的超时时间 |
| 空音频文件 | URL 可能重定向到 HTML 页面，而非音频 |
| Forced aligner 错误 | 音频片段 < 0.5s，自动跳过 |

## 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `RP_CONCURRENCY` | 1 | 并发任务处理数 |
| `RP_THREADS` | 1 | 线程数 |
| `HF_TOKEN` | None | Hugging Face 认证令牌 |