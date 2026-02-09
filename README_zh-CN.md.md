Qwen3-ASR Serverless Worker for RunPod

此仓库用于在 RunPod Serverless 上部署 Qwen3-ASR 音频转录服务。它包含自动处理长音频切片、时间戳对齐和 vLLM 加速推理。

[![Runpod](https://api.runpod.io/badge/konieshadow/runpod-qwen3-asr)](https://console.runpod.io/hub/konieshadow/runpod-qwen3-asr)

特性

模型: Qwen3-ASR-0.6B (主模型) + Qwen3-ForcedAligner-0.6B (时间戳对齐)。

长音频支持: 基于 WebRTC VAD（语音活动检测）智能切分长音频，检测语音区域间的停顿以尽量保持语义完整性，并自动合并时间戳以规避 ForceAligner 的长度限制。

极速推理: 使用 vLLM 后端。

格式: 接受 MP3/WAV/M4A 等格式的 URL。

部署步骤

1. 构建 Docker 镜像

你需要一个 Docker Registry (如 Docker Hub) 来存放镜像。

# 登录 Docker Hub
docker login

# 构建镜像 (这一步会下载模型，可能需要较长时间)
docker build -t your-username/qwen3-asr-serverless:v1 .

# 推送镜像
docker push your-username/qwen3-asr-serverless:v1


2. 在 RunPod 上创建 Template

进入 RunPod Console -> Templates.

点击 New Template.

Container Image: your-username/qwen3-asr-serverless:v1

Container Disk: 建议设置至少 20 GB (模型 + 临时音频文件)。

Environment Variables: 不需要特殊变量，但可以添加 HF_TOKEN 如果你使用私有模型。

3. 创建 Serverless Endpoint

进入 Serverless -> New Endpoint.

选择刚才创建的 Template。

GPU 选择:

推荐: 24GB VRAM (NVIDIA L4, RTX 3090, RTX 4090)。

0.6B 模型本身很小，但在处理长音频切片和 vLLM KV Cache 时，大显存更稳定。

Workers: Min 0, Max 5 (根据需求)。

测试 API

```json
{
  "input": {
    "audio_url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "language": "auto",
    "initial_context": "",
    "use_previous_context": false
  }
}
```

### 语言参数说明

| 参数值 | 说明 |
|--------|------|
| `"auto"` 或 `null` | 自动检测语言（推荐） |
| `"Chinese"` | 中文 |
| `"English"` | 英文 |
| `"Cantonese"` | 粤语 |
| `"Japanese"` | 日语 |
| `"Korean"` | 韩语 |
| `"French"` | 法语 |
| `"German"` | 德语 |
| `"Spanish"` | 西班牙语 |
| `"Portuguese"` | 葡萄牙语 |
| `"Italian"` | 意大利语 |
| `"Arabic"` | 阿拉伯语 |
| `"Russian"` | 俄语 |
| `"Thai"` | 泰语 |
| `"Vietnamese"` | 越南语 |
| `"Indonesian"` | 印尼语 |
| `"Turkish"` | 土耳其语 |
| `"Hindi"` | 印地语 |
| `"Malay"` | 马来语 |
| `"Dutch"` | 荷兰语 |
| `"Swedish"` | 瑞典语 |
| `"Danish"` | 丹麦语 |
| `"Finnish"` | 芬兰语 |
| `"Polish"` | 波兰语 |
| `"Czech"` | 捷克语 |
| `"Filipino"` | 菲律宾语 |
| `"Persian"` | 波斯语 |
| `"Greek"` | 希腊语 |
| `"Romanian"` | 罗马尼亚语 |
| `"Hungarian"` | 匈牙利语 |
| `"Macedonian"` | 马其顿语 |

### 上下文参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `initial_context` | string | `null` | 初始上下文文本，用于第一个音频片段的转录 |
| `use_previous_context` | boolean | `false` | 是否开启上下文传递模式。开启后，上一个 chunk 的转录结果会自动作为下一个 chunk 的上下文 |

**上下文功能说明：**
- `initial_context`: 在转录第一个音频片段前提供前置文本，帮助模型理解上下文，提高转录准确性。
- `use_previous_context`: 对于长音频切分后的连续转录，开启此选项可以将上一个片段的转录结果（最后 200 字符）传递给下一个片段，保持上下文的连贯性。

> **注意**: 如果不指定 `language` 参数或使用 `"auto"`，模型将自动检测音频语言。

### 响应数据格式

```json
{
  "delayTime": 109075,
  "executionTime": 62528,
  "id": "b015cf6d-0ea7-4006-9d12-a87c5aee576d-e2",
  "output": {
    "language_detected": "English",
    "segments": [
      {
        "end": 0.48,
        "start": 0.24,
        "text": "This"
      },
      {
        "end": 0.56,
        "start": 0.48,
        "text": "is"
      }
    ],
    "text": "This is Space Time Series 29 Episode 12..."
  },
  "status": "COMPLETED"
}
```

**响应字段说明：**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `delayTime` | number | 请求延迟时间（毫秒） |
| `executionTime` | number | 实际执行时间（毫秒） |
| `id` | string | 请求唯一标识符 |
| `status` | string | 请求状态（`COMPLETED`、`FAILED` 等） |
| `output` | object | 转录结果 |
| `output.language_detected` | string | 自动检测到的语言 |
| `output.text` | string | 完整的转录文本 |
| `output.segments` | array | 带时间戳的文本片段 |
| `output.segments[].start` | number | 片段开始时间（秒） |
| `output.segments[].end` | number | 片段结束时间（秒） |
| `output.segments[].text` | string | 片段文本内容 |


注意事项

模型切换: 默认配置为 0.6B 模型。如果需要使用 1.7B 模型以获得更好的准确率，请修改 builder/fetch_models.py 和 src/handler.py 中的 MODEL_NAME 变量，并重新构建镜像。

冷启动: 第一次请求可能需要 15-30 秒来加载模型到显存。之后在 Warm 状态下的请求将立即处理。