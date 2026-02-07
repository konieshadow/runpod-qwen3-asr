Qwen3-ASR Serverless Worker for RunPod

此仓库用于在 RunPod Serverless 上部署 Qwen3-ASR 播客转录服务。它包含自动处理长音频切片、时间戳对齐和 vLLM 加速推理。

特性

模型: Qwen3-ASR-0.6B (主模型) + Qwen3-ForcedAligner-0.6B (时间戳对齐)。

长音频支持: 自动将长播客切分为 4.5 分钟的片段，规避 ForceAligner 的长度限制，并自动合并时间戳。

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
    "audio_url": "https://example.com/podcast_episode.mp3",
    "language": "auto"
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

> **注意**: 如果不指定 `language` 参数或使用 `"auto"`，模型将自动检测音频语言。


注意事项

模型切换: 默认配置为 0.6B 模型。如果需要使用 1.7B 模型以获得更好的准确率，请修改 builder/fetch_models.py 和 src/handler.py 中的 MODEL_NAME 变量，并重新构建镜像。

冷启动: 第一次请求可能需要 15-30 秒来加载模型到显存。之后在 Warm 状态下的请求将立即处理。