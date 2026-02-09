# Qwen3-ASR Serverless Worker for RunPod

[中文文档](README_zh-CN.md)

This repository is for deploying the Qwen3-ASR audio transcription service on RunPod Serverless. It includes automatic long audio segmentation, timestamp alignment, and vLLM accelerated inference.

[![Runpod](https://api.runpod.io/badge/konieshadow/runpod-qwen3-asr)](https://console.runpod.io/hub/konieshadow/runpod-qwen3-asr)

## Features

- **Models**: Qwen3-ASR-0.6B (main model) + Qwen3-ForcedAligner-0.6B (timestamp alignment).
- **Long Audio Support**: Intelligently splits long audio using WebRTC VAD (Voice Activity Detection), detecting pauses between speech regions to maintain semantic integrity. Automatically merges timestamps and bypasses ForceAligner length limits.
- **Ultra-fast Inference**: Uses vLLM backend.
- **Format**: Accepts URLs for MP3/WAV/M4A and other formats.

## Deployment Steps

### 1. Build Docker Image

You need a Docker Registry (e.g., Docker Hub) to store the image.

```bash
# Login to Docker Hub
docker login

# Build image (this will download models and may take a while)
docker build -t your-username/qwen3-asr-serverless:v1 .

# Push image
docker push your-username/qwen3-asr-serverless:v1
```

### 2. Create Template on RunPod

1. Go to RunPod Console -> Templates.
2. Click New Template.
   - **Container Image**: `your-username/qwen3-asr-serverless:v1`
   - **Container Disk**: Recommended at least 20 GB (models + temporary audio files).
   - **Environment Variables**: No special variables needed, but you can add `HF_TOKEN` if using private models.

### 3. Create Serverless Endpoint

1. Go to Serverless -> New Endpoint.
2. Select the Template you just created.
3. **GPU Selection**:
   - Recommended: 24GB VRAM (NVIDIA L4, RTX 3090, RTX 4090).
   - The 0.6B model itself is small, but large VRAM is more stable when processing long audio segments and vLLM KV Cache.
4. **Workers**: Min 0, Max 5 (depending on needs).

## API Testing

### Request Body

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

### Language Parameter

| Parameter | Description |
|-----------|-------------|
| `"auto"` or `null` | Auto-detect language (recommended) |
| `"Chinese"` | Chinese |
| `"English"` | English |
| `"Cantonese"` | Cantonese |
| `"Japanese"` | Japanese |
| `"Korean"` | Korean |
| `"French"` | French |
| `"German"` | German |
| `"Spanish"` | Spanish |
| `"Portuguese"` | Portuguese |
| `"Italian"` | Italian |
| `"Arabic"` | Arabic |
| `"Russian"` | Russian |
| `"Thai"` | Thai |
| `"Vietnamese"` | Vietnamese |
| `"Indonesian"` | Indonesian |
| `"Turkish"` | Turkish |
| `"Hindi"` | Hindi |
| `"Malay"` | Malay |
| `"Dutch"` | Dutch |
| `"Swedish"` | Swedish |
| `"Danish"` | Danish |
| `"Finnish"` | Finnish |
| `"Polish"` | Polish |
| `"Czech"` | Czech |
| `"Filipino"` | Filipino |
| `"Persian"` | Persian |
| `"Greek"` | Greek |
| `"Romanian"` | Romanian |
| `"Hungarian"` | Hungarian |
| `"Macedonian"` | Macedonian |

### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_context` | string | `null` | Initial context text for the first audio segment transcription |
| `use_previous_context` | boolean | `false` | Enable context passing mode. When enabled, the previous chunk's transcription result is automatically used as context for the next chunk |

**Context Feature Description:**
- `initial_context`: Provides pre-text before transcribing the first audio segment, helping the model understand context and improve transcription accuracy.
- `use_previous_context`: For continuous transcription of long audio segments, enabling this option passes the previous segment's transcription result (last 200 characters) to the next segment, maintaining context coherence.

> **Note**: If `language` parameter is not specified or set to `"auto"`, the model will automatically detect the audio language.

### Response Format

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

**Response Field Description:**

| Field | Type | Description |
|-------|------|-------------|
| `delayTime` | number | Request delay time (milliseconds) |
| `executionTime` | number | Actual execution time (milliseconds) |
| `id` | string | Request unique identifier |
| `status` | string | Request status (`COMPLETED`, `FAILED`, etc.) |
| `output` | object | Transcription result |
| `output.language_detected` | string | Automatically detected language |
| `output.text` | string | Full transcription text |
| `output.segments` | array | Timestamped text segments |
| `output.segments[].start` | number | Segment start time (seconds) |
| `output.segments[].end` | number | Segment end time (seconds) |
| `output.segments[].text` | string | Segment text content |

## Notes

- **Model Switching**: Default configuration uses 0.6B model. If you need better accuracy with the 1.7B model, modify the `MODEL_NAME` variable in `builder/fetch_models.py` and `src/handler.py`, then rebuild the image.
- **Cold Start**: First request may take 15-30 seconds to load models into VRAM. Subsequent requests in Warm state will process immediately.
