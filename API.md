# FunASR All-in-One API 文档

**Base URL**: `http://{host}:17767`

---

## 目录

1. [OpenAI 兼容 API](#1-openai-兼容-api)
2. [HTTP REST API](#2-http-rest-api)
3. [异步任务 API](#3-异步任务-api)
4. [声纹管理 API](#4-声纹管理-api)
5. [WebSocket 流式 API](#5-websocket-流式-api)
6. [MCP 协议](#6-mcp-协议)
7. [系统接口](#7-系统接口)
8. [公共参数](#8-公共参数)
9. [错误码](#9-错误码)

---

## 1. OpenAI 兼容 API

兼容 OpenAI Audio API，可直接用 `openai` Python SDK 调用。

### POST /v1/audio/transcriptions

**说明**: 转写音频文件，同步返回。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `file` | File | ✅ | — | 音频文件（wav/mp3/mp4/flac/m4a/ogg/webm） |
| `language` | string | ❌ | auto | 语言提示（zh/en/ja/ko/yue） |
| `response_format` | string | ❌ | json | `json` 仅返回文本 / `verbose_json` 含分段 |
| `speaker_diarization` | bool | ❌ | false | 启用说话人分离 |
| `speaker_group` | string | ❌ | — | 声纹组 ID（配合声纹注册使用） |
| `emotion` | bool | ❌ | false | 返回情感标签 |
| `events` | bool | ❌ | false | 返回音频事件标签 |
| `punctuation` | bool | ❌ | true | 标点恢复 |
| `hotwords` | string | ❌ | — | 热词 JSON，如 `{"达摩院":20}` |

**response_format=json**

```json
{ "text": "大家好，欢迎使用语音识别。" }
```

**response_format=verbose_json**

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "segments": [
    {
      "text": "大家好",
      "start": 0.43,
      "end": 1.52,
      "speaker_id": 0,
      "speaker": "张三"
    },
    {
      "text": "欢迎使用语音识别",
      "start": 1.68,
      "end": 3.91,
      "speaker_id": 1,
      "speaker": "李四"
    }
  ],
  "language": "zh",
  "duration": 0.523,
  "model": "SenseVoiceSmall",
  "emotion": "😊",
  "events": ["👏"],
  "speaker_group": "grp_abc123"
}
```

**curl 示例**:

```bash
# 基本转写
curl -X POST http://localhost:17767/v1/audio/transcriptions \
  -F file=@audio.wav

# 带说话人分离 + 情感
curl -X POST http://localhost:17767/v1/audio/transcriptions \
  -F file=@meeting.mp3 \
  -F speaker_diarization=true \
  -F speaker_group=grp_abc123 \
  -F emotion=true \
  -F response_format=verbose_json
```

**Python 示例**:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:17767/v1", api_key="x")
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    language="zh",
    extra_body={"speaker_diarization": True, "emotion": True},
)
print(result.text)
```

---

### GET /v1/models

列出当前部署的模型信息。

```json
{
  "object": "list",
  "data": [
    { "id": "funasr", "object": "model", "created": 1700000000,
      "owned_by": "funasr", "ready": true, "name": "SenseVoiceSmall" }
  ]
}
```

---

## 2. HTTP REST API

### POST /recognition

简单文件上传转写，参数完整版本。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `audio` | File | ✅ | — | 音频文件 |
| `language` | string | ❌ | auto | 语言提示 |
| `speaker_diarization` | bool | ❌ | false | 说话人分离 |
| `speaker_group` | string | ❌ | — | 声纹组 ID |
| `emotion` | bool | ❌ | false | 情感标签 |
| `events` | bool | ❌ | false | 音频事件 |
| `punctuation` | bool | ❌ | true | 标点恢复 |

**返回值**:

```json
{
  "text": "完整转写文本",
  "sentences": [
    { "text": "第一句", "start": 430, "end": 1520, "speaker_id": 0 },
    { "text": "第二句", "start": 1680, "end": 3910, "speaker_id": 1 }
  ],
  "emotion": "😊",
  "events": ["👏"],
  "code": 0
}
```

| 字段 | 说明 |
|------|------|
| `code=0` | 成功 |
| `code=1` | 出错（见 msg） |
| `start/end` | 毫秒时间戳 |
| `speaker_id` | 说话人编号（0,1,2...仅当 speaker_diarization=true） |

---

## 3. 异步任务 API

用于长文件或 URL 远程文件转写。

### POST /api/tasks/submit

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 二选一 | 音频文件 |
| `url` | string | 二选一 | 远程音频 URL |
| 其他 | — | ❌ | 同 OpenAI API 参数（language, speaker_diarization 等） |

**返回值** (HTTP 202):

```json
{ "task_id": "a1b2c3d4e5f6", "status": "queued", "created_at": 1749000000.0, "model": "sensevoice" }
```

### GET /api/tasks

获取所有任务列表。

```json
{
  "tasks": [
    {
      "task_id": "a1b2c3d4e5f6",
      "status": "completed",
      "created_at": 1749000000.0,
      "model": "sensevoice",
      "duration_seconds": 2.34,
      "audio_duration_seconds": 120.5
    }
  ],
  "total": 1
}
```

### GET /api/tasks/{task_id}

查询单个任务。

**完成时**:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "completed",
  "created_at": 1749000000.0,
  "completed_at": 1749000143.5,
  "duration_seconds": 143.5,
  "audio_duration_seconds": 1800.0,
  "result": {
    "text": "完整转写文本...",
    "segments": [
      { "text": "...", "start": 430, "end": 1520, "speaker_id": 0, "speaker": "张三" }
    ],
    "emotion": "😊",
    "events": ["👏"]
  }
}
```

**进行中**:

```json
{ "task_id": "a1b2c3d4e5f6", "status": "processing" }
```

**失败**:

```json
{ "task_id": "a1b2c3d4e5f6", "status": "failed", "error": "下载失败: Connection timeout" }
```

**状态说明**:

| 状态 | 含义 |
|------|------|
| `downloading` | 正在下载 URL 文件 |
| `queued` | 已加入推理队列 |
| `processing` | 正在推理 |
| `completed` | 完成 |
| `failed` | 失败（见 error 字段） |

### DELETE /api/tasks/{task_id}

删除任务及其关联文件。

```json
{ "deleted": true, "task_id": "a1b2c3d4e5f6" }
```

---

## 4. 声纹管理 API

多租户隔离：每个 group 独立存储声纹，互不可见。

### POST /api/speakers/register

注册说话人声纹。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | File | ✅ | 参考音频（5-30s，单人，背景安静） |
| `name` | string | ✅ | 说话人名字（如"张三"） |
| `speaker_group` | string | ❌ | 已有 group ID（不传则自动创建新 group） |

**返回** (HTTP 201):

```json
{
  "group_id": "grp_abc123def456",
  "name": "张三",
  "status": "registered",
  "message": "说话人 '张三' 已注册到 group 'grp_abc123def456'"
}
```

> ⚠️ **group_id 需要保存好**，后续转写时通过 `speaker_group=grp_abc123def456` 带入。

### GET /api/speakers

列出所有声纹组。

```json
{
  "groups": [
    { "group_id": "grp_abc123", "speaker_count": 2, "speakers": ["张三", "李四"] },
    { "group_id": "grp_def456", "speaker_count": 1, "speakers": ["王五"] }
  ],
  "total": 2
}
```

### GET /api/speakers/{group_id}

查看指定 group 的说话人列表。

```json
{ "group_id": "grp_abc123", "speakers": ["张三", "李四"], "count": 2 }
```

### DELETE /api/speakers/{group_id}/{name}

删除指定 group 中的说话人。

```json
{ "deleted": true, "group_id": "grp_abc123", "name": "张三" }
```

---

## 5. WebSocket 流式 API

### ws://host:17767/ws

Subprotocol: `binary`

**流程**:

1. 客户端发送 JSON 配置消息
2. 客户端发送二进制 PCM 音频数据
3. 服务端实时返回识别结果

**配置消息**:

```json
{
  "mode": "2pass",
  "chunk_size": [5, 10, 5],
  "chunk_interval": 10,
  "wav_name": "microphone",
  "wav_format": "pcm",
  "is_speaking": true,
  "audio_fs": 16000,
  "itn": true,
  "speaker_diarization": false,
  "hotwords": "{\"达摩院\":20}"
}
```

| 参数 | 说明 |
|------|------|
| `mode` | `online`(纯实时) / `offline`(离线) / `2pass`(实时+离线修正) |
| `chunk_size` | `[5,10,5]` 表示 600ms 显示窗口，300ms 前瞻 |
| `chunk_interval` | 流式 ASR 触发间隔（帧数） |
| `is_speaking` | `true`=说话中 / `false`=结束（发送 false 触发最终结果） |
| `speaker_diarization` | 启用说话人分离 |

**结束信号**:

```json
{"is_speaking": false}
```

**服务端消息**:

```json
// 实时中间结果
{"mode": "2pass-online", "text": "大家", "wav_name": "microphone", "is_final": false}

// 离线修正结果（含情感/事件标签和说话人信息）
{
  "mode": "2pass-offline",
  "text": "<|zh|><|HAPPY|>大家好，欢迎使用语音识别。",
  "clean_text": "大家好，欢迎使用语音识别。",
  "wav_name": "microphone",
  "is_final": true,
  "timestamp": [[430, 670], [670, 810], ...],
  "sentence_info": [
    {"start": 430, "end": 1520, "text": "<|zh|><|HAPPY|>大家好", "spk": 0}
  ]
}
```

> `2pass-online`: 实时低延迟结果 | `2pass-offline`: VAD 断句后用离线模型修正，更准确，含情感标签和说话人编号

---

## 6. MCP 协议

### POST /mcp

Streamable HTTP 协议，支持 Claude Desktop / Cursor / Claude Code 接入。

**配置示例** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "funasr": {
      "type": "http",
      "url": "http://localhost:17767/mcp"
    }
  }
}
```

**可用工具**:

| 工具 | 参数 | 说明 |
|------|------|------|
| `transcribe_audio` | `audio_path`, `language`, `speaker_diarization` | 转写本地文件 |
| `transcribe_url` | `url`, `language`, `speaker_diarization` | 下载并转写远程文件 |

---

## 7. 系统接口

### GET /health

健康检查和模型状态。

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "SenseVoiceSmall",
  "models_loaded": ["asr", "streaming", "vad", "punc", "sv", "emotion"]
}
```

### Swagger 文档

`http://host:17767/docs` — FastAPI 交互式 API 文档，可直接在浏览器中测试所有接口。

### Web UI

`http://host:17767` — 浏览器管理界面（文件转写 / 实时录音 / 任务管理 / 声纹管理 / 服务状态）。

---

## 8. 公共参数

所有转写接口（OpenAI API / HTTP REST / 异步任务）共享以下参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `language` | string | `auto` | `auto` / `zh` / `en` / `ja` / `ko` / `yue` |
| `speaker_diarization` | bool | `false` | 启用说话人分离（需预加载 cam++ 模型） |
| `speaker_group` | string | — | 声纹组 ID，配合声纹注册使用 |
| `emotion` | bool | `false` | 返回情感标签（SenseVoice 自带） |
| `events` | bool | `false` | 返回音频事件标签（SenseVoice 自带） |
| `punctuation` | bool | `true` | 标点恢复 |
| `hotwords` | string | — | 热词 JSON，提高特定词准确率 |

---

## 9. 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 201 | 创建成功（声纹注册） |
| 202 | 已接受（异步任务提交） |
| 400 | 参数错误（缺少必填参数、文件格式不支持等） |
| 404 | 资源不存在（任务 ID/group ID 无效） |
| 413 | 文件过大 |
| 500 | 服务端错误（模型推理失败、ffmpeg 转码失败等） |

> 任务结果保留 `DATA_TTL_DAYS` 天（默认 7 天），过期自动清理。
>
> 所有上传的音频处理完成后立即删除，不留存原始文件（异步任务除外）。
