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
8. [输出格式说明](#8-输出格式说明)
9. [错误码](#9-错误码)

---

## 1. OpenAI 兼容 API

兼容 OpenAI Audio API，可直接用 `openai` Python SDK 调用。

### POST /v1/audio/transcriptions

**说明**: 转写音频文件，同步返回详细 JSON。**始终返回完整结果**，字段根据请求参数动态包含。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `file` | File | ✅ | — | 音频文件（wav/mp3/mp4/flac/m4a/ogg/webm） |
| `language` | string | ❌ | auto | 语言提示（zh/en/ja/ko/yue） |
| `speaker_diarization` | bool | ❌ | false | 启用说话人分离，返回 `segments` + `speaker_id` |
| `speaker_group` | string | ❌ | — | 声纹组 ID（异步任务中会自动匹配替换 speaker_id 为注册名） |
| `emotion` | bool | ❌ | false | 返回 `emotion` 字段（😊😔😡等） |
| `events` | bool | ❌ | false | 返回 `events` 字段（👏😂🎵等） |
| `punctuation` | bool | ❌ | true | 标点恢复（模型自带，参数保留用于兼容） |
| `hotwords` | string | ❌ | — | 热词 JSON，如 `{"达摩院":20}` |

**基础响应**（无额外参数）:

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "language": "zh",
  "duration": 0.523,
  "model": "SenseVoiceSmall"
}
```

**启用情感 + 事件**（`emotion=true&events=true`）:

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "language": "zh",
  "duration": 0.523,
  "model": "SenseVoiceSmall",
  "emotion": "😊",
  "events": ["👏"]
}
```

**启用说话人分离**（`speaker_diarization=true`）:

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "language": "zh",
  "duration": 0.523,
  "model": "SenseVoiceSmall",
  "segments": [
    {
      "text": "大家好",
      "start": 0.43,
      "end": 1.52,
      "speaker_id": 0
    },
    {
      "text": "欢迎使用语音识别",
      "start": 1.68,
      "end": 3.91,
      "speaker_id": 1
    }
  ]
}
```

**启用说话人分离 + 声纹组**（`speaker_diarization=true&speaker_group=grp_abc123`）：

> 同步接口不做声纹匹配，仅透传 `speaker_group`。异步任务中会自动匹配并替换为注册名。

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "speaker_group": "grp_abc123",
  "segments": [
    {"text": "大家好", "start": 0.43, "end": 1.52, "speaker_id": 0},
    {"text": "欢迎使用语音识别", "start": 1.68, "end": 3.91, "speaker_id": 1}
  ]
}
```

**全参数启用**（`speaker_diarization=true&emotion=true&events=true`）:

```json
{
  "text": "大家好，欢迎使用语音识别。",
  "language": "zh",
  "duration": 0.523,
  "model": "SenseVoiceSmall",
  "emotion": "😊",
  "events": ["👏"],
  "segments": [
    {"text": "大家好", "start": 0.43, "end": 1.52, "speaker_id": 0},
    {"text": "欢迎使用语音识别", "start": 1.68, "end": 3.91, "speaker_id": 1}
  ]
}
```

**curl 示例**:

```bash
# 基本转写
curl -X POST http://localhost:17767/v1/audio/transcriptions -F file=@audio.wav

# 说话人分离 + 情感
curl -X POST http://localhost:17767/v1/audio/transcriptions \
  -F file=@meeting.mp3 \
  -F speaker_diarization=true \
  -F emotion=true \
  -F events=true
```

**Python 示例**:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:17767/v1", api_key="x")
result = client.audio.transcriptions.create(
    model="funasr",
    file=open("meeting.wav", "rb"),
    language="zh",
    extra_body={"speaker_diarization": True, "emotion": True, "events": True},
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
    {"id": "funasr", "object": "model", "created": 1700000000,
     "owned_by": "funasr", "ready": true, "name": "SenseVoiceSmall"}
  ]
}
```

---

## 2. HTTP REST API

### POST /recognition

简单文件上传转写，始终返回完整结果。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `audio` | File | ✅ | — | 音频文件 |
| `language` | string | ❌ | auto | 语言提示 |
| `speaker_diarization` | bool | ❌ | false | 说话人分离，返回 `sentences` + `speaker_id` |
| `speaker_group` | string | ❌ | — | 声纹组 ID（透传） |
| `emotion` | bool | ❌ | false | 情感标签 |
| `events` | bool | ❌ | false | 音频事件 |
| `punctuation` | bool | ❌ | true | 标点恢复 |

**基础响应**: `{"text": "...", "code": 0}`

**说话人分离** (`speaker_diarization=true`):

```json
{
  "text": "完整转写文本",
  "sentences": [
    {"text": "第一句", "start": 430, "end": 1520, "speaker_id": 0},
    {"text": "第二句", "start": 1680, "end": 3910, "speaker_id": 1}
  ],
  "code": 0
}
```

| 字段 | 说明 |
|------|------|
| `code=0` | 成功 |
| `code=1` | 出错（见 msg） |
| `start/end` | 毫秒时间戳 |
| `speaker_id` | 说话人编号（仅当 `speaker_diarization=true`） |
| `emotion` | 情感 emoji（仅当 `emotion=true`） |
| `events` | 事件 emoji 列表（仅当 `events=true`） |

---

## 3. 异步任务 API

用于长文件或 URL 远程文件转写。

### POST /api/tasks/submit

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 二选一 | 音频文件 |
| `url` | string | 二选一 | 远程音频 URL |
| 其他 | — | ❌ | 同 [公共参数](#8-公共参数) |

**返回值** (HTTP 202):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "queued",
  "created_at": 1749000000.0,
  "model": "sensevoice",
  "params": {
    "speaker_diarization": true,
    "emotion": true,
    "events": false,
    "punctuation": true,
    "language": "auto"
  }
}
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
      "params": {"speaker_diarization": true, "emotion": false, "events": false, "punctuation": true, "language": "auto"},
      "duration_seconds": 2.34,
      "audio_duration_seconds": 120.5
    }
  ],
  "total": 1
}
```

### GET /api/tasks/{task_id}

查询单个任务。**`result` 字段结构按请求参数动态包含**（同 [输出格式说明](#8-输出格式说明)）。

**仅说话人分离** (`speaker_diarization=true`):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "completed",
  "created_at": 1749000000.0,
  "completed_at": 1749000143.5,
  "duration_seconds": 143.5,
  "params": {
    "speaker_diarization": true,
    "emotion": false,
    "events": false,
    "punctuation": true,
    "language": "auto"
  },
  "result": {
    "text": "完整转写文本...",
    "segments": [
      {"text": "...", "start": 430, "end": 1520, "speaker_id": 0},
      {"text": "...", "start": 1680, "end": 3910, "speaker_id": 1}
    ]
  }
}
```

**说话人分离 + 声纹匹配** (`speaker_diarization=true&speaker_group=grp_abc`):

> 异步任务会自动用 SV 模型提取每个 segment 的声纹，匹配数据库后替换 `speaker_id`。
> 匹配到的 segment 会有 `speaker` 字段（注册名），未匹配的保留数字 `speaker_id`。

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "completed",
  "params": {
    "speaker_diarization": true,
    "speaker_group": "grp_abc",
    "emotion": false,
    "events": false,
    "punctuation": true,
    "language": "auto"
  },
  "result": {
    "text": "完整转写文本...",
    "segments": [
      {"text": "大家好", "start": 430, "end": 1520, "speaker_id": 0, "speaker": "张三"},
      {"text": "欢迎使用", "start": 1680, "end": 3910, "speaker_id": 1}
    ]
  }
}
```

> 上面示例中 `speaker_id=0` 匹配到了声纹库中的"张三"，`speaker_id=1` 未匹配到（库中无此人）。

**全参数启用** (`speaker_diarization=true&emotion=true&events=true&speaker_group=grp_abc`):

```json
{
  "result": {
    "text": "完整转写文本...",
    "emotion": "😊",
    "events": ["👏"],
    "segments": [
      {"text": "大家好", "start": 430, "end": 1520, "speaker_id": 0, "speaker": "张三"},
      {"text": "欢迎使用", "start": 1680, "end": 3910, "speaker_id": 1}
    ]
  }
}
```

**进行中**: `{"task_id": "...", "status": "processing"}`

**失败**: `{"task_id": "...", "status": "failed", "error": "错误信息"}`

**状态说明**:

| 状态 | 含义 |
|------|------|
| `downloading` | 正在下载 URL 文件 |
| `queued` | 已加入推理队列 |
| `processing` | 正在推理 |
| `completed` | 完成 |
| `failed` | 失败（见 error 字段） |

### DELETE /api/tasks/{task_id}

删除任务及其关联文件。`{"deleted": true, "task_id": "..."}`

---

## 4. 声纹管理 API

多租户隔离：每个 group 独立存储声纹，互不可见。

**使用流程**：
1. 注册说话人 → 获得 `group_id`
2. 转写时传入 `speaker_group=group_id` → 自动将 `speaker_id` 替换为注册名

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
    {"group_id": "grp_abc123", "speaker_count": 2, "speakers": ["张三", "李四"]}
  ],
  "total": 1
}
```

### GET /api/speakers/{group_id}

查看指定 group 的说话人列表。

### DELETE /api/speakers/{group_id}/{name}

删除指定 group 中的说话人。

---

## 5. WebSocket 流式 API

### ws://host:17767/ws

Subprotocol: `binary`

**流程**: 发送 JSON 配置 → 发送二进制 PCM → 接收实时结果

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
| `is_speaking` | `true`=说话中 / `false`=结束（触发最终结果） |
| `speaker_diarization` | 启用说话人分离 |
| `itn` | 逆文本归一化 |

**服务端消息**:

```json
// 实时中间结果
{"mode": "2pass-online", "text": "大家", "wav_name": "microphone", "is_final": false}

// 离线修正结果（VAD 断句后）
{
  "mode": "2pass-offline",
  "text": "大家好，欢迎使用语音识别。",
  "clean_text": "大家好，欢迎使用语音识别。",
  "wav_name": "microphone",
  "is_final": true,
  "timestamp": [[430, 670], [670, 810]],
  "sentence_info": [
    {"start": 430, "end": 1520, "text": "大家好", "spk": 0}
  ]
}
```

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

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "SenseVoiceSmall",
  "models_loaded": ["asr", "asr_spk", "streaming", "vad", "punc", "sv", "emotion"]
}
```

### Swagger 文档

`http://host:17767/docs` — 交互式 API 文档。

### Web UI

`http://host:17767` — 浏览器管理界面。

---

## 8. 输出格式说明

核心原则：**请求了什么参数，JSON 里就有什么字段。没传的字段不出现。**

| 传入参数 | 输出字段 | 说明 |
|----------|---------|------|
| （无） | `text` | 仅返回清洗后的纯文本 |
| `emotion=true` | + `emotion` | 情感 emoji，如 `"😊"` |
| `events=true` | + `events` | 事件 emoji 列表，如 `["👏","😂"]` |
| `speaker_diarization=true` | + `segments[]` | 分段数组，每段含 `text`/`start`/`end`/`speaker_id` |
| `speaker_group=xxx` | `segments[].speaker` | 异步任务自动匹配，将 `speaker_id` 替换为注册名 |

**字段详情**：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `text` | string | 始终 | 清洗后的纯文本（去除 `<\|...\|>` 标签） |
| `emotion` | string | SenseVoice | 情感 emoji：😊😔😡😰🤢😮 |
| `events` | string[] | SenseVoice | 事件 emoji：👏😂🎵😭🤧😷 |
| `segments[].text` | string | 说话人分离 | 清洗后的分段文本 |
| `segments[].start` | number | 说话人分离 | 开始时间（毫秒） |
| `segments[].end` | number | 说话人分离 | 结束时间（毫秒） |
| `segments[].speaker_id` | int | 说话人分离 | cam++ 聚类编号（0, 1, 2...） |
| `segments[].speaker` | string | 声纹匹配 | 声纹库匹配到的注册名（仅异步任务） |

### 公共参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `language` | string | `auto` | `auto` / `zh` / `en` / `ja` / `ko` / `yue` |
| `speaker_diarization` | bool | `false` | 启用说话人分离 |
| `speaker_group` | string | — | 声纹组 ID（提前通过 `/api/speakers/register` 注册） |
| `emotion` | bool | `false` | 情感标签（SenseVoice 模型自带） |
| `events` | bool | `false` | 音频事件标签（SenseVoice 模型自带） |
| `punctuation` | bool | `true` | 标点恢复 |
| `hotwords` | string | — | 热词 JSON，如 `{"达摩院":20}` |

---

## 9. 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 201 | 创建成功（声纹注册） |
| 202 | 已接受（异步任务提交） |
| 400 | 参数错误 |
| 404 | 资源不存在 |
| 413 | 文件过大 |
| 500 | 服务端错误 |

> 任务结果保留 `FUNASR_DATA_TTL_DAYS` 天（默认 7 天），过期自动清理。
