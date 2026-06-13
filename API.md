# FunASR All-in-One 接口文档

基础地址：`http://{host}:17767`

本项目只提供一套标准 API：`/api/v1/*`。OpenAI 接口 `/v1/*` 仅用于兼容 OpenAI SDK。所有转写最终都会进入同一个标准结果结构。

## 设计原则

- 部署模型由 `.env` 中的 `MODEL=...` 决定，请求参数不允许临时切换模型。
- 所有转写结果必须包含全文、段落、句子、句子开始时间、句子结束时间。
- 可选能力只在标准结果上追加字段，不改变基准结构。
- 不支持的能力默认返回 `422`；传 `fallback=auto` 时跳过不支持能力，并在 `features.warnings` 中说明。
- 公开参数统一使用产品语义：`diarization`、`speaker_match`、`emotion`、`events`、`punctuation`、`hotwords`。

## WebUI

默认访问：

```text
http://{host}:17767/
```

根路径会跳转到 Gradio WebUI：

```text
http://{host}:17767/ui
```

WebUI 全中文，包含：

- 文件转写
- 异步任务
- 声纹组管理
- 服务状态
- 接口说明

## 认证

`.env` 中设置 `API_TOKEN=your-secret` 后启用认证。

HTTP 请求：

```http
Authorization: Bearer your-secret
```

WebSocket：

```text
ws://host:17767/api/v1/realtime/transcriptions?token=your-secret
```

`API_TOKEN` 为空时不启用认证。

## 端点总览

| 端点 | 方法 | 用途 |
|---|---:|---|
| `/api/v1/transcriptions` | POST | 标准同步转写 |
| `/api/v1/transcription-jobs` | POST | 提交异步转写任务 |
| `/api/v1/transcription-jobs` | GET | 列出异步任务 |
| `/api/v1/transcription-jobs/{task_id}` | GET | 查询单个异步任务 |
| `/api/v1/transcription-jobs/{task_id}` | DELETE | 删除异步任务 |
| `/api/v1/realtime/transcriptions` | WebSocket | 实时在线/离线/2pass 转写 |
| `/api/v1/speaker-groups` | POST | 创建声纹组 |
| `/api/v1/speaker-groups` | GET | 列出声纹组 |
| `/api/v1/speaker-groups/{group_id}` | DELETE | 删除整个声纹组 |
| `/api/v1/speaker-groups/{group_id}/speakers` | POST | 注册说话人 |
| `/api/v1/speaker-groups/{group_id}/speakers` | GET | 列出组内说话人 |
| `/api/v1/speaker-groups/{group_id}/speakers/{name}` | DELETE | 删除说话人 |
| `/api/v1/capabilities` | GET | 查询当前模型能力 |
| `/api/v1/models` | GET | 查询内置模型能力矩阵 |
| `/v1/audio/transcriptions` | POST | OpenAI 兼容转写 |
| `/v1/models` | GET | OpenAI 兼容模型列表 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

## 标准转写结果

所有成功转写都返回下面的基准结构：

```json
{
  "id": "tr_0123456789abcdef",
  "object": "transcription",
  "model": "SenseVoiceSmall",
  "source": "upload",
  "duration": 12.34,
  "processing_time": 0.82,
  "text": "完整转写文本。",
  "language": "auto",
  "paragraph_count": 1,
  "sentence_count": 2,
  "paragraphs": [
    {
      "id": 0,
      "start": 0.32,
      "end": 8.1,
      "text": "第一句。第二句。",
      "sentence_ids": [0, 1]
    }
  ],
  "sentences": [
    {
      "id": 0,
      "paragraph_id": 0,
      "start": 0.32,
      "end": 3.6,
      "text": "第一句。"
    },
    {
      "id": 1,
      "paragraph_id": 0,
      "start": 3.8,
      "end": 8.1,
      "text": "第二句。"
    }
  ],
  "features": {
    "requested": ["asr", "sentence_timestamps", "paragraphs"],
    "applied": ["asr", "sentence_timestamps", "paragraphs"],
    "warnings": []
  }
}
```

时间单位统一为秒。FunASR 原始 `sentence_info` 中的毫秒时间戳会被转换为秒。

如果模型没有返回句子时间戳，服务会返回一个覆盖整段音频的单句，并加入警告：

```json
{
  "code": "sentence_timestamps_unavailable",
  "feature": "sentence_timestamps",
  "message": "模型未返回 sentence_info，已使用全文生成单句时间轴"
}
```

## 可选增强字段

启用 `diarization=true` 后，句子会增加匿名说话人：

```json
{
  "speaker": {
    "id": "speaker_0"
  }
}
```

启用 `speaker_match=true&speaker_group=grp_xxx` 后，匹配成功的句子会增加注册名和分数：

```json
{
  "speaker": {
    "id": "speaker_0",
    "name": "alice",
    "score": 0.8421,
    "group_id": "grp_xxx"
  }
}
```

启用 `emotion=true` 或 `events=true` 后，会在顶层或句子上增加：

```json
{
  "emotion": "HAPPY",
  "events": ["Laughter"]
}
```

## 请求参数

Multipart 请求支持平铺表单字段，也支持一个 JSON 字符串字段 `config`。新集成推荐使用 `config`。

```json
{
  "features": {
    "diarization": true,
    "speaker_match": {
      "enabled": true,
      "group_id": "grp_xxx"
    },
    "emotion": true,
    "events": true,
    "punctuation": true,
    "words": false,
    "raw": false
  },
  "options": {
    "language": "auto",
    "hotwords": {"FunASR": 20},
    "paragraph": {
      "enabled": true,
      "max_gap_seconds": 1.2,
      "max_sentences": 6
    }
  },
  "fallback": "error",
  "response_format": "json"
}
```

平铺字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---:|---|
| `file` | file | 必填 | 同步转写音频文件 |
| `language` | string | `auto` | 语言提示 |
| `diarization` | bool | `false` | 使用 cam++ 进行匿名说话人分离 |
| `speaker_match` | bool | `false` | 使用声纹组匹配注册说话人 |
| `speaker_group` | string | 空 | 声纹组 ID，启用 `speaker_match` 时必填 |
| `emotion` | bool | `false` | 情感识别 |
| `events` | bool | `false` | 音频事件检测 |
| `punctuation` | bool | `true` | 标点恢复或模型内置标点 |
| `hotwords` | string | 空 | 热词 JSON 字符串 |
| `words` | bool | `false` | 预留词级时间戳 |
| `raw` | bool | `false` | 返回原始 FunASR 输出 |
| `fallback` | string | `error` | `error` 或 `auto` |
| `response_format` | string | `json` | `json`、`verbose_json`、`text`、`srt`、`vtt` |

## 标准同步转写

```bash
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@audio.wav
```

带常用能力：

```bash
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@meeting.wav \
  -F language=zh \
  -F diarization=true \
  -F speaker_match=true \
  -F speaker_group=grp_xxx \
  -F emotion=true \
  -F events=true \
  -F fallback=auto
```

使用 JSON 配置：

```bash
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@meeting.wav \
  -F 'config={"features":{"diarization":true,"emotion":true},"options":{"language":"zh"}}'
```

输出字幕：

```bash
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@audio.wav \
  -F response_format=srt
```

## 异步任务

提交文件：

```bash
curl -X POST http://localhost:17767/api/v1/transcription-jobs \
  -F file=@long_meeting.mp3 \
  -F diarization=true
```

提交远程 URL：

```bash
curl -X POST http://localhost:17767/api/v1/transcription-jobs \
  -F url=https://example.com/audio.mp3 \
  -F emotion=true
```

响应：

```json
{
  "task_id": "abc123def456",
  "status": "queued",
  "created_at": 1749000000.0,
  "model": "SenseVoiceSmall",
  "params": {
    "diarization": true,
    "emotion": false,
    "events": false,
    "punctuation": true,
    "language": "auto"
  }
}
```

查询任务：

```bash
curl http://localhost:17767/api/v1/transcription-jobs/abc123def456
```

任务完成后，`result` 字段就是标准转写结果。

删除任务：

```bash
curl -X DELETE http://localhost:17767/api/v1/transcription-jobs/abc123def456
```

## 声纹组

创建声纹组：

```bash
curl -X POST http://localhost:17767/api/v1/speaker-groups
```

响应：

```json
{
  "group_id": "grp_abc123def456",
  "speaker_count": 0
}
```

注册说话人：

```bash
curl -X POST http://localhost:17767/api/v1/speaker-groups/grp_abc123def456/speakers \
  -F name=alice \
  -F audio=@alice.m4a
```

参考音频会通过 ffmpeg 转成 16k 单声道 PCM，支持 `wav/mp3/m4a/mp4/aac/flac/ogg/opus/webm/wma/amr` 等常见格式。建议 5-30 秒、单人说话、背景安静。

列出说话人：

```bash
curl http://localhost:17767/api/v1/speaker-groups/grp_abc123def456/speakers
```

删除说话人：

```bash
curl -X DELETE http://localhost:17767/api/v1/speaker-groups/grp_abc123def456/speakers/alice
```

删除整个声纹组：

```bash
curl -X DELETE http://localhost:17767/api/v1/speaker-groups/grp_abc123def456
```

转写时使用声纹组：

```bash
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@meeting.wav \
  -F diarization=true \
  -F speaker_match=true \
  -F speaker_group=grp_abc123def456
```

## 实时 WebSocket

Swagger `/docs` 不展示 WebSocket 路由；实时接口以本节为准。

连接地址：

```text
ws://localhost:17767/api/v1/realtime/transcriptions
```

如果启用了 `API_TOKEN`：

```text
ws://localhost:17767/api/v1/realtime/transcriptions?token=your-token
```

连接建立后先发送 `session.start` 配置事件。参数结构与 HTTP API 一致：

```json
{
  "type": "session.start",
  "mode": "2pass",
  "audio_fs": 16000,
  "wav_format": "pcm",
  "chunk_size": [5, 10, 5],
  "chunk_interval": 10,
  "wav_name": "microphone",
  "itn": true,
  "features": {
    "diarization": true,
    "speaker_match": {
      "enabled": true,
      "group_id": "grp_xxx"
    },
    "emotion": true,
    "events": true,
    "punctuation": true,
    "raw": false
  },
  "options": {
    "language": "auto",
    "hotwords": {"FunASR": 20}
  },
  "fallback": "auto"
}
```

随后发送二进制 PCM 帧：16 kHz、单声道、signed 16-bit little-endian。

结束录音：

```json
{"type": "audio.end"}
```

在线临时结果：

```json
{
  "type": "transcript.delta",
  "mode": "2pass-online",
  "text": "临时结果",
  "is_final": false
}
```

最终片段：

```json
{
  "type": "transcript.segment",
  "mode": "2pass-offline",
  "is_final": true,
  "text": "原始结果",
  "clean_text": "清洗后文本",
  "paragraph_count": 1,
  "sentence_count": 1,
  "paragraphs": [],
  "sentences": []
}
```

服务端事件：

| 事件 | 说明 |
|---|---|
| `session.started` | 配置已生效 |
| `transcript.delta` | 实时中间文本，`online` / `2pass` 返回 |
| `transcript.segment` | VAD 断句后的最终片段，`offline` / `2pass` 返回 |
| `error` | 配置或推理错误 |

三种模式：

| 模式 | 返回逻辑 | 增强字段 |
|---|---|---|
| `online` | 只返回实时中间文本，延迟最低 | 不返回说话人、声纹、情感、事件 |
| `offline` | VAD 断句后返回最终片段 | 最终段可返回说话人、声纹、情感、事件、标点 |
| `2pass` | 先返回实时中间文本，断句后返回最终修正片段 | 最终段可返回全部增强字段 |

说话人一致性在单个 WebSocket 连接会话内维护。开启 `diarization` 后，
服务端会用会话级 `SpeakerTracker` 尽量保持跨句 `speaker_0` 指向同一人；
开启 `speaker_match` 并指定 `group_id` 后，最终段会继续匹配注册声纹并返回姓名和分数。

## OpenAI 兼容接口

端点：

```text
POST /v1/audio/transcriptions
```

常用 OpenAI 字段：

| 字段 | 说明 |
|---|---|
| `file` | 必填 |
| `model` | 仅兼容 SDK，实际模型仍由 `.env` 决定 |
| `language` | 语言提示 |
| `prompt` | 映射为 `hotwords` |
| `response_format` | `json`、`verbose_json`、`text`、`srt`、`vtt` |
| `timestamp_granularities` | `word` 目前会按能力校验，可能返回 422 |

FunASR 扩展字段：

| 字段 | 说明 |
|---|---|
| `diarization` | 说话人分离 |
| `speaker_match` | 声纹匹配 |
| `speaker_group` | 声纹组 ID |
| `emotion` | 情感识别 |
| `events` | 事件检测 |
| `hotwords` | 热词 JSON |
| `fallback` | `error` 或 `auto` |

`response_format=json` 返回：

```json
{"text": "完整文本"}
```

`response_format=verbose_json` 返回 OpenAI 风格 `segments`，并附带标准 `paragraphs`、`sentences`。

Python 示例：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:17767/v1", api_key="x")
result = client.audio.transcriptions.create(
    model="funasr",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
    extra_body={"diarization": True, "emotion": True}
)
print(result.text)
```

## 能力查询

当前部署：

```bash
curl http://localhost:17767/api/v1/capabilities
```

内置模型矩阵：

```bash
curl http://localhost:17767/api/v1/models
```

常见能力字段：

| 字段 | 含义 |
|---|---|
| `asr` | 离线语音识别 |
| `sentence_timestamps` | 句子时间轴 |
| `paragraphs` | 段落生成 |
| `diarization` | 匿名说话人分离 |
| `speaker_match` | 注册声纹匹配 |
| `emotion` | 情感识别 |
| `events` | 音频事件 |
| `punctuation` | 标点能力 |
| `hotwords` | 热词能力 |
| `streaming` | 实时接口是否启用 |

## 错误码

不支持能力：

```json
{
  "detail": {
    "error": "unsupported_feature",
    "feature": "events",
    "message": "当前模型不支持 events"
  }
}
```

缺少声纹组：

```json
{
  "detail": {
    "error": "missing_speaker_group",
    "message": "speaker_match 需要提供 speaker_group/group_id"
  }
}
```

常见 HTTP 状态码：

| 状态码 | 含义 |
|---:|---|
| 200 | 成功 |
| 201 | 已创建声纹组或说话人 |
| 202 | 异步任务已提交 |
| 400 | 请求参数或 JSON 配置错误 |
| 401 | Token 缺失或错误 |
| 404 | 任务或说话人不存在 |
| 422 | 能力不支持或缺少依赖参数 |
| 500 | 转码或推理失败 |

## FunASR 语义说明

- 句子时间轴来自 FunASR `sentence_info`。
- 说话人分离通过 `spk_model="cam++"` 启用。
- 声纹匹配是在说话人分离后的二次匹配。
- SenseVoice 会在识别文本中输出情感和事件标签。
- 非 SenseVoice 情感识别可使用 emotion2vec 辅助模型。
- 事件检测仅 SenseVoice 明确支持。
- 实时接口使用独立 streaming pipeline、VAD 和 cache。
