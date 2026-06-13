# FunASR All-in-One 架构说明

本项目在 FunASR 之上提供一套标准 API。旧的演示型多入口已经移除，避免用户猜测该用哪个端点、哪个返回结构。

## 公开 API

| 能力 | 端点 |
|---|---|
| 同步转写 | `POST /api/v1/transcriptions` |
| 异步任务 | `POST/GET /api/v1/transcription-jobs` |
| 单个异步任务 | `GET/DELETE /api/v1/transcription-jobs/{task_id}` |
| 实时转写 | `WS /api/v1/realtime/transcriptions` |
| 声纹组 | `POST/GET /api/v1/speaker-groups` |
| 说话人 | `POST/GET /api/v1/speaker-groups/{group_id}/speakers` |
| 能力查询 | `GET /api/v1/capabilities`, `GET /api/v1/models` |
| OpenAI 兼容 | `POST /v1/audio/transcriptions`, `GET /v1/models` |
| 原生 WebUI | `GET /` |

## 内部层次

1. `server/models/capabilities.py`
   - 声明模型和能力兼容关系。
   - 路由根据该注册表校验请求能力。

2. `server/core/schemas.py`
   - 解析平铺表单字段和 JSON `config`。
   - 公开能力名称使用产品语义：`diarization`, `speaker_match`,
     `emotion`, `events`, `punctuation`, `hotwords`.

3. `server/core/transcription.py`
   - 执行当前部署的 FunASR 模型。
   - 始终产出标准 `paragraphs` 和 `sentences`。
   - 根据请求追加说话人、情感、事件、原始输出和 warnings。

4. `server/core/formatters.py`
   - 将标准结果转换为 `json`, `verbose_json`, `text`, `srt`, `vtt`。
   - OpenAI 兼容格式只在边界层处理，不进入推理逻辑。

5. API modules
   - `openai_api.py`：标准同步接口和 OpenAI 兼容接口。
   - `tasks.py`：异步任务接口，同样调用标准转写服务。
   - `speakers.py`：声纹组和说话人注册接口。
   - `websocket.py`：实时接口，最终片段返回标准结构。
   - `web/index.html`：原生单页中文 WebUI。

## 标准结果

每个转写结果必须包含：

- `text`
- `duration`
- `paragraph_count`
- `sentence_count`
- `paragraphs[]`
- `sentences[]`
- `sentences[].start`
- `sentences[].end`
- `sentences[].text`

如果 FunASR 没有返回 `sentence_info`，服务会返回一个覆盖整段音频的单句，并包含 warning。

## FunASR 语义

- 离线 ASR 模型由 `.env` 中的 `MODEL=...` 选择。
- 句子时间轴来自 FunASR `sentence_info`。
- 说话人分离使用 `spk_model="cam++"`。
- 声纹匹配是在注册声纹库上的二次匹配。
- SenseVoice 会在识别文本中直接输出情感和事件标签。
- 非 SenseVoice 情感识别可由辅助模型 `emotion2vec` 产出。
- 事件检测仅 SenseVoice 明确支持。
- 实时接口使用独立 streaming pipeline、cache 和 VAD。

## 设计规则

新路由不能发明新的返回结构。必须调用 `transcribe_pcm(...)`，只在协议边界适配标准结果。
