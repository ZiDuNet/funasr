# FunASR All-in-One 快速启动

---

## 最快启动（云端镜像）

```bash
git clone https://github.com/ZiDuNet/funasr.git
cd funasr/api
cp .env.example .env
docker compose up -d
```

等模型下载完（`docker logs -f funasr` 看进度），打开 `http://localhost:17767/`，会进入原生单页 WebUI。标准转写、OpenAI 兼容、异步任务、实时流式、声纹管理和服务状态都在同一个页面用 Tab 切换。

---

## 四种场景

### 1. CPU 机器（最常见）

```bash
cp docker/.env.cpu .env
docker compose up -d
```

### 2. GPU 机器（需 NVIDIA Container Toolkit）

```bash
# 先装 NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

cp docker/.env.gpu .env
docker compose up -d
```

### 3. 本地构建（不用云端镜像）

```bash
cp .env.example .env
docker compose -f docker-compose.build.yml build
docker compose -f docker-compose.build.yml up -d
```

### 4. 指定 GPU + 本地构建

```bash
cp docker/.env.gpu .env
docker compose -f docker/docker-compose.gpu.build.yml build
docker compose -f docker/docker-compose.gpu.build.yml up -d
```

---

## 验证

```bash
curl http://localhost:17767/health
# → {"status":"ok","device":"cpu","model":"Fun-ASR-Nano",...}

curl -X POST http://localhost:17767/api/v1/transcriptions -F file=@test.wav
# → {"text":"大家好，欢迎使用语音识别。","paragraphs":[...],"sentences":[...]}
```

---

## 常用命令

```bash
docker compose up -d       # 启动
docker compose logs -f     # 看日志
docker compose restart     # 重启（换模型后）
docker compose down        # 停止
```

---

## 换模型

编辑 `.env` 中 `MODEL=xxx`，重启即可：

```bash
# CPU 推荐
MODEL=fun-asr-nano        # 31 语言，800M（默认）
MODEL=sensevoice           # 中英日韩粤 + 情感 + 事件，234M
MODEL=paraformer           # 中文生产级，220M

# GPU 才能用
MODEL=qwen3-asr            # 52 语言，1.7B
MODEL=glm-asr-nano         # 17 语言，1.5B
MODEL=whisper-large-v3-turbo  # 多语言识别+翻译，809M
```

---

## 接口速览

| 端点 | 用途 |
|------|------|
| `http://localhost:17767/` | 原生单页 WebUI |
| `http://localhost:17767/docs` | Swagger 交互文档 |
| `POST /api/v1/transcriptions` | 标准同步转写 |
| `POST /v1/audio/transcriptions` | OpenAI 兼容转写 |
| `ws://localhost:17767/api/v1/realtime/transcriptions` | WebSocket 流式 |
| `POST /api/v1/transcription-jobs` | 异步长文件转写 |
| `POST /api/v1/speaker-groups/{id}/speakers` | 声纹注册 |

---

## 标准返回

主接口永远返回全文、段落数量、句子数量、段落数组和句子时间轴：

```json
{
  "text": "...",
  "paragraph_count": 1,
  "sentence_count": 2,
  "paragraphs": [{"start": 0.32, "end": 8.1, "sentence_ids": [0, 1]}],
  "sentences": [{"start": 0.32, "end": 3.6, "text": "第一句。"}]
}
```

查看当前模型能力：

```bash
curl http://localhost:17767/api/v1/capabilities
```

---

## 常用功能示例

```bash
# 基础转写
curl -X POST http://localhost:17767/api/v1/transcriptions -F file=@audio.wav

# 说话人分离 + 情感 + 事件
curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@meeting.wav \
  -F diarization=true \
  -F emotion=true \
  -F events=true

# 声纹注册 → 转写匹配
curl -X POST http://localhost:17767/api/v1/speaker-groups
curl -X POST http://localhost:17767/api/v1/speaker-groups/grp_abc/speakers \
  -F audio=@zhangsan.wav -F name=张三

curl -X POST http://localhost:17767/api/v1/transcriptions \
  -F file=@meeting.wav \
  -F diarization=true \
  -F speaker_match=true \
  -F speaker_group=grp_abc

# 异步长文件
curl -X POST http://localhost:17767/api/v1/transcription-jobs -F file=@long_meeting.mp3
# → {"task_id":"abc123","status":"queued"}
curl http://localhost:17767/api/v1/transcription-jobs/abc123
# → {"task_id":"abc123","status":"completed","result":{...}}

# Token 认证（生产环境）
# .env 中设置 API_TOKEN=your-secret
curl -H "Authorization: Bearer your-secret" \
  http://localhost:17767/api/v1/transcriptions -F file=@audio.wav
```
