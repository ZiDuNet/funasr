# FunASR All-in-One

**一站式语音识别 Docker 服务** — 一个容器，全部能力，开箱即用。

---

## ✨ 能力总览

| 维度 | 说明 |
|------|------|
| **7 种接口** | OpenAI API / HTTP REST / WebSocket 流式 / MCP / 异步任务 / 声纹管理 / Web UI |
| **7 个 ASR 模型** | 从 234M 到 1.7B，覆盖 5-52 语言，`.env` 一键切换 |
| **6 个辅助模型** | 流式 + VAD + 标点 + 说话人分离 + 情感识别，全部魔搭下载 |
| **单镜像通用** | 一个镜像 CPU/GPU 通用，`DEVICE=cpu` 或 `DEVICE=cuda` 切换 |
| **高并发** | 线程池 + Semaphore 并发控制，7 个维度可独立调节 |
| **多租户** | 声纹分组隔离，group 级别管理，互不可见 |
| **持久化** | 任务 JSON 落盘 + 自动清理（可配 TTL） |
| **国内加速** | apt/pip 阿里云镜像 + 模型走魔搭，全链路国内源 |

---

## 🧠 ASR 模型（.env 中 `MODEL=` 切换）

### 🤔 快速选型

| 你的场景 | 推荐 `MODEL=` | 原因 |
|----------|--------------|------|
| 🎤 中文会议/通话 + 区分说话人 | `paraformer` | 中文最强，支持时间戳 + 说话人分离 |
| 🌍 多语言（日/韩/法/德/方言等） | `fun-asr-nano` | 31 语言，LLM-based |
| 😊 需要情感识别/音频事件 | `sensevoice` | 极快（10s 音频仅需 70ms），自带情感+事件 |
| 🏆 最高精度，不在乎延迟 | `qwen3-asr` | 52 语言，上下文理解，需 GPU |
| 🔀 识别 + 翻译 | `whisper-large-v3-turbo` | 性价比高，809M |
| 📺 直播实时字幕 | 任意模型 + WebSocket 2pass | 实时 + 离线修正 |

### 模型详情

| `MODEL=` | 模型 | 能力 | 语言 | 大小 | GPU |
|----------|------|------|------|------|-----|
| `sensevoice` | SenseVoiceSmall | ASR + 情感 + 事件 | 中英日韩粤 | 234M | 可选 |
| `paraformer` | Paraformer-zh | 中文生产级 ASR + 字级时间戳 | 中英 | 220M | 可选 |
| **`fun-asr-nano`** ← 默认 | Fun-ASR-Nano | LLM-based ASR | 31 语言 | 800M | 可选 |
| `qwen3-asr` | Qwen3-ASR-1.7B | 高精度 ASR | 52 语言 | 1.7B | ⚠️ 必须 |
| `glm-asr-nano` | GLM-ASR-Nano-2512 | 高精度 ASR（超 Whisper V3） | 17 语言 | 1.5B | ⚠️ 必须 |
| `whisper-large-v3` | Whisper-large-v3 | 识别 + 翻译 | 多语言 | 1550M | 可选 |
| `whisper-large-v3-turbo` | Whisper-large-v3-turbo | 识别 + 翻译（加速版） | 多语言 | 809M | 可选 |

> 首次使用自动从魔搭下载，后续秒启动。`⚠️ 必须 GPU` 的模型需要 bf16 精度，仅 Ampere+ GPU 支持。

**辅助模型**（自动加载）：

| 模型 | 能力 | 大小 |
|------|------|------|
| Paraformer-zh-streaming | 流式实时识别 | 220M |
| fsmn-vad | 语音活动检测 | 0.4M |
| ct-punc | 标点恢复 | 290M |
| cam++ | 说话人分离 / 声纹 | 7.2M |
| emotion2vec+large | 独立情感识别 | 300M |

> ⚠️ 说话人分离、情感识别、事件检测仅 **SenseVoice / Paraformer** 支持。Qwen3 / GLM / Whisper 不支持这些附加能力。

---

## 🔌 接口能力

| 接口 | 协议 | 端点 | 用途 |
|------|------|------|------|
| 🔌 **OpenAI 兼容 API** | HTTP | `POST /v1/audio/transcriptions` | 用 `openai` SDK 直接调，零改造成本 |
| 📡 **HTTP REST** | HTTP | `POST /recognition` | 文件上传转写，参数最全 |
| 🔄 **WebSocket 流式** | WS | `ws://host:17767/ws` | 实时麦克风 / 流式音频，支持 3 种模式 |
| 🤖 **MCP 协议** | HTTP | `POST /mcp` | Claude Desktop / Cursor / Claude Code 直连 |
| 📦 **异步任务** | HTTP | `POST /api/tasks/submit` | 长文件 / URL 远程转写，提交后轮询结果 |
| 👥 **声纹管理** | HTTP | `POST /api/speakers/register` | 多租户声纹注册、匹配、删除 |
| 🌐 **Web UI** | Browser | `http://host:17767` | 浏览器管理界面（文件转写 / 实时录音 / 任务 / 声纹） |

完整接口文档 → **[API.md](API.md)**

---

## 🎛️ 接口可控参数

| 参数 | 默认 | 说明 | 支持模型 |
|------|------|------|----------|
| `language` | `auto` | 语言提示 | 全部 |
| `speaker_diarization` | `false` | 说话人分离 | SenseVoice / Paraformer |
| `speaker_group` | — | 声纹组 ID（多租户） | 支持分离的模型 |
| `emotion` | `false` | 情感标签（😊😢😠） | SenseVoice |
| `events` | `false` | 音频事件（👏😂🎶） | SenseVoice |
| `punctuation` | `true` | 标点恢复 | 全部 |
| `hotwords` | — | 热词 JSON | SenseVoice / Paraformer |

---

## 🔄 流式模式

| 模式 | 说明 | 延迟 |
|------|------|------|
| `online` | 纯流式，实时输出 | ~300ms |
| `offline` | VAD 断句后离线识别 | 1-3s |
| `2pass` | 实时 + 离线修正（推荐） | 兼顾低延迟和高精度 |

---

## 快速开始

```bash
git clone https://github.com/ZiDuNet/funasr.git
cd funasr/api

# 配置
cp .env.example .env
# 编辑 .env 选择 MODEL=xxx

# 构建并启动
docker compose up --build -d

# 等模型下载完成（首次，后续秒启动）
docker logs -f funasr

# 验证
curl http://localhost:17767/health
```

## 切换模型

```bash
# 编辑 .env
MODEL=qwen3-asr        # 切到 Qwen3-ASR
DEVICE=cuda            # 大模型建议 GPU

# 重启（自动下载新模型）
docker compose restart
docker logs -f funasr  # 看下载进度
```

## GPU 模式

```bash
# 1. .env 中 DEVICE=cuda
# 2. docker-compose.yml 取消 deploy 注释
# 3. 重启
docker compose down && docker compose up -d
```

## 常用操作

```bash
docker compose up -d          # 启动（已构建过）
docker compose up --build -d  # 重新构建 + 启动
docker compose logs -f        # 日志
docker compose restart        # 重启（切换模型/配置后）
docker compose down           # 停止
```

## 使用在线镜像

```bash
# 不构建，直接拉取远程镜像
docker compose -f docker-compose.pull.yml up -d
```

---

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DEVICE` | `cpu` | `cpu` / `cuda`（同一镜像，无需重建） |
| `MODEL` | `fun-asr-nano` | ASR 模型选择（见上表） |
| `PRELOAD_ALL` | `true` | 启动预加载所有模型 |
| `ENABLE_STREAMING` | `true` | WebSocket 流式 |
| `ENABLE_MCP` | `true` | MCP 协议 |
| `DATA_TTL_DAYS` | `7` | 任务保留天数 |
| `API_TOKEN` | 空 | Token 认证（留空不认证） |
| `PORT` | `17767` | 服务端口 |

---

## API 速览

```bash
# 转写
curl http://localhost:17767/v1/audio/transcriptions -F file=@audio.wav

# 带 Token 认证（.env 中设置了 API_TOKEN 时）
curl -H "Authorization: Bearer your-token" \
  http://localhost:17767/v1/audio/transcriptions -F file=@audio.wav

# 带说话人分离 + 情感
curl http://localhost:17767/v1/audio/transcriptions \
  -F file=@meeting.wav -F speaker_diarization=true -F emotion=true

# 声纹注册
curl -X POST http://localhost:17767/api/speakers/register \
  -F audio=@ref.wav -F name=张三

# 异步任务（URL）
curl -X POST http://localhost:17767/api/tasks/submit \
  -F url=https://example.com/long_audio.mp3

# OpenAI SDK（api_key 即为 API_TOKEN）
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:17767/v1', api_key='your-token')
r = c.audio.transcriptions.create(model='funasr', file=open('audio.wav','rb'))
print(r.text)
"
```

---

## 目录结构

```
api/
├── server/                  ← FastAPI 后端
│   ├── models/
│   │   ├── config.py        ← 7 个模型预设 + .env 驱动
│   │   └── registry.py      ← 单例模型注册中心
│   ├── core/
│   │   ├── inference.py     ← 统一推理层
│   │   ├── audio.py         ← ffmpeg 音频转码
│   │   ├── postprocess.py   ← 文本清洗/情感/事件
│   │   ├── task_manager.py  ← 异步任务
│   │   └── speaker_db.py    ← 声纹库（多租户）
│   ├── api/                 ← 6 个路由
│   ├── mcp_server.py        ← MCP 协议
│   ├── app.py               ← FastAPI + CORS
│   └── main.py              ← 0.0.0.0 启动
├── web/                     ← Web UI（挂载，改前端无需重建）
├── models/                  ← 模型缓存（挂载持久化）
├── data/                    ← 任务 + 声纹（挂载持久化）
├── docker-compose.yml       ← 本地构建版
├── docker-compose.pull.yml  ← 在线镜像版
├── Dockerfile
├── requirements.txt
├── .env.example
├── API.md
└── README.md
```

---

## ⚠️ 注意事项

| 事项 | 说明 |
|------|------|
| **局域网麦克风** | 浏览器安全策略限制，非 HTTPS 页面禁止麦克风。本地用 `localhost`，局域网需 Chrome flag 或 HTTPS |
| **emotion2vec 警告** | `Warning, miss key in ckpt` 是 FunASR 正常日志，不影响功能 |
| **大模型 GPU** | Qwen3-ASR / GLM-ASR 需要 bf16 精度，仅 NVIDIA Ampere+ (A100/3090/4090 等) 支持 |
| **模型兼容性** | 说话人分离/情感/事件仅 SenseVoice 和 Paraformer 支持，其他模型这些参数无效 |

---

## License

MIT — 基于 [FunASR](https://github.com/modelscope/FunASR) 构建
