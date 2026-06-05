# FunASR All-in-One

**一站式语音识别 Docker 部署方案** — 一个容器，全部能力。

## 功能矩阵

| 能力 | 接口 | 文档 |
|------|------|------|
| 🔌 OpenAI 兼容 API | `POST /v1/audio/transcriptions` | [API.md](API.md) |
| 🔄 WebSocket 流式 | `ws://host:17767/ws` | [API.md](API.md#5-websocket-流式-api) |
| 📡 HTTP REST | `POST /recognition` | [API.md](API.md#2-http-rest-api) |
| 🤖 MCP 协议 | `POST /mcp` | [API.md](API.md#6-mcp-协议) |
| 👥 声纹管理 | `POST /api/speakers/register` | [API.md](API.md#4-声纹管理-api) |
| 📦 异步任务 | `POST /api/tasks/submit` | [API.md](API.md#3-异步任务-api) |
| 🌐 Web UI | `http://host:17767` | `web/` |

## 默认模型

`.env` 中 `MODEL=sensevoice`（开箱即用），全部走魔搭自动下载：

| 模型 | 能力 | 语言 | 大小 |
|------|------|------|------|
| **SenseVoiceSmall** | ASR + 情感识别 + 事件检测 | 中/英/日/韩/粤 | 234M |
| **fsmn-vad** | 语音活动检测 | — | 0.4M |
| **ct-punc** | 标点恢复 | 中/英 | 290M |
| **cam++** | 说话人分离/声纹识别 | — | 7.2M |
| **emotion2vec+large** | 独立情感识别 | — | 300M |
| **Paraformer-zh-streaming** | 流式实时识别 | 中/英 | 220M |

> 总计约 1.2GB，首次启动自动下载。也可配置 `MODEL=paraformer` 或 `MODEL=fun-asr-nano` 切换。

## 快速开始

```bash
git clone https://github.com/ZiDuNet/funasr.git
cd funasr/api

# 配置
cp .env.example .env

# 启动 CPU（默认）
docker compose up --build -d

# 等模型下载完成（首次约 2-5 分钟，后续秒启动）
docker logs -f funasr

# 验证
curl http://localhost:17767/health
```

## GPU 模式

```bash
# .env 中修改两处:
#   DEVICE=cuda                          ← 选 GPU
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118  ← GPU 版 PyTorch
#
# docker-compose.yml 中取消 deploy 部分的注释
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: all
#             capabilities: [gpu]

docker compose down && docker compose up --build -d
```

> 构建时自动选 PyTorch 版本：CPU ~200MB / GPU ~2GB，无需手动改 Dockerfile。

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DEVICE` | `cpu` | `cpu` / `cuda` |
| `MODEL` | `sensevoice` | 离线模型: sensevoice / paraformer / fun-asr-nano |
| `PRELOAD_ALL` | `true` | 启动预加载所有模型 |
| `ENABLE_STREAMING` | `true` | WebSocket 流式 |
| `ENABLE_MCP` | `true` | MCP 协议 |
| `DATA_TTL_DAYS` | `7` | 任务保留天数 |

完整参数见 `.env.example`。

## API 速览

```bash
# 转写
curl http://localhost:17767/v1/audio/transcriptions -F file=@audio.wav

# 带说话人分离
curl http://localhost:17767/v1/audio/transcriptions \
  -F file=@meeting.wav -F speaker_diarization=true -F emotion=true

# 声纹注册
curl -X POST http://localhost:17767/api/speakers/register \
  -F audio=@ref.wav -F name=张三

# 异步任务
curl -X POST http://localhost:17767/api/tasks/submit \
  -F url=https://example.com/long_audio.mp3

# 健康
curl http://localhost:17767/health
```

完整文档 → **[API.md](API.md)**

## 目录结构

```
api/
├── server/          ← FastAPI 后端
├── web/             ← Web UI
├── models/          ← 模型缓存（自动下载，挂载持久化）
├── data/            ← 任务结果 + 声纹库（挂载持久化）
├── docker-compose.yml
├── Dockerfile
├── .env.example     ← 配置模板
├── API.md           ← 接口文档
└── README.md
```

## License

MIT — 基于 [FunASR](https://github.com/modelscope/FunASR) 构建
