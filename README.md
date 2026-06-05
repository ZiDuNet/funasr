# FunASR All-in-One

**一站式语音识别 Docker 部署方案** — 一个容器，全部能力。

## 功能矩阵

| 能力 | 接口 | 说明 |
|------|------|------|
| 🔌 OpenAI 兼容 API | `POST /v1/audio/transcriptions` | 可直接用 OpenAI SDK 调用 |
| 🔄 WebSocket 流式 | `ws://host:8000/ws` | 实时麦克风识别（offline/online/2pass） |
| 📡 HTTP REST | `POST /recognition` | 简单文件上传转写 |
| 🤖 MCP 协议 | `POST /mcp` | Claude/Cursor 等 AI 工具接入 |
| 👥 声纹管理 | `POST /api/speakers/register` | 多租户声纹注册与匹配 |
| 📦 异步任务 | `POST /api/tasks/submit` | 长文件/URL 异步转写 |
| 🌐 Web UI | `http://host:8000` | 浏览器管理界面 |

## 支持的模型

| 模型 | 能力 | 语言 | 大小 |
|------|------|------|------|
| **SenseVoiceSmall** | ASR + 情感识别 + 事件检测 | 中/英/日/韩/粤 | 234M |
| **Paraformer-zh-streaming** | 流式实时识别 | 中/英 | 220M |
| **fsmn-vad** | 语音活动检测 | 中/英 | 0.4M |
| **ct-punc** | 标点恢复 | 中/英 | 290M |
| **cam++** | 说话人分离/声纹识别 | — | 7.2M |

首次启动自动从 ModelScope 下载，模型持久化在 `./models` 目录。

## 快速开始

```bash
git clone https://github.com/ZiDuNet/funasr.git
cd funasr/api

# 1. 配置环境
cp .env.example .env

# 2. 启动 CPU 模式
docker compose up --build -d

# 3. 等待模型下载（首次约 2-5 分钟）
docker logs -f funasr

# 4. 验证
curl http://localhost:8000/health
```

## GPU 模式

```bash
# 1. 安装 NVIDIA Container Toolkit
# 2. 取消 docker-compose.yml 中 deploy 部分的注释
# 3. 修改 .env: DEVICE=cuda
docker compose up -d
```

## API 使用示例

### OpenAI 兼容 API

```bash
# 基本转写
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F model=sensevoice

# 带说话人分离
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=sensevoice \
  -F speaker_diarization=true \
  -F speaker_group=grp_abc123 \
  -F emotion=true \
  -F response_format=verbose_json
```

### Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
result = client.audio.transcriptions.create(
    model="sensevoice", file=open("meeting.wav", "rb"),
    extra_body={"speaker_diarization": True, "emotion": True}
)
print(result.text)
```

### 声纹注册

```bash
# 注册说话人
curl -X POST http://localhost:8000/api/speakers/register \
  -F audio=@zhangsan.wav -F name=张三
# → {"group_id": "grp_abc123", "name": "张三"}

# 查看已注册
curl http://localhost:8000/api/speakers
```

### 异步任务

```bash
# 提交长文件
curl -X POST http://localhost:8000/api/tasks/submit \
  -F file=@long_meeting.mp3

# 提交 URL
curl -X POST http://localhost:8000/api/tasks/submit \
  -F url=https://example.com/audio.mp3

# 查询结果
curl http://localhost:8000/api/tasks/{task_id}
```

### WebSocket

```python
# 用 Python 客户端
python FunASR/runtime/python/websocket/funasr_wss_client.py \
  --host localhost --port 8000 --mode 2pass --wav_path test.wav
```

### MCP

在 `~/.claude.json` 中配置：

```json
{
  "mcpServers": {
    "funasr": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## 配置参数

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DEVICE` | `cpu` | `cpu` 或 `cuda` |
| `PORT` | `8000` | 服务端口 |
| `DATA_TTL_DAYS` | `7` | 任务结果保留天数（0=不清理） |
| `WORKER_THREADS` | `8` | 推理线程数 |
| `CONCURRENT_ASR_OFFLINE` | `2` | 离线 ASR 并发上限 |

完整参数见 `.env.example`。

## API 参数控制

所有转写端点支持以下参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | string | sensevoice | 模型选择 |
| `language` | string | auto | 语言提示 |
| `speaker_diarization` | bool | false | 说话人分离 |
| `speaker_group` | string | — | 声纹组 ID |
| `emotion` | bool | false | 情感标签 |
| `events` | bool | false | 音频事件检测 |
| `punctuation` | bool | true | 标点恢复 |
| `response_format` | string | json | json / verbose_json |

## 目录结构

```
api/
├── server/          ← FastAPI 后端
│   ├── models/      ← 模型注册表
│   ├── api/         ← API 路由
│   ├── core/        ← 推理层 + 任务管理 + 声纹
│   └── mcp_server.py
├── web/             ← Web 管理界面
├── models/          ← 模型缓存（自动下载）
├── data/            ← 任务结果持久化
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## 架构

```
单容器 (端口 8000)
├── OpenAI API (/v1/...)          ← 同步转写
├── 异步任务 (/api/tasks)          ← 长文件/URL
├── WebSocket (/ws)               ← 实时流式
├── MCP (/mcp)                    ← AI 工具
├── 声纹 (/api/speakers)          ← 多租户
└── Web UI (/)                    ← 管理界面

模型层（只加载一次，所有 API 共享）
├── SenseVoiceSmall   ← 离线 ASR + 情感 + 事件
├── paraformer-online ← 流式 ASR
├── fsmn-vad          ← 语音检测
├── ct-punc           ← 标点恢复
└── cam++             ← 声纹识别
```

## License

MIT — 基于 [FunASR](https://github.com/modelscope/FunASR) 构建
