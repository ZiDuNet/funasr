# FunASR 大而全统一容器方案

## Context

单一 Docker 容器，支持 4 种 API（WebSocket 流式、HTTP REST、OpenAI 兼容、MCP）+ Web 管理界面。所有功能共享一套模型，支持高并发、参数化控制、异步任务。

## 架构总览

```
单容器 funasr-all-in-one (端口 8000)
│
├── 🌐 Web UI (/)
│   ├── 文件转写（上传 + URL）
│   ├── 实时录音（麦克风 WebSocket）
│   ├── 异步任务管理
│   └── 服务状态 / 设置
│
├── 📡 OpenAI 兼容 API (/v1/...)          ← 纯同步，不异步
│   ├── POST /v1/audio/transcriptions     ← 同步直接返回
│   ├── GET  /v1/models
│   └── GET  /health
│
├── 📦 异步任务 API (/api/tasks)           ← 独立端点，非 OpenAI 兼容
│   ├── POST /api/tasks/submit            ← 提交文件/URL，返回 task_id
│   ├── GET  /api/tasks                   ← 任务列表
│   ├── GET  /api/tasks/{task_id}         ← 查询结果
│   └── DELETE /api/tasks/{task_id}       ← 删除任务
│
├── 🔌 HTTP REST
│   └── POST /recognition                 ← 简单同步转写
│
├── 🔄 WebSocket
│   └── /ws                               ← 流式 offline/online/2pass
│
├── 🤖 MCP
│   └── /mcp                              ← Streamable HTTP
│
├── 🧠 模型层（只加载一次）
│   ├── SenseVoiceSmall       (离线 ASR + 情感 + 事件)
│   ├── paraformer-zh-streaming (流式 ASR)
│   ├── fsmn-vad              (语音检测)
│   ├── ct-punc               (标点恢复)
│   └── cam++                 (说话人分离)
│
├── 🎛️ 参数化控制（每个请求独立控制）
│   ├── speaker_diarization=true/false
│   ├── emotion=true/false
│   ├── events=true/false
│   ├── punctuation=true/false
│   └── model=sensevoice/paraformer
│
└── GPU 显存 ~2GB ｜ ThreadPoolExecutor + Semaphore 并发控制
```

## 参数化控制设计

所有端点统一支持以下可选参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | string | sensevoice | 模型选择 |
| `speaker_diarization` | bool | false | 启用说话人分离（需加载 cam++） |
| `emotion` | bool | false | 返回情感标签（SenseVoice 自带） |
| `events` | bool | false | 返回音频事件标签（SenseVoice 自带） |
| `punctuation` | bool | true | 标点恢复 |
| `language` | string | auto | 语言提示 zh/en/ja/ko/yue |
| `response_format` | string | json | json / verbose_json |
| `hotwords` | string | - | 热词 JSON 字符串 |

**推理时动态控制**：
- `speaker_diarization=false` → 跳过声纹模型，省 GPU 算力
- `punctuation=false` → 跳过标点模型
- `emotion=false` + `events=false` → clean_text 去掉标签

## 异步任务系统

### OpenAI API vs 异步任务 — 分离设计

**OpenAI 兼容 API 始终同步返回**，与真实 OpenAI API 行为一致：
```bash
POST /v1/audio/transcriptions -F file=@audio.wav
→ { "text": "你好世界" }   # 总是直接返回结果
```

**异步任务是独立端点**，用于长文件和 URL：
```bash
# 提交本地文件
POST /api/tasks/submit -F file=@meeting.mp3
→ { "task_id": "abc123", "status": "queued" }

# 提交 URL
POST /api/tasks/submit -F url=https://example.com/audio.mp3
→ { "task_id": "def456", "status": "downloading" }

# 查询结果
GET /api/tasks/abc123
→ { "task_id": "abc123", "status": "completed",
    "result": { "text": "...", "segments": [...], "speakers": [...] } }
```

### 状态机

```
downloading → queued → processing → completed
                                    → failed
```

### 高并发设计

```
请求进入
  │
  ├── OpenAI API / HTTP REST → Semaphore 直接推理 → 同步返回
  │                           (sem_asr=4 并发上限)
  │
  ├── 异步任务 → asyncio.Queue 排队 → 后台 Worker 消费
  │             (不阻塞同步请求)
  │
  └── WebSocket → 独立 Semaphore 组
                 (sem_vad=4, sem_asr_online=4, sem_asr_offline=2)
```

## Web UI 设计

### 页面结构

**1. 文件转写页** (`/`)
- 拖拽上传区域（支持 wav/mp3/mp4/flac/m4a 等）
- URL 输入框（远程文件，提交到异步任务）
- 参数面板：模型 / 语言 / 说话人分离 / 情感 / 事件 / 标点
- 结果展示区：
  - 完整文本
  - 时间轴视图（分段 + 说话人颜色区分）
  - 情感/事件标签
  - JSON 原始结果

**2. 实时录音页** (`/realtime`)
- 开始/停止录音按钮
- 模式选择：offline / online / 2pass
- 实时文本滚动显示
- 参数：热词输入

**3. 任务管理页** (`/tasks`)
- 任务列表（task_id / 状态 / 时长 / 创建时间）
- 点击查看已完成任务详情
- 删除任务

**4. 服务状态页** (`/status`)
- 模型加载状态
- GPU 显存 / CPU 使用
- 并发 Semaphore 使用情况
- 热词管理

### 技术方案

- 纯前端：HTML + CSS + Vanilla JS（无框架依赖，单文件部署）
- 由 FastAPI 静态文件服务：`app.mount("/", StaticFiles(...))`
- 复用 `FunASR/runtime/html5/static/` 的录音组件 (recorder-core.js, pcm.js)

## 目录结构

```
api/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── models/                          ← 魔搭模型挂载
│
├── web/                             ← Web UI 静态文件
│   ├── index.html                   ← 文件转写 + URL 转写
│   ├── realtime.html                ← 实时录音
│   ├── tasks.html                   ← 任务管理
│   ├── status.html                  ← 服务状态
│   ├── css/style.css
│   └── js/
│       ├── api.js                   ← API 调用封装
│       ├── recorder.js              ← 浏览器录音
│       └── ws.js                    ← WebSocket 客户端
│
└── server/
    ├── __init__.py
    ├── main.py                      ← 入口
    ├── app.py                       ← FastAPI 应用工厂
    │
    ├── models/
    │   ├── __init__.py
    │   ├── registry.py              ← 模型注册表
    │   └── config.py                ← 模型配置
    │
    ├── api/
    │   ├── __init__.py
    │   ├── openai_api.py            ← OpenAI 兼容（纯同步）
    │   ├── tasks.py                 ← 异步任务 API（独立端点）
    │   ├── http_rest.py             ← HTTP REST
    │   └── websocket.py             ← WebSocket 流式
    │
    ├── mcp_server.py                ← MCP (FastMCP)
    │
    └── core/
        ├── __init__.py
        ├── inference.py             ← 推理层 + Semaphore
        ├── task_manager.py          ← 异步任务管理器
        ├── audio.py                 ← ffmpeg 转码
        └── postprocess.py           ← 后处理（情感/事件标签）
```

## 实施步骤（10 步）

### Step 1: 基础框架 — `server/models/` + `server/core/inference.py`

- 模型注册表：单例，懒加载，5 个模型
- `run_blocking(fn, sem)` 统一推理接口
- Semaphore 并发控制

**复用来源：**
- Semaphore 模式来自 `FunASR/runtime/python/websocket/funasr_wss_server.py:295-316`
- MODEL_CONFIGS 模式来自 `FunASR/examples/openai_api/server.py:35-57`

### Step 2: 音频工具 — `server/core/audio.py` + `server/core/postprocess.py`

- ffmpeg 转 PCM
- `rich_transcription_postprocess()` 情感/事件后处理

**复用来源：** `FunASR/runtime/python/onnxruntime/funasr_onnx/utils/postprocess_utils.py`

### Step 3: 异步任务管理器 — `server/core/task_manager.py`

- asyncio.Queue + 后台 Worker
- httpx 异步下载 URL
- 任务状态机 + 自动清理过期任务

### Step 4: OpenAI 兼容 API — `server/api/openai_api.py`

- `POST /v1/audio/transcriptions`：**纯同步**，直接返回结果
- 参数化控制：speaker_diarization / emotion / events / punctuation
- 所有推理走 `run_blocking()` + Semaphore

**复用来源：** `FunASR/examples/openai_api/server.py`（改造）

### Step 5: 异步任务 API — `server/api/tasks.py`

- `POST /api/tasks/submit`：提交文件或 URL
- `GET /api/tasks`：任务列表
- `GET /api/tasks/{task_id}`：查询结果
- `DELETE /api/tasks/{task_id}`：删除任务

### Step 6: HTTP REST — `server/api/http_rest.py`

- `POST /recognition`：简单同步转写
- 参数化控制同样支持

**复用来源：** `FunASR/runtime/python/http/server.py`（简化）

### Step 7: WebSocket 流式 — `server/api/websocket.py`

- FastAPI 原生 `@app.websocket("/ws")`
- offline / online / 2pass 模式
- 离线用 SenseVoiceSmall，流式用 paraformer-zh-streaming
- 共享 registry 中的独立 VAD/Punc/SV 模型

**复用来源：** `FunASR/runtime/python/websocket/funasr_wss_server.py`（改造为 FastAPI WebSocket）

关键改造：
```python
# 用 websocket.receive() 区分文本/二进制帧
while True:
    msg = await websocket.receive()
    if msg["type"] == "websocket.receive":
        if "text" in msg: handle_config(msg["text"])
        elif "bytes" in msg: handle_audio(msg["bytes"])
```

### Step 8: MCP — `server/mcp_server.py`

- FastMCP 定义工具
- `transcribe_audio` / `transcribe_url` / `query_task`
- `mcp.http_app()` 挂载到 `/mcp`

**复用来源：** `FunASR/examples/mcp_server/funasr_mcp.py`（改为 FastMCP）

### Step 9: Web UI — `web/`

- 4 个页面：文件转写 / 实时录音 / 任务管理 / 服务状态
- 复用 `FunASR/runtime/html5/static/` 的录音组件
- FastAPI `StaticFiles` serve

### Step 10: Docker + 应用工厂

- `server/app.py`：组装所有路由 + 模型加载 + 任务 Worker 启动
- `server/main.py`：Uvicorn 入口
- `Dockerfile` + `docker-compose.yml`
- `requirements.txt` + `.env.example`

## 模型下载清单

| # | 魔搭短名 | 完整 ID | 用途 |
|---|----------|--------|------|
| 1 | `iic/SenseVoiceSmall` | 同左 | ASR + 情感 + 事件 |
| 2 | `paraformer-zh-streaming` | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 流式 ASR |
| 3 | `fsmn-vad` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | VAD |
| 4 | `ct-punc` | `iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727` | 标点 |
| 5 | `cam++` | `iic/speech_campplus_sv_zh-cn_16k-common` | 说话人 |

环境变量 `MODELSCOPE_CACHE=/app/models` 指向本地。

## 验证方式

1. `docker compose up --build`
2. 浏览器打开 `http://localhost:8000` → Web UI
3. `curl http://localhost:8000/health` → 健康检查
4. OpenAI 同步：`curl http://localhost:8000/v1/audio/transcriptions -F file=@test.wav -F model=sensevoice`
5. HTTP REST：`curl http://localhost:8000/recognition -F audio=@test.wav`
6. 异步任务：`curl http://localhost:8000/api/tasks/submit -F file=@meeting.mp3`
7. 查询任务：`curl http://localhost:8000/api/tasks/{task_id}`
8. WebSocket：用客户端连接 `ws://localhost:8000/ws`
9. MCP：MCP Inspector 连接 `http://localhost:8000/mcp`
