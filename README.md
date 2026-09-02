# FunASR Web Service 部署说明

基于 Docker 的语音识别（ASR）+ 说话人分离（Speaker Diarization）服务，CPU 运行，提供 Web 上传界面、历史记录管理、接口文档。

## 目录结构

```
funasr-docker/
├── Dockerfile              # 镜像构建
├── docker-compose.yml      # 容器编排
├── entrypoint.sh           # 启动脚本
├── server.py               # Web服务 + 识别逻辑
├── .env                    # 环境配置
├── .env.example            # 配置模板
├── data/                   # 持久化数据（自动创建）
│   ├── audio/              # 上传的原始音频
│   ├── records/            # 识别记录 JSON
│   └── results/            # 识别结果 JSON
└── models/                 # 本地模型目录（可选，预下载）
```

## 快速启动

```bash
cd funasr-docker
docker compose up -d --build
```

启动后访问：
- **Web 管理界面**: http://localhost:11800
- **API 文档 (Swagger)**: http://localhost:11800/docs
- **ReDoc**: http://localhost:11800/redoc

## 配置 (.env)

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FUNASR_HOST_PORT` | `11800` | 宿主机映射端口 |
| `FUNASR_DEVICE` | `cpu` | 运行设备 |
| `FUNASR_MODEL` | `sensevoice` | 默认模型（sensevoice/paraformer） |
| `FUNASR_WORKERS` | `1` | 并发 worker 数，每个约占 1.2GB 内存 |
| `FUNASR_TOKEN` | 空 | 访问令牌，为空则不鉴权 |

调整并发：修改 `.env` 中 `FUNASR_WORKERS=2` 后 `docker compose up -d`。

## 功能

### Web 界面
- 上传音频（支持多文件、拖拽）
- 选择模型 + 说话人分离开关
- 查看识别结果（分段 + 说话人标签 + 时间戳）
- 历史记录管理（查看/下载/删除）

### API 接口
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/audio/transcriptions` | OpenAI 兼容转写接口 |
| GET  | `/api/records` | 历史记录列表 |
| GET  | `/api/records/{id}` | 单条记录详情 |
| GET  | `/api/records/{id}/audio` | 下载原音频 |
| GET  | `/api/records/{id}/result` | 下载结果 JSON |
| DELETE | `/api/records/{id}` | 删除记录及文件 |
| GET  | `/v1/models` | 模型列表 |
| GET  | `/health` | 健康检查 |

### curl 调用示例

> 配置了 `FUNASR_TOKEN` 后，所有受保护接口需带 `Authorization: Bearer <token>` 请求头（`/health`、`/docs`、`/v1/models` 除外）。

```bash
TOKEN=wushuo1998

# 语音识别 + 说话人分离
curl http://localhost:11800/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.wav \
  -F model=sensevoice \
  -F spk=true \
  -F response_format=verbose_json

# 仅识别
curl http://localhost:11800/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@audio.mp3 -F model=paraformer

# 列历史记录
curl http://localhost:11800/api/records \
  -H "Authorization: Bearer $TOKEN"

# 删除记录
curl -X DELETE http://localhost:11800/api/records/{id} \
  -H "Authorization: Bearer $TOKEN"
```

## Python SDK 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11800/v1", api_key="wushuo1998")

result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
    extra_body={"spk": True},
)
for seg in result.segments:
    print(seg.speaker, seg.start, seg.end, seg.text)
```

## 模型下载

首次启动会自动从 ModelScope 下载模型（约 1.2GB），国内速度较快。也可预下载到 `models/` 目录：

```bash
python -m pip install modelscope
modelscope download --model iic/SenseVoiceSmall --local_dir ./models/SenseVoiceSmall
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./models/cam++
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-4.0.1 --local_dir ./models/fsmn-vad
```

## 资源占用

| 组件 | 内存 | 说明 |
|---|---|---|
| SenseVoice (ASR) | ~600MB | 多语言识别 |
| CAM++ (说话人) | ~400MB | 惰性加载，仅 spk=true 时启用 |
| Paraformer (备用) | ~400MB | 可选中文模型 |
| 每个 worker | ~1GB | 额外复制 |

1 worker 约需 **2GB** 内存，多 worker 按比例增加。