# ═══════════════════════════════════════════════════
#  FunASR All-in-One Dockerfile
#  ──────────────────────────────────────────────
#  包含: OpenAI API / HTTP REST / WebSocket / MCP / Web UI
#  ──────────────────────────────────────────────
#  构建: docker compose build
#  运行: docker compose up -d
#  ═══════════════════════════════════════════════════

FROM python:3.10-slim

# ── 元信息 ──────────────────────────────────────
LABEL maintainer="ZiDuNet"
LABEL description="FunASR 统一语音识别服务 - OpenAI API + WebSocket + MCP + Web UI"

# ── 系统依赖 ────────────────────────────────────
# ffmpeg:  音频格式转码（mp3/mp4→PCM）
# git:     funasr 从魔搭下载模型时需要
# libsndfile1: 音频文件读写
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ─────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── 应用代码 ────────────────────────────────────
COPY server/ ./server/
COPY web/    ./web/

# ── 数据目录 ────────────────────────────────────
# /root/.cache/modelscope : 模型缓存（挂载持久化）
# /app/data               : 任务结果 + 声纹库（挂载持久化）
RUN mkdir -p /app/data /root/.cache/modelscope
ENV MODELSCOPE_CACHE=/root/.cache/modelscope

# ── 环境变量（docker-compose 中可覆盖）────────────
ENV FUNASR_DEVICE=cpu
ENV FUNASR_PORT=17767
ENV MODEL=sensevoice
ENV PRELOAD_ALL=true
ENV ENABLE_STREAMING=true
ENV ENABLE_MCP=true

# ── 端口 ────────────────────────────────────────
EXPOSE 17767

# ── 健康检查 ────────────────────────────────────
# 容器启动后 120s 开始检查，每 30s 一次，失败 3 次重启
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:17767/health', timeout=3)" || exit 1

# ── 启动 ────────────────────────────────────────
CMD ["sh", "-c", "python -m server.main"]
