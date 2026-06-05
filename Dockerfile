# ── FunASR All-in-One ──────────────────────────
# 支持: OpenAI API / HTTP REST / WebSocket 流式 / MCP / Web UI
# ────────────────────────────────────────────────

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FUNASR_DEVICE=cpu \
    FUNASR_PORT=8000

WORKDIR /app

# 系统依赖：ffmpeg（音频转码）、git（funasr 下载模型）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY server/ ./server/
COPY web/ ./web/

# 模型目录（由 docker-compose 挂载）
RUN mkdir -p /app/models
ENV MODELSCOPE_CACHE=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "server.main"]
