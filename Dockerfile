# ═══════════════════════════════════════════════════
#  FunASR All-in-One Dockerfile
#  ──────────────────────────────────────────────
#  包含: OpenAI API / HTTP REST / WebSocket / MCP / Web UI
#  ──────────────────────────────────────────────
#  构建: docker compose build
#  运行: docker compose up -d
#  ═══════════════════════════════════════════════════

# 默认使用阿里云容器镜像加速，海外构建可覆盖: --build-arg BASE_IMAGE=python:3.10-slim
ARG BASE_IMAGE=registry.cn-hangzhou.aliyuncs.com/docker-library/python:3.10-slim
FROM ${BASE_IMAGE}

# ── 元信息 ──────────────────────────────────────
LABEL maintainer="ZiDuNet"
LABEL description="FunASR 统一语音识别服务 - OpenAI API + WebSocket + MCP + Web UI"

# ── 系统依赖（阿里云 apt 镜像）───────────────────
# ffmpeg:  音频格式转码（mp3/mp4→PCM）
# git:     funasr 从魔搭下载模型时需要
# libsndfile1: 音频文件读写
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖（阿里云 pip 镜像）─────────────────
WORKDIR /app
COPY requirements.txt .

# PyTorch: CUDA 版本支持 CPU 回退，单镜像 CPU/GPU 通用
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir torch torchaudio \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/

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
ENV MODEL=fun-asr-nano
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
