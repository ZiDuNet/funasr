"""FunASR 模型配置

模型使用 ModelScope ID，首次运行自动下载到 MODELSCOPE_CACHE 目录。
挂载 ./models:/root/.cache/modelscope 即可持久化，无需手动下载脚本。
"""

import os

# ── 设备配置 ────────────────────────────────
DEVICE = os.environ.get("FUNASR_DEVICE", "cpu")
NGPU = 1 if DEVICE.startswith("cuda") else 0
NCPU = int(os.environ.get("FUNASR_NCPU", "4"))

# ── 模型缓存 ────────────────────────────────
# 挂载此目录即可持久化自动下载的模型
MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

# ── 数据目录（任务结果、音频文件、声纹库）─────
DATA_DIR = os.environ.get("FUNASR_DATA_DIR", "/app/data")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")

# ── 并发控制 ────────────────────────────────
WORKER_THREADS = int(os.environ.get("FUNASR_WORKER_THREADS", "8"))
CONCURRENT_VAD = int(os.environ.get("FUNASR_CONCURRENT_VAD", "4"))
CONCURRENT_ASR_ONLINE = int(os.environ.get("FUNASR_CONCURRENT_ASR_ONLINE", "4"))
CONCURRENT_ASR_OFFLINE = int(os.environ.get("FUNASR_CONCURRENT_ASR_OFFLINE", "2"))
CONCURRENT_PUNC = int(os.environ.get("FUNASR_CONCURRENT_PUNC", "1"))
CONCURRENT_SV = int(os.environ.get("FUNASR_CONCURRENT_SV", "1"))

# ── 异步任务 ────────────────────────────────
ASYNC_THRESHOLD_SEC = int(os.environ.get("FUNASR_ASYNC_THRESHOLD", "60"))
MAX_TASKS = int(os.environ.get("FUNASR_MAX_TASKS", "1000"))

# ── 自动清理（天数，0 表示不清理）─────────────
DATA_TTL_DAYS = int(os.environ.get("FUNASR_DATA_TTL_DAYS", "7"))

# ── 模型定义（用 ModelScope ID，自动下载）─────
MODEL_CONFIGS = {
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
    },
}

# WebSocket 流式模型
STREAMING_MODEL = {
    "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "model_revision": "v2.0.4",
}

# 独立模型（WebSocket 手动 pipeline 使用）
VAD_MODEL = {
    "model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "model_revision": "v2.0.4",
}

PUNC_MODEL = {
    "model": "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    "model_revision": "v2.0.4",
}

SV_MODEL = {
    "model": "iic/speech_campplus_sv_zh-cn_16k-common",
}
