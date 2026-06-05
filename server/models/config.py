"""FunASR 模型配置

通过 .env 的 MODEL=xxx 选择离线模型，API 不暴露模型切换。
模型首次运行时自动下载，挂载 ./models 持久化缓存。
"""

import os
import logging

logger = logging.getLogger(__name__)

# ── 设备 ─────────────────────────────────────
DEVICE = os.environ.get("FUNASR_DEVICE", "cpu")
NGPU = 1 if DEVICE.startswith("cuda") else 0
NCPU = int(os.environ.get("FUNASR_NCPU", "4"))

# ── 模型选择（配置文件中切换，API 不暴露）────────
MODEL = os.environ.get("MODEL", "sensevoice")

# ── 功能开关 ─────────────────────────────────
PRELOAD_ALL = os.environ.get("PRELOAD_ALL", "true").lower() == "true"
ENABLE_STREAMING = os.environ.get("ENABLE_STREAMING", "true").lower() == "true"
ENABLE_MCP = os.environ.get("ENABLE_MCP", "true").lower() == "true"

# ── 目录 ────────────────────────────────────
MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
DATA_DIR = os.environ.get("FUNASR_DATA_DIR", "/app/data")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")

# ── 并发 ────────────────────────────────────
WORKER_THREADS = int(os.environ.get("FUNASR_WORKER_THREADS", "8"))
CONCURRENT_VAD = int(os.environ.get("FUNASR_CONCURRENT_VAD", "4"))
CONCURRENT_ASR_ONLINE = int(os.environ.get("FUNASR_CONCURRENT_ASR_ONLINE", "4"))
CONCURRENT_ASR_OFFLINE = int(os.environ.get("FUNASR_CONCURRENT_ASR_OFFLINE", "2"))
CONCURRENT_PUNC = int(os.environ.get("FUNASR_CONCURRENT_PUNC", "1"))
CONCURRENT_SV = int(os.environ.get("FUNASR_CONCURRENT_SV", "1"))
CONCURRENT_EMOTION = int(os.environ.get("FUNASR_CONCURRENT_EMOTION", "1"))

# ── 任务 ────────────────────────────────────
ASYNC_THRESHOLD_SEC = int(os.environ.get("FUNASR_ASYNC_THRESHOLD", "60"))
MAX_TASKS = int(os.environ.get("FUNASR_MAX_TASKS", "1000"))
DATA_TTL_DAYS = int(os.environ.get("FUNASR_DATA_TTL_DAYS", "7"))

# ── 推理 ────────────────────────────────────
DEFAULT_BATCH_SIZE_S = int(os.environ.get("FUNASR_BATCH_SIZE_S", "300"))
DEFAULT_BATCH_THRESHOLD_S = int(os.environ.get("FUNASR_BATCH_THRESHOLD_S", "60"))

# ═══════════════════════════════════════════════════
#  离线模型预设（部署时 .env 里选一个）
# ═══════════════════════════════════════════════════

_PRESETS = {
    "sensevoice": {
        "name": "SenseVoiceSmall",
        "config": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    },
    "paraformer": {
        "name": "Paraformer-zh",
        "config": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
        },
    },
    "fun-asr-nano": {
        "name": "Fun-ASR-Nano",
        "config": {
            "model": "FunAudioLLM/Fun-ASR-Nano-2512",  # 魔搭 ID，无需 hub="hf"
            "trust_remote_code": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    },
}

# 根据 MODEL 环境变量选择
if MODEL in _PRESETS:
    ASR_CONFIG = _PRESETS[MODEL]["config"]
    ASR_CONFIG_WITH_SPK = {**ASR_CONFIG, "spk_model": "cam++"}
    MODEL_NAME = _PRESETS[MODEL]["name"]
    logger.info(f"离线模型: {MODEL_NAME}")
else:
    # 兜底：直接使用 MODEL 作为模型 ID
    ASR_CONFIG = {"model": MODEL}
    ASR_CONFIG_WITH_SPK = {"model": MODEL, "spk_model": "cam++"}
    MODEL_NAME = MODEL
    logger.info(f"离线模型（自定义）: {MODEL}")

# ── 流式模型 ─────────────────────────────────
STREAMING_CONFIG = {
    "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "model_revision": "v2.0.4",
}

# ── 辅助模型 ─────────────────────────────────
VAD_CONFIG = {"model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "model_revision": "v2.0.4"}
PUNC_CONFIG = {"model": "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "model_revision": "v2.0.4"}
SV_CONFIG = {"model": "iic/speech_campplus_sv_zh-cn_16k-common"}
EMOTION_CONFIG = {"model": "iic/emotion2vec_plus_large"}
