"""FunASR 模型配置

通过 .env 的 MODEL=xxx 选择离线模型，API 不暴露模型切换。
模型首次使用时自动从魔搭下载，挂载 ./models 持久化缓存。
"""

import os
import logging

logger = logging.getLogger(__name__)

# ── 设备 ─────────────────────────────────────
DEVICE = os.environ.get("FUNASR_DEVICE", "cpu")
NGPU = 1 if DEVICE.startswith("cuda") else 0
NCPU = int(os.environ.get("FUNASR_NCPU", "4"))

# ── 模型选择（配置文件中切换，API 不暴露）────────
MODEL = os.environ.get("MODEL", "fun-asr-nano")

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
#  首次使用自动从魔搭下载，后续从缓存加载
# ═══════════════════════════════════════════════════

_PRESETS = {
    # ── 轻量级 ────────────────────────────────
    "sensevoice": {
        "name": "SenseVoiceSmall",
        "desc": "ASR + 情感 + 事件，中英日韩粤，234M",
        "config": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    },
    "paraformer": {
        "name": "Paraformer-zh",
        "desc": "中文生产级 ASR + 字级时间戳，220M",
        "config": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
        },
    },
    "fun-asr-nano": {
        "name": "Fun-ASR-Nano",
        "desc": "LLM-based ASR，31 语言，800M",
        "config": {
            "model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "trust_remote_code": True,
            "remote_code": "./model.py",
            "hub": "hf",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    },
    # ── 大模型（需 GPU）──────────────────────
    "qwen3-asr": {
        "name": "Qwen3-ASR-1.7B",
        "desc": "52 语言，1.7B，需 GPU + bf16",
        "config": {
            "model": "Qwen/Qwen3-ASR-1.7B",
            "trust_remote_code": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": "ct-punc",
        },
        "dtype": "bf16",
    },
    "glm-asr-nano": {
        "name": "GLM-ASR-Nano-2512",
        "desc": "17 语言，1.5B，需 GPU + bf16",
        "config": {
            "model": "ZhipuAI/GLM-ASR-Nano-2512",
            "trust_remote_code": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": "ct-punc",
        },
        "dtype": "bf16",
    },
    # ── Whisper ──────────────────────────────
    "whisper-large-v3": {
        "name": "Whisper-large-v3",
        "desc": "识别 + 翻译，多语言，1550M",
        "config": {
            "model": "iic/Whisper-large-v3",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": "ct-punc",
        },
    },
    "whisper-large-v3-turbo": {
        "name": "Whisper-large-v3-turbo",
        "desc": "识别 + 翻译，多语言，809M",
        "config": {
            "model": "iic/Whisper-large-v3-turbo",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": "ct-punc",
        },
    },
}

# 根据 MODEL 环境变量选择
if MODEL in _PRESETS:
    preset = _PRESETS[MODEL]
    ASR_CONFIG = preset["config"]
    ASR_CONFIG_WITH_SPK = {**ASR_CONFIG, "spk_model": "cam++"}
    MODEL_NAME = preset["name"]

    # 大模型需要 dtype=bf16（仅 GPU）
    if preset.get("dtype") == "bf16":
        if DEVICE.startswith("cuda"):
            ASR_CONFIG["dtype"] = "bf16"
            logger.info(f"离线模型: {MODEL_NAME} (bf16, GPU)")
        else:
            ASR_CONFIG["dtype"] = "float32"
            logger.warning(f"离线模型: {MODEL_NAME} (float32, CPU 模式，大模型建议使用 GPU)")
    else:
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
