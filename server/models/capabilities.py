"""Model capability registry.

This module is the product contract between FunASR's flexible model zoo and
the public API. Routes should validate requests against these capabilities
instead of guessing whether a model/feature combination will work.
"""

from __future__ import annotations

from copy import deepcopy

from server.models.config import ENABLE_STREAMING, MODEL, MODEL_NAME


FEATURES = {
    "asr": "语音识别",
    "sentence_timestamps": "句子级时间戳",
    "paragraphs": "根据句子时间轴生成段落",
    "diarization": "使用 cam++ 进行匿名说话人分离",
    "speaker_match": "使用声纹组进行注册说话人匹配",
    "emotion": "情感标签；SenseVoice 使用内置标签，其他模型使用 emotion2vec",
    "events": "SenseVoice 音频事件标签",
    "punctuation": "标点恢复；可能由 ASR 模型内置或 ct-punc 提供",
    "hotwords": "热词增强",
    "words": "词级时间戳",
    "raw": "原始 FunASR 输出",
    "streaming": "专用流式 pipeline 提供在线/离线/2pass 识别",
}


_BASE = {
    "asr": True,
    "offline": True,
    "sentence_timestamps": True,
    "paragraphs": True,
    "diarization": True,
    "speaker_match": True,
    "emotion": True,
    "events": False,
    "punctuation": True,
    "hotwords": False,
    "words": False,
    "raw": True,
    "streaming": ENABLE_STREAMING,
    "streaming_pipeline": "paraformer-large-online + fsmn-vad",
}


_MODEL_CAPABILITIES = {
    "sensevoice": {
        **_BASE,
        "events": True,
        "hotwords": True,
        "emotion_source": "sensevoice",
        "events_source": "sensevoice",
    },
    "paraformer": {
        **_BASE,
        "hotwords": True,
        "emotion_source": "emotion2vec",
    },
    "fun-asr-nano": {
        **_BASE,
        "emotion_source": "emotion2vec",
    },
    "qwen3-asr": {
        **_BASE,
        "emotion_source": "emotion2vec",
    },
    "glm-asr-nano": {
        **_BASE,
        "emotion_source": "emotion2vec",
    },
    "whisper-large-v3": {
        **_BASE,
        "emotion_source": "emotion2vec",
    },
    "whisper-large-v3-turbo": {
        **_BASE,
        "emotion_source": "emotion2vec",
    },
}


def capabilities_for_model(model_key: str | None = None) -> dict:
    """Return the capability contract for a configured model key."""
    key = model_key or MODEL
    caps = deepcopy(_MODEL_CAPABILITIES.get(key, _BASE))
    caps["model"] = key
    caps["name"] = MODEL_NAME if key == MODEL else key
    caps["features"] = FEATURES
    return caps


def list_model_capabilities() -> list[dict]:
    """Return capabilities for all built-in model presets."""
    return [capabilities_for_model(key) for key in _MODEL_CAPABILITIES]
