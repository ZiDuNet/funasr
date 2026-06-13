"""API 请求解析和标准响应辅助工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


SUPPORTED_RESPONSE_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


@dataclass
class WarningItem:
    code: str
    message: str
    feature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"code": self.code, "message": self.message}
        if self.feature:
            data["feature"] = self.feature
        return data


@dataclass
class FeatureConfig:
    diarization: bool = False
    speaker_match: bool = False
    speaker_group: str = ""
    emotion: bool = False
    events: bool = False
    punctuation: bool = True
    words: bool = False
    raw: bool = False


@dataclass
class ParagraphConfig:
    enabled: bool = True
    max_gap_seconds: float = 1.2
    max_sentences: int = 6


@dataclass
class TranscriptionConfig:
    language: str = "auto"
    hotwords: str = ""
    fallback: Literal["error", "auto"] = "error"
    response_format: str = "json"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    paragraph: ParagraphConfig = field(default_factory=ParagraphConfig)

    def requested_features(self) -> list[str]:
        requested = ["asr", "sentence_timestamps", "paragraphs"]
        f = self.features
        if f.diarization:
            requested.append("diarization")
        if f.speaker_match:
            requested.append("speaker_match")
        if f.emotion:
            requested.append("emotion")
        if f.events:
            requested.append("events")
        if f.punctuation:
            requested.append("punctuation")
        if self.hotwords:
            requested.append("hotwords")
        if f.words:
            requested.append("words")
        if f.raw:
            requested.append("raw")
        return requested


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config 必须是合法 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config 必须是 JSON 对象")
    return parsed


def _stringify_hotwords(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def build_config(
    *,
    config_json: str | None = None,
    language: str | None = None,
    speaker_diarization: bool | None = None,
    diarization: bool | None = None,
    speaker_group: str | None = None,
    speaker_match: bool | None = None,
    emotion: bool | None = None,
    events: bool | None = None,
    punctuation: bool | None = None,
    hotwords: str | None = None,
    words: bool | None = None,
    raw: bool | None = None,
    fallback: str | None = None,
    response_format: str | None = None,
) -> TranscriptionConfig:
    """从 JSON config 或平铺表单字段构造转写配置。"""
    payload = parse_json_object(config_json)

    features_obj = payload.get("features") or {}
    options_obj = payload.get("options") or {}
    paragraph_obj = options_obj.get("paragraph") or payload.get("paragraph") or {}

    speaker_match_obj = features_obj.get("speaker_match")
    if isinstance(speaker_match_obj, dict):
        json_speaker_match = parse_bool(speaker_match_obj.get("enabled"), False)
        json_speaker_group = str(speaker_match_obj.get("group_id") or "")
    else:
        json_speaker_match = parse_bool(speaker_match_obj, False)
        json_speaker_group = ""

    flat_diarization = diarization if diarization is not None else speaker_diarization
    cfg = TranscriptionConfig()
    cfg.language = (
        language
        or options_obj.get("language")
        or payload.get("language")
        or "auto"
    )
    cfg.hotwords = _stringify_hotwords(
        hotwords
        if hotwords not in (None, "")
        else options_obj.get("hotwords", payload.get("hotwords", ""))
    )
    cfg.fallback = (fallback or payload.get("fallback") or "error").lower()
    if cfg.fallback not in {"error", "auto"}:
        raise ValueError("fallback 必须是 'error' 或 'auto'")

    cfg.response_format = (
        response_format
        or payload.get("response_format")
        or "json"
    )
    if cfg.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(
            "response_format 必须是以下之一："
            + ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS))
        )

    f = cfg.features
    f.diarization = parse_bool(
        flat_diarization
        if flat_diarization is not None
        else features_obj.get("diarization", features_obj.get("speaker_diarization")),
        False,
    )
    f.speaker_group = (
        speaker_group
        or json_speaker_group
        or str(features_obj.get("speaker_group") or payload.get("speaker_group") or "")
    )
    f.speaker_match = parse_bool(
        speaker_match if speaker_match is not None else json_speaker_match,
        bool(f.speaker_group),
    )
    if f.speaker_match:
        f.diarization = True

    f.emotion = parse_bool(
        emotion if emotion is not None else features_obj.get("emotion"),
        False,
    )
    f.events = parse_bool(
        events if events is not None else features_obj.get("events"),
        False,
    )
    f.punctuation = parse_bool(
        punctuation if punctuation is not None else features_obj.get("punctuation"),
        True,
    )
    f.words = parse_bool(words if words is not None else features_obj.get("words"), False)
    f.raw = parse_bool(raw if raw is not None else features_obj.get("raw"), False)

    p = cfg.paragraph
    p.enabled = parse_bool(paragraph_obj.get("enabled"), True)
    if "max_gap_seconds" in paragraph_obj:
        p.max_gap_seconds = float(paragraph_obj["max_gap_seconds"])
    if "max_sentences" in paragraph_obj:
        p.max_sentences = int(paragraph_obj["max_sentences"])

    return cfg


def config_to_dict(config: TranscriptionConfig) -> dict[str, Any]:
    data = {
        "language": config.language,
        "fallback": config.fallback,
        "response_format": config.response_format,
        "features": {
            "diarization": config.features.diarization,
            "speaker_match": config.features.speaker_match,
            "emotion": config.features.emotion,
            "events": config.features.events,
            "punctuation": config.features.punctuation,
            "words": config.features.words,
            "raw": config.features.raw,
        },
        "paragraph": {
            "enabled": config.paragraph.enabled,
            "max_gap_seconds": config.paragraph.max_gap_seconds,
            "max_sentences": config.paragraph.max_sentences,
        },
    }
    if config.features.speaker_group:
        data["features"]["speaker_group"] = config.features.speaker_group
    if config.hotwords:
        data["hotwords"] = config.hotwords
    return data
