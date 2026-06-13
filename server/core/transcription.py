"""Unified transcription service.

All public APIs should call this module and then adapt the canonical result to
their protocol. The canonical result always contains paragraphs and sentences
with start/end timestamps.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import HTTPException

from server.core.audio import pcm_duration_ms
from server.core.inference import _generate_sync, infer_emotion, run_blocking
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.core.schemas import TranscriptionConfig, WarningItem
from server.core.speaker_db import match_segments
from server.models.capabilities import capabilities_for_model
from server.models.config import DEFAULT_BATCH_SIZE_S, MODEL_NAME
from server.models.registry import ModelRegistry


def _feature_warning(feature: str, message: str) -> WarningItem:
    return WarningItem(code="feature_not_supported", feature=feature, message=message)


def validate_capabilities(config: TranscriptionConfig) -> list[WarningItem]:
    """Validate requested features for the configured deployment model."""
    caps = capabilities_for_model()
    warnings: list[WarningItem] = []

    for feature in config.requested_features():
        if caps.get(feature, False):
            continue
        warning = _feature_warning(feature, f"当前模型不支持 {feature}")
        if config.fallback == "auto":
            warnings.append(warning)
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unsupported_feature",
                    "feature": feature,
                    "message": warning.message,
                    "capabilities": caps,
                },
            )

    if config.features.speaker_match and not config.features.speaker_group:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_speaker_group",
                "message": "speaker_match 需要提供 speaker_group/group_id",
            },
        )

    return warnings


def _disable_unsupported_features(
    config: TranscriptionConfig, warnings: list[WarningItem]
) -> None:
    disabled = {w.feature for w in warnings}
    if "diarization" in disabled:
        config.features.diarization = False
    if "speaker_match" in disabled:
        config.features.speaker_match = False
    if "emotion" in disabled:
        config.features.emotion = False
    if "events" in disabled:
        config.features.events = False
    if "punctuation" in disabled:
        config.features.punctuation = False
    if "hotwords" in disabled:
        config.hotwords = ""
    if "words" in disabled:
        config.features.words = False
    if "raw" in disabled:
        config.features.raw = False


def _speaker_object(sentence: dict[str, Any], group_id: str = "") -> dict[str, Any] | None:
    raw_id = sentence.get("speaker_id")
    if raw_id is None:
        raw_id = sentence.get("spk")
    if raw_id is None:
        return None

    speaker = {"id": f"speaker_{raw_id}"}
    if "speaker" in sentence:
        speaker["name"] = sentence["speaker"]
    if "speaker_score" in sentence:
        speaker["score"] = sentence["speaker_score"]
    if group_id:
        speaker["group_id"] = group_id
    return speaker


def _sentence_text(seg: dict[str, Any]) -> str:
    return clean_text(str(seg.get("text") or seg.get("sentence") or ""))


def _milliseconds_to_seconds(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number / 1000.0


def _sentences_from_raw(
    raw: dict[str, Any],
    config: TranscriptionConfig,
    *,
    audio_duration: float,
) -> tuple[list[dict[str, Any]], list[WarningItem]]:
    warnings: list[WarningItem] = []
    raw_sentences = raw.get("sentence_info") or []

    if not raw_sentences:
        text = clean_text(raw.get("text", ""))
        if text:
            warnings.append(
                WarningItem(
                    code="sentence_timestamps_unavailable",
                    feature="sentence_timestamps",
                    message="模型未返回 sentence_info，已使用全文生成单句时间轴",
                )
            )
        return ([{
            "id": 0,
            "paragraph_id": 0,
            "start": 0.0,
            "end": round(audio_duration, 3),
            "text": text,
        }] if text else []), warnings

    sentences = []
    raw_text = raw.get("text", "")
    top_emotion = extract_emotion(raw_text)
    top_events = extract_events(raw_text)
    for idx, seg in enumerate(raw_sentences):
        sentence = {
            "id": idx,
            "paragraph_id": 0,
            "start": round(_milliseconds_to_seconds(seg.get("start")), 3),
            "end": round(_milliseconds_to_seconds(seg.get("end")), 3),
            "text": _sentence_text(seg),
        }
        speaker = _speaker_object(seg, config.features.speaker_group)
        if config.features.diarization and speaker:
            sentence["speaker"] = speaker

        seg_text = str(seg.get("text") or seg.get("sentence") or "")
        if config.features.emotion:
            sentence["emotion"] = extract_emotion(seg_text) or top_emotion
        if config.features.events:
            sentence["events"] = extract_events(seg_text) or top_events
        sentences.append(sentence)
    return sentences, warnings


def _build_paragraphs(
    sentences: list[dict[str, Any]],
    *,
    max_gap_seconds: float,
    max_sentences: int,
) -> list[dict[str, Any]]:
    if not sentences:
        return []

    paragraphs: list[dict[str, Any]] = []
    current_ids: list[int] = []
    paragraph_id = 0

    def flush():
        nonlocal paragraph_id, current_ids
        if not current_ids:
            return
        items = [sentences[i] for i in current_ids]
        for item in items:
            item["paragraph_id"] = paragraph_id
        paragraphs.append({
            "id": paragraph_id,
            "start": items[0]["start"],
            "end": items[-1]["end"],
            "text": "".join(item["text"] for item in items).strip(),
            "sentence_ids": [item["id"] for item in items],
        })
        paragraph_id += 1
        current_ids = []

    for idx, sentence in enumerate(sentences):
        if current_ids:
            prev = sentences[current_ids[-1]]
            gap = sentence["start"] - prev["end"]
            if gap > max_gap_seconds or len(current_ids) >= max_sentences:
                flush()
        current_ids.append(idx)
    flush()
    return paragraphs


def _applied_features(config: TranscriptionConfig, result: dict[str, Any]) -> list[str]:
    applied = ["asr", "sentence_timestamps", "paragraphs"]
    if config.features.diarization:
        applied.append("diarization")
    if config.features.speaker_match:
        applied.append("speaker_match")
    if config.features.emotion and "emotion" in result:
        applied.append("emotion")
    if config.features.events and "events" in result:
        applied.append("events")
    if config.features.punctuation:
        applied.append("punctuation")
    if config.hotwords:
        applied.append("hotwords")
    if config.features.raw:
        applied.append("raw")
    return applied


async def transcribe_pcm(
    pcm_bytes: bytes,
    config: TranscriptionConfig,
    *,
    source: str = "upload",
) -> dict[str, Any]:
    """Run ASR and return the canonical transcription result."""
    warnings = validate_capabilities(config)
    _disable_unsupported_features(config, warnings)

    registry = ModelRegistry.get_instance()
    model = registry.get(with_spk=config.features.diarization)
    gen_kwargs = {
        "batch_size_s": DEFAULT_BATCH_SIZE_S,
        "batch_size_threshold_s": 0,
        "language": config.language or "auto",
        "use_itn": True,
        "merge_vad": True,
        "merge_length_s": 15,
        "sentence_timestamp": True,
    }
    if config.hotwords:
        gen_kwargs["hotword"] = config.hotwords
    if config.features.diarization:
        gen_kwargs["output_timestamp"] = False
        gen_kwargs["return_spk_res"] = True

    started = time.time()
    result_list = await run_blocking(
        _generate_sync,
        model,
        pcm_bytes,
        sem=registry.sem_asr_offline,
        **gen_kwargs,
    )
    elapsed = time.time() - started
    raw = result_list[0] if result_list else {}
    raw_text = str(raw.get("text") or "")
    speaker_match_summary = None

    if config.features.speaker_match and raw.get("sentence_info"):
        speaker_match_summary = match_segments(
            raw["sentence_info"],
            pcm_bytes,
            config.features.speaker_group,
            registry.get_aux("sv"),
        )

    audio_duration = round(pcm_duration_ms(pcm_bytes) / 1000.0, 3)
    sentences, sentence_warnings = _sentences_from_raw(
        raw,
        config,
        audio_duration=audio_duration,
    )
    warnings.extend(sentence_warnings)
    paragraphs = (
        _build_paragraphs(
            sentences,
            max_gap_seconds=config.paragraph.max_gap_seconds,
            max_sentences=config.paragraph.max_sentences,
        )
        if config.paragraph.enabled
        else []
    )

    text = clean_text(raw_text) or "".join(s["text"] for s in sentences)
    result = {
        "id": f"tr_{uuid.uuid4().hex[:16]}",
        "object": "transcription",
        "model": MODEL_NAME,
        "source": source,
        "duration": audio_duration,
        "processing_time": round(elapsed, 3),
        "text": text,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "paragraphs": paragraphs,
        "sentences": sentences,
    }

    if config.language:
        result["language"] = config.language
    if config.features.emotion:
        result["emotion"] = extract_emotion(raw_text)
        if result["emotion"] is None:
            try:
                emotion_raw = await infer_emotion(pcm_bytes)
                labels = emotion_raw.get("labels") or emotion_raw.get("label")
                scores = emotion_raw.get("scores") or emotion_raw.get("score")
                if isinstance(labels, list) and labels:
                    result["emotion"] = labels[0]
                    result["emotion_score"] = scores[0] if isinstance(scores, list) and scores else None
                elif isinstance(labels, str):
                    result["emotion"] = labels
            except Exception as exc:
                warnings.append(
                    WarningItem(
                        code="emotion_failed",
                        feature="emotion",
                        message=f"情感识别失败: {exc}",
                    )
                )
    if config.features.events:
        result["events"] = extract_events(raw_text)
    if config.features.speaker_group:
        result["speaker_group"] = config.features.speaker_group
    if speaker_match_summary is not None:
        result["speaker_match"] = speaker_match_summary
    if config.features.raw:
        result["raw"] = raw

    result["features"] = {
        "requested": config.requested_features(),
        "applied": _applied_features(config, result),
        "warnings": [w.to_dict() for w in warnings],
    }
    return result
