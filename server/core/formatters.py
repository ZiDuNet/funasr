"""Output format adapters for canonical transcription results."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse, PlainTextResponse, Response


def _timestamp(seconds: float, sep: str = ",") -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def to_srt(result: dict[str, Any]) -> str:
    blocks = []
    for idx, sentence in enumerate(result.get("sentences", []), start=1):
        start = _timestamp(float(sentence.get("start", 0)), ",")
        end = _timestamp(float(sentence.get("end", 0)), ",")
        blocks.append(f"{idx}\n{start} --> {end}\n{sentence.get('text', '')}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(result: dict[str, Any]) -> str:
    blocks = ["WEBVTT", ""]
    for sentence in result.get("sentences", []):
        start = _timestamp(float(sentence.get("start", 0)), ".")
        end = _timestamp(float(sentence.get("end", 0)), ".")
        blocks.append(f"{start} --> {end}\n{sentence.get('text', '')}\n")
    return "\n".join(blocks)


def canonical_to_openai(result: dict[str, Any], *, verbose: bool = False) -> dict[str, Any]:
    """Return an OpenAI-shaped JSON object while keeping FunASR extensions."""
    if not verbose:
        return {"text": result.get("text", "")}

    payload = {
        "task": "transcribe",
        "language": result.get("language", "auto"),
        "duration": result.get("duration", 0),
        "text": result.get("text", ""),
        "segments": [],
    }
    for sentence in result.get("sentences", []):
        seg = {
            "id": sentence.get("id"),
            "start": sentence.get("start", 0),
            "end": sentence.get("end", 0),
            "text": sentence.get("text", ""),
        }
        if "speaker" in sentence:
            seg["speaker"] = sentence["speaker"]
        if "emotion" in sentence:
            seg["emotion"] = sentence["emotion"]
        if "events" in sentence:
            seg["events"] = sentence["events"]
        payload["segments"].append(seg)

    for key in (
        "id",
        "object",
        "model",
        "processing_time",
        "paragraph_count",
        "sentence_count",
        "paragraphs",
        "sentences",
        "emotion",
        "emotion_score",
        "events",
        "speaker_group",
        "features",
    ):
        if key in result:
            payload[key] = result[key]
    return payload


def response_for_format(
    result: dict[str, Any],
    *,
    response_format: str = "json",
    openai: bool = False,
) -> Response:
    if response_format == "text":
        return PlainTextResponse(result.get("text", ""))
    if response_format == "srt":
        return PlainTextResponse(to_srt(result), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result), media_type="text/vtt")
    if openai:
        return JSONResponse(
            canonical_to_openai(
                result,
                verbose=response_format == "verbose_json",
            )
        )
    return JSONResponse(result)
