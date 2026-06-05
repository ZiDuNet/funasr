"""OpenAI 兼容 API — 纯同步，支持 speaker_group 参数"""

import os
import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.core.speaker_db import match_speaker
from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="音频文件"),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    speaker_diarization: bool = Form(default=False),
    speaker_group: Optional[str] = Form(default=None),
    emotion: bool = Form(default=False),
    events: bool = Form(default=False),
    punctuation: bool = Form(default=True),
    hotwords: Optional[str] = Form(default=None),
):
    """OpenAI 兼容音频转写 — 同步直接返回"""
    registry = ModelRegistry.get_instance()

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    content = await file.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        pcm_bytes = await convert_to_pcm(tmp_path)

        asr_model = registry.get(model)
        gen_kwargs = {
            "batch_size_s": getattr(registry, "batch_size_s", 300),
            "language": language or "auto",
            "use_itn": True,
        }
        # SenseVoice 官方推荐：合并短句 + 防 OOM
        if model == "sensevoice":
            gen_kwargs["merge_vad"] = True
            gen_kwargs["merge_length_s"] = 15
            gen_kwargs["batch_size_threshold_s"] = 60

        t0 = time.time()
        result_list = await run_blocking(
            _generate_sync, asr_model, pcm_bytes,
            sem=registry.sem_asr_offline,
            **gen_kwargs,
        )
        elapsed = time.time() - t0

        if not result_list:
            return JSONResponse({"text": ""})

        raw = result_list[0]
        raw_text = raw.get("text", "")

        text = clean_text(raw_text)

        if response_format == "verbose_json":
            emo = extract_emotion(raw_text) if emotion else None
            evt = extract_events(raw_text) if events else []

            # 提取整段声纹（用于匹配）
            spk_embedding = raw.get("spk_embedding")
            if spk_embedding is not None and hasattr(spk_embedding, "cpu"):
                spk_embedding = spk_embedding[0].cpu().numpy().tolist()

            segments = []
            if "sentence_info" in raw:
                for seg in raw["sentence_info"]:
                    s = {
                        "start": seg.get("start", 0) / 1000.0,
                        "end": seg.get("end", 0) / 1000.0,
                        "text": clean_text(seg.get("text", "")),
                    }
                    if "spk" in seg:
                        s["speaker_id"] = seg["spk"]
                        if speaker_group and spk_embedding:
                            matched = match_speaker(speaker_group, spk_embedding)
                            if matched:
                                s["speaker"] = matched
                    segments.append(s)

            resp = {
                "text": text,
                "segments": segments,
                "language": language or "auto",
                "duration": round(elapsed, 3),
                "model": model,
            }
            if speaker_group:
                resp["speaker_group"] = speaker_group
            if emo:
                resp["emotion"] = emo
            if evt:
                resp["events"] = evt
            return JSONResponse(resp)
        else:
            return JSONResponse({"text": text})

    except Exception as e:
        logger.error(f"转写错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/v1/models")
async def list_models():
    registry = ModelRegistry.get_instance()
    models = [
        {"id": n, "object": "model", "created": 1700000000, "owned_by": "funasr",
         "ready": any(m in registry.loaded_models() for m in ("sensevoice", "paraformer")),
         "description": desc}
        for n, desc in [
            ("sensevoice", "ASR + 情感 + 事件 (5 语言)"),
            ("paraformer", "中文生产级 ASR + 时间戳"),
            ("fun-asr-nano", "LLM-based ASR (31 语言)"),
        ]
    ]
    return JSONResponse({"object": "list", "data": models})


@router.get("/health")
async def health():
    registry = ModelRegistry.get_instance()
    return {
        "status": "ok",
        "device": registry.device,
        "models_loaded": registry.loaded_models(),
    }
