"""标准转写 API 与 OpenAI 兼容接口。"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.formatters import response_for_format
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.schemas import build_config
from server.core.transcription import transcribe_pcm
from server.models.capabilities import capabilities_for_model, list_model_capabilities
from server.models.config import MODEL_NAME

logger = logging.getLogger(__name__)
router = APIRouter(tags=["转写"])


async def _transcribe_upload(file: UploadFile, config, *, source: str = "upload") -> dict:
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    content = await file.read()
    tmp_path = await save_temp_upload(content, suffix)
    try:
        pcm_bytes = await convert_to_pcm(tmp_path)
        return await transcribe_pcm(pcm_bytes, config, source=source)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/api/v1/transcriptions")
async def transcribe_v1(
    file: UploadFile = File(..., description="音频文件（wav/mp3/mp4/flac/m4a/ogg/webm）"),
    config: Optional[str] = Form(default=None, description="统一 JSON 配置"),
    language: Optional[str] = Form(default=None),
    diarization: Optional[bool] = Form(default=None),
    speaker_match: Optional[bool] = Form(default=None),
    speaker_group: Optional[str] = Form(default=None),
    emotion: Optional[bool] = Form(default=None),
    events: Optional[bool] = Form(default=None),
    punctuation: Optional[bool] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    words: Optional[bool] = Form(default=None),
    raw: Optional[bool] = Form(default=None),
    fallback: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default=None),
):
    """标准同步转写接口。

    响应必定包含 text、paragraph_count、sentence_count、paragraphs、sentences。
    可选能力会在基准结果上追加字段。
    """
    try:
        cfg = build_config(
            config_json=config,
            language=language,
            diarization=diarization,
            speaker_match=speaker_match,
            speaker_group=speaker_group,
            emotion=emotion,
            events=events,
            punctuation=punctuation,
            hotwords=hotwords,
            words=words,
            raw=raw,
            fallback=fallback,
            response_format=response_format,
        )
        result = await _transcribe_upload(file, cfg)
        return response_for_format(result, response_format=cfg.response_format)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("转写错误")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/transcriptions")
async def transcribe_openai(
    file: UploadFile = File(..., description="音频文件（wav/mp3/mp4/flac/m4a/ogg/webm）"),
    model: Optional[str] = Form(default=None, description="OpenAI 兼容字段；实际模型由 .env MODEL 决定"),
    language: Optional[str] = Form(default=None, description="语言提示（auto/zh/en/ja/ko/yue）"),
    prompt: Optional[str] = Form(default=None, description="OpenAI 兼容字段，会映射为 hotwords"),
    response_format: Optional[str] = Form(default="json", description="json/verbose_json/text/srt/vtt"),
    timestamp_granularities: Optional[str] = Form(default=None),
    diarization: Optional[bool] = Form(default=None),
    speaker_match: Optional[bool] = Form(default=None),
    speaker_group: Optional[str] = Form(default=None),
    emotion: Optional[bool] = Form(default=None),
    events: Optional[bool] = Form(default=None),
    punctuation: Optional[bool] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    config: Optional[str] = Form(default=None),
    fallback: Optional[str] = Form(default=None),
):
    """OpenAI 兼容转写接口。

    response_format=json 只返回 {"text": "..."}；
    response_format=verbose_json 返回 OpenAI segments 和标准 paragraphs/sentences。
    """
    try:
        cfg = build_config(
            config_json=config,
            language=language,
            diarization=diarization,
            speaker_match=speaker_match,
            speaker_group=speaker_group,
            emotion=emotion,
            events=events,
            punctuation=punctuation,
            hotwords=hotwords or prompt,
            fallback=fallback,
            response_format=response_format,
        )
        result = await _transcribe_upload(file, cfg)
        return response_for_format(
            result,
            response_format=cfg.response_format,
            openai=True,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("转写错误")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    """列出当前部署模型（OpenAI 兼容格式）"""
    return JSONResponse({"object": "list", "data": [
        {"id": "funasr", "object": "model", "created": 1700000000,
         "owned_by": "funasr", "ready": True,
         "name": MODEL_NAME,
         "capabilities": capabilities_for_model()}
    ]})


@router.get("/api/v1/capabilities")
async def capabilities():
    """查询当前部署模型能力。"""
    return JSONResponse(capabilities_for_model())


@router.get("/api/v1/models")
async def canonical_models():
    """查询内置模型能力矩阵。"""
    return JSONResponse({
        "current": capabilities_for_model(),
        "models": list_model_capabilities(),
    })


@router.get("/health")
async def health():
    """健康检查 + 模型加载状态"""
    from server.models.registry import ModelRegistry
    registry = ModelRegistry.get_instance()
    return {
        "status": "ok",
        "device": registry.device,
        "model": MODEL_NAME,
        "models_loaded": registry.loaded_models(),
        "capabilities": capabilities_for_model(),
    }
