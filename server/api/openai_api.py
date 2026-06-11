"""OpenAI 兼容 API — 同步转写，模型由配置文件决定"""

import os
import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.core.speaker_db import match_segments
from server.models.registry import ModelRegistry
from server.models.config import DEFAULT_BATCH_SIZE_S, MODEL_NAME

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OpenAI 兼容"])


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="音频文件（wav/mp3/mp4/flac/m4a/ogg/webm）"),
    language: Optional[str] = Form(default=None, description="语言提示（auto/zh/en/ja/ko/yue）"),
    speaker_diarization: bool = Form(default=False, description="启用说话人分离，返回 segments + speaker_id"),
    speaker_group: Optional[str] = Form(default=None, description="声纹组 ID，匹配后将 speaker_id 替换为注册名"),
    emotion: bool = Form(default=False, description="返回情感标签（HAPPY/SAD/ANGRY/FEARFUL/DISGUSTED/SURPRISED）"),
    events: bool = Form(default=False, description="返回事件标签列表（BGM/Applause/Laughter/Cry/Sneeze/Cough）"),
    punctuation: bool = Form(default=True, description="标点恢复"),
    hotwords: Optional[str] = Form(default=None, description='热词 JSON，如 {"达摩院":20}'),
):
    """OpenAI 兼容音频转写

    始终返回详细 JSON，字段按请求参数条件返回：
    - 传了 emotion=true 才返回 emotion 字段
    - 传了 events=true 才返回 events 字段
    - 传了 speaker_diarization=true 才返回 segments（含 speaker_id）
    - 传了 speaker_group 且启用 speaker_diarization 才做声纹匹配
    """
    registry = ModelRegistry.get_instance()

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    content = await file.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        pcm_bytes = await convert_to_pcm(tmp_path)

        model = registry.get(with_spk=speaker_diarization)
        gen_kwargs = {
            "batch_size_s": DEFAULT_BATCH_SIZE_S,
            "batch_size_threshold_s": 0,   # 禁用批量解码（SenseVoice 不支持）
            "language": language or "auto",
            "use_itn": True,
            "merge_vad": True,
            "merge_length_s": 15,
            "sentence_timestamp": True,     # 始终返回时间戳
        }

        t0 = time.time()
        result_list = await run_blocking(
            _generate_sync, model, pcm_bytes,
            sem=registry.sem_asr_offline, **gen_kwargs,
        )
        elapsed = time.time() - t0

        if not result_list:
            return JSONResponse({"text": ""})

        raw = result_list[0]
        raw_text = raw.get("text", "")
        text = clean_text(raw_text)

        # 构建详细响应
        resp = {
            "text": text,
            "language": language or "auto",
            "duration": round(elapsed, 3),
            "model": MODEL_NAME,
        }

        # 情感（仅在请求时返回，始终有值或 null）
        if emotion:
            resp["emotion"] = extract_emotion(raw_text)

        # 事件（仅在请求时返回，始终有值或空数组）
        if events:
            resp["events"] = extract_events(raw_text)

        # 声纹分组
        if speaker_group:
            resp["speaker_group"] = speaker_group

        # 分段信息（含时间戳；说话人信息仅在说话人分离时返回）
        if "sentence_info" in raw:
            segments = []
            for seg in raw["sentence_info"]:
                s = {
                    "start": seg.get("start", 0),       # 毫秒，匹配后再转秒
                    "end": seg.get("end", 0),
                    "text": clean_text(seg.get("text") or seg.get("sentence", "")),
                }
                if speaker_diarization and "spk" in seg:
                    s["speaker_id"] = seg["spk"]
                segments.append(s)

            # 声纹匹配（需同时启用 speaker_diarization + speaker_group）
            if speaker_diarization and speaker_group:
                match_segments(segments, pcm_bytes, speaker_group,
                               registry.get_aux("sv"))

            # 时间戳统一转秒（OpenAI 兼容格式）
            for s in segments:
                s["start"] = s["start"] / 1000.0
                s["end"] = s["end"] / 1000.0

            resp["segments"] = segments

        return JSONResponse(resp)

    except Exception as e:
        logger.error(f"转写错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/v1/models")
async def list_models():
    """列出可用模型（OpenAI 兼容格式）"""
    return JSONResponse({"object": "list", "data": [
        {"id": "funasr", "object": "model", "created": 1700000000,
         "owned_by": "funasr", "ready": True,
         "name": MODEL_NAME}
    ]})


@router.get("/health")
async def health():
    """健康检查 + 模型加载状态"""
    registry = ModelRegistry.get_instance()
    return {
        "status": "ok",
        "device": registry.device,
        "model": MODEL_NAME,
        "models_loaded": registry.loaded_models(),
    }
