"""HTTP REST — 简单同步转写端点"""

import os
import time
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.postprocess import clean_text
from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recognition")
async def recognition(
    audio: UploadFile = File(..., description="音频文件"),
    model: str = Form(default="sensevoice"),
    language: str = Form(default="auto"),
    speaker_diarization: bool = Form(default=False),
    punctuation: bool = Form(default=True),
):
    """简单文件上传转写（同步）"""
    registry = ModelRegistry.get_instance()

    # 保存文件
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    content = await audio.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        # 转 PCM
        pcm_bytes = await convert_to_pcm(tmp_path)

        # 推理
        asr_model = registry.get(model)
        gen_kwargs = {"batch_size_s": 300, "sentence_timestamp": True}
        if language and language != "auto":
            gen_kwargs["language"] = language

        result_list = await run_blocking(
            _generate_sync, asr_model, pcm_bytes,
            sem=registry.sem_asr_offline,
            **gen_kwargs,
        )

        if not result_list or not result_list[0].get("text"):
            return {"text": "", "sentences": [], "code": 0}

        raw = result_list[0]
        text = clean_text(raw.get("text", ""))

        sentences = []
        for seg in raw.get("sentence_info", []):
            s = {
                "text": clean_text(seg.get("text", "")),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            }
            if "spk" in seg:
                s["speaker"] = seg["spk"]
            sentences.append(s)

        return {"text": text, "sentences": sentences, "code": 0}

    except Exception as e:
        logger.error(f"转写错误: {e}")
        return {"msg": str(e), "code": 1}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
