"""HTTP REST — 简单同步转写端点"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recognition")
async def recognition(
    audio: UploadFile = File(..., description="音频文件"),
    model: str = Form(default="sensevoice"),
    language: str = Form(default="auto"),
    speaker_diarization: bool = Form(default=False),
    speaker_group: str | None = Form(default=None),
    emotion: bool = Form(default=False),
    events: bool = Form(default=False),
    punctuation: bool = Form(default=True),
    hotwords: str | None = Form(default=None),
):
    """简单文件上传转写（同步）"""
    registry = ModelRegistry.get_instance()

    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    content = await audio.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        pcm_bytes = await convert_to_pcm(tmp_path)

        model_name = "sensevoice_spk" if speaker_diarization else model
        asr_model = registry.get(model_name)

        gen_kwargs = {
            "batch_size_s": 300,
            "language": language,
            "use_itn": True,
            "sentence_timestamp": True,
        }
        if model == "sensevoice" or model_name.startswith("sensevoice"):
            gen_kwargs["merge_vad"] = True
            gen_kwargs["merge_length_s"] = 15
            gen_kwargs["batch_size_threshold_s"] = 60

        result_list = await run_blocking(
            _generate_sync, asr_model, pcm_bytes,
            sem=registry.sem_asr_offline,
            **gen_kwargs,
        )

        if not result_list or not result_list[0].get("text"):
            return {"text": "", "sentences": [], "code": 0}

        raw = result_list[0]
        raw_text = raw.get("text", "")
        text = clean_text(raw_text)

        emo = extract_emotion(raw_text) if emotion else None
        evt = extract_events(raw_text) if events else []

        sentences = []
        for seg in raw.get("sentence_info", []):
            s = {
                "text": clean_text(seg.get("text", "")),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            }
            if "spk" in seg:
                s["speaker_id"] = seg["spk"]
            sentences.append(s)

        resp = {"text": text, "sentences": sentences, "code": 0}
        if emo:
            resp["emotion"] = emo
        if evt:
            resp["events"] = evt
        if speaker_group:
            resp["speaker_group"] = speaker_group

        return resp

    except Exception as e:
        logger.error(f"转写错误: {e}")
        return {"msg": str(e), "code": 1}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
