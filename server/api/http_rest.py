"""HTTP REST — 简单同步转写"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.core.speaker_db import match_segments
from server.models.registry import ModelRegistry
from server.models.config import DEFAULT_BATCH_SIZE_S

logger = logging.getLogger(__name__)
router = APIRouter(tags=["HTTP REST"])


@router.post("/recognition")
async def recognition(
    audio: UploadFile = File(..., description="音频文件（wav/mp3/mp4/flac/m4a/ogg/webm）"),
    language: str = Form(default="auto", description="语言提示（auto/zh/en/ja/ko/yue）"),
    speaker_diarization: bool = Form(default=False, description="启用说话人分离，返回 sentences + speaker_id"),
    speaker_group: str | None = Form(default=None, description="声纹组 ID，匹配后将 speaker_id 替换为注册名"),
    emotion: bool = Form(default=False, description="返回情感标签（HAPPY/SAD/ANGRY 等）"),
    events: bool = Form(default=False, description="返回事件标签列表（BGM/Applause/Laughter 等）"),
    punctuation: bool = Form(default=True, description="标点恢复"),
    hotwords: str | None = Form(default=None, description='热词 JSON，如 {"达摩院":20}'),
):
    """HTTP REST 同步转写 — 简单文件上传，字段按请求参数条件返回"""
    """简单文件上传转写"""
    registry = ModelRegistry.get_instance()

    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    content = await audio.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        pcm_bytes = await convert_to_pcm(tmp_path)

        model = registry.get(with_spk=speaker_diarization)
        gen_kwargs = {
            "batch_size_s": DEFAULT_BATCH_SIZE_S,
            "language": language, "use_itn": True,
            "merge_vad": True, "merge_length_s": 15,
            "batch_size_threshold_s": 60, "sentence_timestamp": True,
        }

        result_list = await run_blocking(
            _generate_sync, model, pcm_bytes,
            sem=registry.sem_asr_offline, **gen_kwargs,
        )

        if not result_list or not result_list[0].get("text"):
            return {"text": "", "code": 0}

        raw = result_list[0]
        raw_text = raw.get("text", "")
        text = clean_text(raw_text)

        resp = {"text": text, "code": 0}

        # 情感（仅在请求时返回，始终有值或 null）
        if emotion:
            resp["emotion"] = extract_emotion(raw_text)

        # 事件（仅在请求时返回，始终有值或空数组）
        if events:
            resp["events"] = extract_events(raw_text)

        if speaker_group:
            resp["speaker_group"] = speaker_group

        # 分段信息（含时间戳；说话人信息仅在说话人分离时返回）
        if "sentence_info" in raw:
            sentences = []
            for seg in raw["sentence_info"]:
                s = {"text": clean_text(seg.get("text") or seg.get("sentence", "")),
                     "start": seg.get("start", 0), "end": seg.get("end", 0)}
                if speaker_diarization and "spk" in seg:
                    s["speaker_id"] = seg["spk"]
                sentences.append(s)

            # 声纹匹配（需同时启用 speaker_diarization + speaker_group）
            if speaker_diarization and speaker_group:
                match_segments(sentences, pcm_bytes, speaker_group,
                               registry.get_aux("sv"))

            resp["sentences"] = sentences

        return resp

    except Exception as e:
        logger.error(f"转写错误: {e}")
        return {"msg": str(e), "code": 1}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
