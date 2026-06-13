"""标准声纹组 API。"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.models.registry import ModelRegistry
from server.core.speaker_db import (
    create_group,
    remove_group,
    register_speaker,
    remove_speaker,
    list_speakers,
    list_groups,
    extract_embedding,
)
from server.core.audio import save_temp_upload, convert_to_pcm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/speaker-groups", tags=["声纹组"])


@router.post("")
async def create_speaker_group():
    """创建空声纹组。"""
    group_id = create_group()
    return JSONResponse({"group_id": group_id, "speaker_count": 0}, status_code=201)


@router.get("")
async def list_speaker_groups():
    """列出所有声纹组。"""
    groups = list_groups()
    return JSONResponse({"groups": groups, "total": len(groups)})


@router.get("/{group_id}/speakers")
async def get_speakers(group_id: str):
    """列出指定声纹组内的说话人。"""
    speakers = list_speakers(group_id)
    return JSONResponse({"group_id": group_id, "speakers": speakers, "count": len(speakers)})


@router.delete("/{group_id}")
async def delete_speaker_group(group_id: str):
    """删除整个声纹组及其中所有说话人。"""
    try:
        deleted = remove_group(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"声纹组不存在：{group_id}")
    return JSONResponse({"deleted": True, "group_id": group_id})


@router.post("/{group_id}/speakers")
async def create_speaker(
    group_id: str,
    audio: UploadFile = File(..., description="说话人的参考音频（建议 5-30 秒，单人说话，背景安静）"),
    name: str = Form(..., description="说话人名字"),
):
    """注册说话人声纹到指定声纹组。"""
    registry = ModelRegistry.get_instance()
    sv_model = registry.get_aux("sv")

    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    content = await audio.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        pcm_bytes = await convert_to_pcm(tmp_path)
        embedding = extract_embedding(sv_model, pcm_bytes)
        if embedding is None:
            raise HTTPException(status_code=400, detail="提取声纹失败，请检查音频质量")

        register_speaker(group_id, name, embedding)
        return JSONResponse({
            "group_id": group_id,
            "speaker_id": name,
            "name": name,
            "status": "registered",
        }, status_code=201)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.delete("/{group_id}/speakers/{name}")
async def delete_speaker(group_id: str, name: str):
    """删除指定声纹组中的说话人。"""
    if not remove_speaker(group_id, name):
        raise HTTPException(status_code=404, detail=f"说话人 '{name}' 不存在于 group '{group_id}'")
    return JSONResponse({"deleted": True, "group_id": group_id, "name": name})
