"""声纹注册 API — 多租户隔离"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.models.registry import ModelRegistry
from server.core.speaker_db import (
    generate_group_id,
    register_speaker,
    remove_speaker,
    list_speakers,
    list_groups,
    extract_embedding,
)
from server.core.audio import save_temp_upload, convert_to_pcm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/speakers", tags=["声纹管理"])


@router.post("/register")
async def register(
    audio: UploadFile = File(..., description="说话人的参考音频（建议 5-30 秒，单人说话，背景安静）"),
    name: str = Form(..., description="说话人名字（如'张三'）"),
    speaker_group: str | None = Form(default=None, description="已有 group_id（不传则自动创建新 group）"),
):
    """注册说话人声纹

    不传 speaker_group 自动创建新 group，传入则加入已有 group。
    返回的 group_id 用于后续转写时传入 speaker_group 参数实现声纹匹配。
    转写时匹配到的 segment 会自动添加 speaker 字段替换 speaker_id。
    """
    registry = ModelRegistry.get_instance()
    sv_model = registry.get_aux("sv")

    # 保存音频
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    content = await audio.read()
    tmp_path = await save_temp_upload(content, suffix)

    try:
        # 转 PCM
        pcm_bytes = await convert_to_pcm(tmp_path)

        # 提取声纹
        embedding = extract_embedding(sv_model, pcm_bytes)
        if embedding is None:
            raise HTTPException(status_code=400, detail="提取声纹失败，请检查音频质量")

        # 注册
        group_id = speaker_group or generate_group_id()
        register_speaker(group_id, name, embedding)

        return JSONResponse({
            "group_id": group_id,
            "name": name,
            "status": "registered",
            "message": f"说话人 '{name}' 已注册到 group '{group_id}'",
        }, status_code=201)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("")
async def list_all_groups():
    """列出所有声纹 group"""
    groups = list_groups()
    return JSONResponse({"groups": groups, "total": len(groups)})


@router.get("/{group_id}")
async def get_group(group_id: str):
    """查看某个 group 的说话人列表"""
    speakers = list_speakers(group_id)
    return JSONResponse({"group_id": group_id, "speakers": speakers, "count": len(speakers)})


@router.delete("/{group_id}/{name}")
async def delete_speaker(group_id: str, name: str):
    """删除某个 group 中的说话人"""
    if not remove_speaker(group_id, name):
        raise HTTPException(status_code=404, detail=f"说话人 '{name}' 不存在于 group '{group_id}'")
    return JSONResponse({"deleted": True, "group_id": group_id, "name": name})
