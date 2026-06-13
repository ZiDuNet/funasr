"""标准异步转写任务 API。"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.schemas import build_config
from server.core.task_manager import TaskManager
from server.models.config import MODEL_NAME

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/transcription-jobs", tags=["异步转写任务"])

_task_manager: TaskManager | None = None


def set_task_manager(tm: TaskManager):
    global _task_manager
    _task_manager = tm


def get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise RuntimeError("TaskManager 未初始化")
    return _task_manager


@router.post("")
async def create_transcription_job(
    file: UploadFile | None = File(default=None, description="音频文件（与 url 二选一）"),
    url: str | None = Form(default=None, description="远程音频 URL（与 file 二选一）"),
    config: str | None = Form(default=None, description="统一 JSON 配置"),
    diarization: bool | None = Form(default=None, description="启用说话人分离"),
    speaker_match: bool | None = Form(default=None, description="启用注册声纹匹配"),
    speaker_group: str | None = Form(default=None, description="声纹组 ID"),
    emotion: bool | None = Form(default=None, description="返回情感标签"),
    events: bool | None = Form(default=None, description="返回事件标签"),
    punctuation: bool | None = Form(default=None, description="标点恢复"),
    language: str | None = Form(default=None, description="语言提示"),
    hotwords: str | None = Form(default=None, description="热词 JSON"),
):
    """提交异步转写任务，适合长音频或远程 URL。"""
    tm = get_task_manager()
    if not file and not url:
        raise HTTPException(status_code=400, detail="必须提供 file 或 url 参数")

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
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    kwargs = dict(
        model=MODEL_NAME,
        speaker_diarization=cfg.features.diarization,
        speaker_group=cfg.features.speaker_group,
        emotion=cfg.features.emotion,
        events=cfg.features.events,
        punctuation=cfg.features.punctuation,
        language=cfg.language,
        hotwords=cfg.hotwords,
    )

    if url:
        task = await tm.submit_url(url=url, **kwargs)
    else:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        content = await file.read()
        from server.core.audio import save_temp_upload
        path = await save_temp_upload(content, suffix)
        task = await tm.submit_file(file_path=path, **kwargs)

    return JSONResponse(task.to_dict(), status_code=202)


@router.get("")
async def list_transcription_jobs():
    """列出所有异步任务。"""
    tm = get_task_manager()
    tasks = [t.to_dict() for t in tm.list_tasks()]
    return JSONResponse({"jobs": tasks, "total": len(tasks)})


@router.get("/{task_id}")
async def get_transcription_job(task_id: str):
    """查询单个异步任务状态和结果。"""
    tm = get_task_manager()
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse(task.to_dict())


@router.delete("/{task_id}")
async def delete_transcription_job(task_id: str):
    """删除异步任务及其关联音频文件。"""
    tm = get_task_manager()
    if not tm.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse({"deleted": True, "task_id": task_id})
