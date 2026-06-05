"""异步任务 API — 提交/查询/删除长文件转写任务"""

import os
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.core.task_manager import TaskManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["异步任务"])

# 全局任务管理器实例（在 app.py 中初始化并注入）
_task_manager: TaskManager | None = None


def set_task_manager(tm: TaskManager):
    global _task_manager
    _task_manager = tm


def get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise RuntimeError("TaskManager 未初始化")
    return _task_manager


@router.post("/submit")
async def submit_task(
    file: UploadFile | None = File(default=None, description="音频文件"),
    url: str | None = Form(default=None, description="远程音频 URL"),
    model: str = Form(default="sensevoice"),
    speaker_diarization: bool = Form(default=False),
    emotion: bool = Form(default=False),
    events: bool = Form(default=False),
    punctuation: bool = Form(default=True),
    language: str = Form(default="auto"),
    hotwords: str | None = Form(default=None),
):
    """提交异步转写任务（文件上传或 URL）"""
    tm = get_task_manager()

    if not file and not url:
        raise HTTPException(status_code=400, detail="必须提供 file 或 url 参数")

    kwargs = dict(
        model=model,
        speaker_diarization=speaker_diarization,
        emotion=emotion,
        events=events,
        punctuation=punctuation,
        language=language,
        hotwords=hotwords or "",
    )

    if url:
        task = await tm.submit_url(url=url, **kwargs)
    else:
        # 保存上传文件
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        content = await file.read()
        path = await _save_upload(content, suffix)
        task = await tm.submit_file(file_path=path, **kwargs)

    return JSONResponse(task.to_dict(), status_code=202)


@router.get("")
async def list_tasks():
    """列出所有任务"""
    tm = get_task_manager()
    tasks = [t.to_dict() for t in tm.list_tasks()]
    return JSONResponse({"tasks": tasks, "total": len(tasks)})


@router.get("/{task_id}")
async def get_task(task_id: str):
    """查询任务状态和结果"""
    tm = get_task_manager()
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse(task.to_dict())


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    tm = get_task_manager()
    if not tm.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse({"deleted": True, "task_id": task_id})


async def _save_upload(content: bytes, suffix: str) -> str:
    from server.core.audio import save_temp_upload
    return await save_temp_upload(content, suffix)
