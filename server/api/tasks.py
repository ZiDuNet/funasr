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
    file: UploadFile | None = File(default=None, description="音频文件（与 url 二选一）"),
    url: str | None = Form(default=None, description="远程音频 URL（与 file 二选一）"),
    speaker_diarization: bool = Form(default=False, description="启用说话人分离，返回 segments + speaker_id"),
    speaker_group: str | None = Form(default=None, description="声纹组 ID，匹配后 speaker_id 替换为注册名"),
    emotion: bool = Form(default=False, description="返回情感标签（HAPPY/SAD/ANGRY 等）"),
    events: bool = Form(default=False, description="返回事件标签（BGM/Applause/Laughter 等）"),
    punctuation: bool = Form(default=True, description="标点恢复"),
    language: str = Form(default="auto", description="语言提示（auto/zh/en/ja/ko/yue）"),
    hotwords: str | None = Form(default=None, description='热词 JSON，如 {"达摩院":20}'),
):
    """提交异步转写任务

    支持文件上传或 URL 远程文件，提交后返回 task_id，轮询 GET /api/tasks/{task_id} 获取结果。
    字段按请求参数条件返回：传了什么参数，result 中就有什么字段。
    """
    tm = get_task_manager()

    if not file and not url:
        raise HTTPException(status_code=400, detail="必须提供 file 或 url 参数")

    kwargs = dict(
        speaker_diarization=speaker_diarization,
        speaker_group=speaker_group or "",
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
    """列出所有任务（含已完成和进行中的）"""
    tm = get_task_manager()
    tasks = [t.to_dict() for t in tm.list_tasks()]
    return JSONResponse({"tasks": tasks, "total": len(tasks)})


@router.get("/{task_id}")
async def get_task(task_id: str):
    """查询单个任务状态和转写结果

    result 字段按提交时的参数动态包含：text、emotion、events、segments 等。
    """
    tm = get_task_manager()
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse(task.to_dict())


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务及其关联的音频文件"""
    tm = get_task_manager()
    if not tm.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return JSONResponse({"deleted": True, "task_id": task_id})


async def _save_upload(content: bytes, suffix: str) -> str:
    from server.core.audio import save_temp_upload
    return await save_temp_upload(content, suffix)
