"""异步任务管理器 — 持久化存储 + 自动清理"""

import asyncio
import json
import os
import shutil
import uuid
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import httpx

from server.core.inference import run_blocking, _generate_sync
from server.core.audio import convert_to_pcm, save_temp_upload
from server.models.registry import ModelRegistry
from server.models.config import (
    MAX_TASKS, TASKS_DIR, AUDIO_DIR, DATA_TTL_DAYS,
)

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    DOWNLOADING = "downloading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionTask:
    task_id: str
    status: TaskStatus
    created_at: float
    file_path: str = ""
    audio_saved: str = ""      # 持久化的音频文件路径
    url: str = ""
    model: str = "sensevoice"
    speaker_diarization: bool = False
    speaker_group: str = ""
    emotion: bool = False
    events: bool = False
    punctuation: bool = True
    language: str = "auto"
    hotwords: str = ""
    result: Optional[dict] = None
    error: str = ""
    completed_at: float = 0.0
    duration_seconds: float = 0.0
    audio_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "model": self.model,
        }
        if self.url:
            d["url"] = self.url
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        if self.completed_at:
            d["completed_at"] = self.completed_at
            d["duration_seconds"] = self.duration_seconds
        if self.audio_duration_seconds:
            d["audio_duration_seconds"] = self.audio_duration_seconds
        return d

    def to_file(self):
        """写入 JSON 文件"""
        path = os.path.join(TASKS_DIR, f"{self.task_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._serialize(), f, ensure_ascii=False, indent=2)

    def _serialize(self) -> dict:
        d = self.to_dict()
        d["status"] = self.status.value
        return d

    @classmethod
    def from_file(cls, task_id: str) -> Optional["TranscriptionTask"]:
        path = os.path.join(TASKS_DIR, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            created_at=data["created_at"],
            file_path=data.get("file_path", ""),
            url=data.get("url", ""),
            model=data.get("model", "sensevoice"),
            speaker_diarization=data.get("speaker_diarization", False),
            speaker_group=data.get("speaker_group", ""),
            result=data.get("result"),
            error=data.get("error", ""),
            completed_at=data.get("completed_at", 0.0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


class TaskManager:
    """异步任务管理器（内存 + 磁盘双写）"""

    def __init__(self):
        self.tasks: dict[str, TranscriptionTask] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False

        # 确保数据目录存在
        os.makedirs(TASKS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)

        # 恢复未完成的任务
        self._recover()

    def _recover(self):
        """启动时从磁盘恢复未完成的旧任务"""
        if not os.path.isdir(TASKS_DIR):
            return
        for fname in os.listdir(TASKS_DIR):
            if not fname.endswith(".json"):
                continue
            task_id = fname.replace(".json", "")
            try:
                task = TranscriptionTask.from_file(task_id)
                if task and task.status in (TaskStatus.queued, TaskStatus.processing, TaskStatus.downloading):
                    task.status = TaskStatus.FAILED
                    task.error = "服务重启，任务中断"
                    task.completed_at = time.time()
                    task.to_file()
                    logger.warning(f"恢复任务 {task_id}: 已标记为失败（服务重启）")
            except Exception as e:
                logger.warning(f"恢复任务失败 {task_id}: {e}")

    async def start(self):
        self._running = True
        asyncio.create_task(self._worker())
        if DATA_TTL_DAYS > 0:
            asyncio.create_task(self._cleanup_loop())
            logger.info(f"自动清理已启用: {DATA_TTL_DAYS} 天后过期")

    async def stop(self):
        self._running = False

    async def submit_file(
        self,
        file_path: str,
        **kwargs,
    ) -> TranscriptionTask:
        """提交本地文件转写任务"""
        if len(self.tasks) >= MAX_TASKS:
            raise RuntimeError(f"任务数量已达上限 ({MAX_TASKS})")

        # 持久化音频文件
        audio_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex[:12]}.pcm")
        shutil.copy(file_path, audio_path)

        task = TranscriptionTask(
            task_id=uuid.uuid4().hex[:12],
            status=TaskStatus.QUEUED,
            created_at=time.time(),
            file_path=file_path,
            audio_saved=audio_path,
            **kwargs,
        )
        self.tasks[task.task_id] = task
        task.to_file()
        await self.queue.put(task)
        logger.info(f"任务已提交: {task.task_id}")
        return task

    async def submit_url(self, url: str, **kwargs) -> TranscriptionTask:
        """提交 URL 远程文件转写任务"""
        if len(self.tasks) >= MAX_TASKS:
            raise RuntimeError(f"任务数量已达上限 ({MAX_TASKS})")

        task = TranscriptionTask(
            task_id=uuid.uuid4().hex[:12],
            status=TaskStatus.DOWNLOADING,
            created_at=time.time(),
            url=url,
            **kwargs,
        )
        self.tasks[task.task_id] = task
        task.to_file()
        asyncio.create_task(self._download_and_queue(task))
        logger.info(f"URL 任务已提交: {task.task_id}")
        return task

    async def _download_and_queue(self, task: TranscriptionTask):
        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                resp = await client.get(task.url)
                resp.raise_for_status()
                content = resp.content

            suffix = os.path.splitext(task.url.split("?")[0])[-1] or ".wav"
            path = await save_temp_upload(content, suffix)

            # 持久化音频
            audio_path = os.path.join(AUDIO_DIR, f"{task.task_id}.pcm")
            shutil.copy(path, audio_path)
            task.audio_saved = audio_path
            task.file_path = path
            task.status = TaskStatus.QUEUED
            task.to_file()
            await self.queue.put(task)
            logger.info(f"URL 下载完成: {task.task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"下载失败: {e}"
            task.completed_at = time.time()
            task.to_file()
            logger.error(f"URL 下载失败: {task.task_id}: {e}")

    async def _worker(self):
        while self._running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            task.status = TaskStatus.PROCESSING
            task.to_file()
            t0 = time.time()

            try:
                # ffmpeg 转 PCM
                pcm_bytes = await convert_to_pcm(task.file_path)
                gen_kwargs = {"batch_size_s": 300}
                if task.language and task.language != "auto":
                    gen_kwargs["language"] = task.language

                registry = ModelRegistry.get_instance()
                model = registry.get(task.model)
                result_list = await run_blocking(
                    _generate_sync, model, pcm_bytes,
                    sem=registry.sem_asr_offline,
                    **gen_kwargs,
                )

                if result_list:
                    task.result = _format_result(result_list[0], task)
                else:
                    task.result = {"text": ""}

                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.duration_seconds = task.completed_at - t0
                task.to_file()
                logger.info(f"任务完成: {task.task_id} ({task.duration_seconds:.1f}s)")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                task.to_file()
                logger.error(f"任务失败: {task.task_id}: {e}")

    async def _cleanup_loop(self):
        """每 6 小时清理一次过期数据"""
        while self._running:
            await asyncio.sleep(21600)
            if DATA_TTL_DAYS <= 0:
                continue
            cutoff = time.time() - DATA_TTL_DAYS * 86400
            cleaned = 0
            for fname in os.listdir(TASKS_DIR):
                if not fname.endswith(".json"):
                    continue
                task_id = fname.replace(".json", "")
                try:
                    task = TranscriptionTask.from_file(task_id)
                    if task and task.created_at < cutoff:
                        self._delete_files(task_id, task)
                        cleaned += 1
                except Exception:
                    pass
            # 清理孤立音频文件
            for fname in os.listdir(AUDIO_DIR):
                fpath = os.path.join(AUDIO_DIR, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    cleaned += 1
            if cleaned:
                logger.info(f"自动清理: 删除 {cleaned} 个过期文件（{DATA_TTL_DAYS} 天前）")

    def _delete_files(self, task_id: str, task: TranscriptionTask):
        """删除任务相关文件"""
        # 删除 JSON
        tpath = os.path.join(TASKS_DIR, f"{task_id}.json")
        if os.path.exists(tpath):
            os.remove(tpath)
        # 删除音频
        if task.audio_saved and os.path.exists(task.audio_saved):
            os.remove(task.audio_saved)
        # 删除内存
        self.tasks.pop(task_id, None)

    def get_task(self, task_id: str) -> Optional[TranscriptionTask]:
        if task_id in self.tasks:
            return self.tasks[task_id]
        # 从磁盘恢复
        task = TranscriptionTask.from_file(task_id)
        if task:
            self.tasks[task_id] = task
        return task

    def list_tasks(self) -> list[TranscriptionTask]:
        # 优先内存
        tasks = list(self.tasks.values())
        # 补充磁盘中的任务
        if os.path.isdir(TASKS_DIR):
            for fname in os.listdir(TASKS_DIR):
                tid = fname.replace(".json", "")
                if tid not in self.tasks:
                    t = TranscriptionTask.from_file(tid)
                    if t:
                        self.tasks[tid] = t
                        tasks.append(t)
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def delete_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task:
            self._delete_files(task_id, task)
            return True
        return False


def _format_result(raw: dict, task: TranscriptionTask) -> dict:
    """格式化推理结果（包含声纹匹配）"""
    from server.core.postprocess import clean_text, extract_emotion, extract_events
    from server.core.speaker_db import match_speaker

    text = raw.get("text", "")
    emotion = extract_emotion(text) if task.emotion else None
    events = extract_events(text) if task.events else []
    clean = clean_text(text)

    result = {"text": clean}
    if emotion:
        result["emotion"] = emotion
    if events:
        result["events"] = events

    if "sentence_info" in raw:
        segments = []
        for seg in raw["sentence_info"]:
            s = {
                "text": clean_text(seg.get("text", "")),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            }
            if "spk" in seg:
                s["speaker_id"] = seg["spk"]
                # 如果有 group，尝试匹配声纹
                if task.speaker_group:
                    matched = match_speaker(task.speaker_group, seg.get("spk_embedding"))
                    if matched:
                        s["speaker"] = matched
            segments.append(s)
        result["segments"] = segments

    return result
