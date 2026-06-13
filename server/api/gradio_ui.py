"""FunASR All-in-One 的 Gradio 中文 WebUI。"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any

import numpy as np
import gradio as gr
from scipy.signal import resample_poly

from server.core.audio import convert_to_pcm, save_temp_upload
from server.core.schemas import build_config
from server.core.transcription import transcribe_pcm
from server.core.task_manager import TaskManager
from server.core.speaker_db import (
    create_group,
    list_groups,
    list_speakers,
    register_speaker,
    remove_group,
    remove_speaker,
    extract_embedding,
)
from server.models.capabilities import capabilities_for_model, list_model_capabilities
from server.models.config import MODEL_NAME
from server.models.registry import ModelRegistry

_TASK_MANAGER: TaskManager | None = None
SUPPORTED_AUDIO_FILE_TYPES = [
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".wma",
    ".amr",
]

_REALTIME_WS_URL = (
    f"ws://{os.environ.get('FUNASR_HOST', '127.0.0.1')}:"
    f"{os.environ.get('FUNASR_PORT', '17767')}/api/v1/realtime/transcriptions"
)
_API_TOKEN = os.environ.get("API_TOKEN", "").strip()


def numpy_to_pcm16k(sample_rate: int, data: np.ndarray) -> bytes:
    """Gradio gr.Audio(type='numpy') 拿到 (sr, float32[]) → 16kHz 单声道 int16 PCM bytes。

    Gradio 给的是 float32（-1~1），采样率取决于设备（常见 48k/44.1k）。
    WS 服务期望 16kHz 16bit 单声道 PCM（每 ms 32 字节）。
    """
    if data is None or len(data) == 0:
        return b""
    data = np.asarray(data, dtype=np.float32).flatten()
    if sample_rate != 16000:
        # scipy 重采样（整数比插值，抗混叠）
        from math import gcd
        g = gcd(int(sample_rate), 16000)
        up = 16000 // g
        down = int(sample_rate) // g
        data = resample_poly(data, up, down)
    # float32 → int16
    clipped = np.clip(data, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    return int16.tobytes()


class RealtimeSession:
    """每个 Gradio 会话独立持有的 WS 代理，前端 mic → 本类 → 本地 WS 服务。

    Gradio 的 streaming handler 是 async 的，能直接 await websockets。
    _reader 协程在 handler 之间持续运行，把 WS 返回的中间/最终结果累积到 snapshot。
    """

    def __init__(self):
        self.ws = None
        self.reader_task: asyncio.Task | None = None
        self.partial = ""
        self.finals: list[dict] = []
        self.error = ""
        self.last_event: dict = {}
        self.last_render = ""
        self.last_event_json = "{}"
        self.sent_samples = 0
        self.pending_pcm = b""
        self.frame_bytes = 1920

    async def connect(self, *, mode: str, hotwords: str,
                      spk: bool, speaker_match: bool, speaker_group: str,
                      emo: bool, events: bool, punctuation: bool) -> None:
        import websockets
        url = _REALTIME_WS_URL
        if _API_TOKEN:
            url = f"{url}?token={_API_TOKEN}"
        self.ws = await websockets.connect(url, subprotocols=["binary"], max_size=None)
        cfg = {
            "type": "session.start",
            "mode": mode,
            "chunk_size": [0, 10, 5],
            "chunk_interval": 10,
            "wav_name": "gradio",
            "wav_format": "pcm",
            "audio_fs": 16000,
            "itn": True,
            "features": {
                "diarization": spk,
                "speaker_match": {
                    "enabled": speaker_match,
                    "group_id": speaker_group or "",
                },
                "emotion": emo,
                "events": events,
                "punctuation": punctuation,
            },
            "options": {
                "language": "auto",
            },
            "fallback": "auto",
        }
        if hotwords:
            cfg["options"]["hotwords"] = hotwords
        self.frame_bytes = int(16000 * 2 * (cfg["chunk_size"][1] * 60 / cfg["chunk_interval"]) / 1000)
        await self.ws.send(json.dumps(cfg))
        self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self.last_event = data
                mode = data.get("mode", "")
                if mode in ("2pass-online", "online"):
                    self.partial = data.get("text", "")
                elif mode in ("2pass-offline", "offline") and data.get("is_final"):
                    text = data.get("clean_text") or data.get("text", "")
                    if text:
                        speaker = None
                        sentences = data.get("sentences") or []
                        if sentences:
                            speaker = sentences[0].get("speaker")
                        self.finals.append({
                            "text": text,
                            "speaker": speaker,
                            "emotion": data.get("emotion"),
                            "events": data.get("events"),
                        })
                    self.partial = ""
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.error = str(exc)

    async def send_pcm(self, pcm_bytes: bytes) -> None:
        if self.ws and pcm_bytes:
            await self.ws.send(pcm_bytes)

    async def send_audio_array(self, sample_rate: int, data: np.ndarray) -> None:
        """Send only new audio, split into FunASR-style 60ms PCM frames."""
        if data is None:
            return
        data = np.asarray(data, dtype=np.float32).flatten()
        if len(data) <= self.sent_samples:
            return
        new_data = data[self.sent_samples:]
        self.sent_samples = len(data)
        self.pending_pcm += numpy_to_pcm16k(sample_rate, new_data)
        while len(self.pending_pcm) >= self.frame_bytes:
            frame = self.pending_pcm[:self.frame_bytes]
            self.pending_pcm = self.pending_pcm[self.frame_bytes:]
            await self.send_pcm(frame)

    async def stop(self) -> None:
        if not self.ws:
            return
        try:
            if self.pending_pcm:
                await self.send_pcm(self.pending_pcm)
                self.pending_pcm = b""
            await self.ws.send(json.dumps({"type": "audio.end"}))
            # 给服务端时间回最后一帧离线结果
            await asyncio.sleep(1.5)
        except Exception:
            pass
        if self.reader_task:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass
        try:
            await self.ws.close()
        except Exception:
            pass
        self.ws = None

    def snapshot(self) -> dict:
        return {
            "partial": self.partial,
            "finals": list(self.finals),
            "error": self.error,
            "last_event": self.last_event,
        }


def set_task_manager(task_manager: TaskManager | None) -> None:
    global _TASK_MANAGER
    _TASK_MANAGER = task_manager


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _file_path(file_obj) -> str:
    if file_obj is None:
        raise gr.Error("请先上传音频文件")
    if isinstance(file_obj, str):
        return file_obj
    path = getattr(file_obj, "name", None)
    if not path:
        raise gr.Error("无法读取上传文件路径")
    return path


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _format_timeline(result: dict) -> str:
    sentences = result.get("sentences") or []
    if not sentences:
        return "暂无句子时间轴"
    lines = []
    for s in sentences:
        speaker = s.get("speaker") or {}
        speaker_text = ""
        if speaker.get("name"):
            score = speaker.get("score")
            speaker_text = f" [{speaker['name']}" + (f" {score:.3f}" if isinstance(score, (int, float)) else "") + "]"
        elif speaker.get("id"):
            speaker_text = f" [{speaker['id']}]"
        emotion = f" <{s.get('emotion')}>" if s.get("emotion") else ""
        events = f" 事件:{','.join(s.get('events') or [])}" if s.get("events") else ""
        lines.append(
            f"{s.get('id', 0):03d} | {s.get('start', 0):.2f}-{s.get('end', 0):.2f}s"
            f"{speaker_text}{emotion}{events}\n{s.get('text', '')}"
        )
    return "\n\n".join(lines)


def _format_paragraphs(result: dict) -> str:
    paragraphs = result.get("paragraphs") or []
    if not paragraphs:
        return "暂无段落"
    return "\n\n".join(
        f"段落 {p.get('id', 0)} | {p.get('start', 0):.2f}-{p.get('end', 0):.2f}s\n{p.get('text', '')}"
        for p in paragraphs
    )


async def _transcribe_path(
    path: str,
    language: str,
    diarization: bool,
    speaker_match: bool,
    speaker_group: str,
    emotion: bool,
    events: bool,
    punctuation: bool,
    hotwords: str,
    fallback: str,
    raw: bool,
) -> dict:
    cfg = build_config(
        language=language or "auto",
        diarization=diarization,
        speaker_match=speaker_match,
        speaker_group=speaker_group or None,
        emotion=emotion,
        events=events,
        punctuation=punctuation,
        hotwords=hotwords or None,
        fallback=fallback or "error",
        raw=raw,
    )
    pcm = await convert_to_pcm(path)
    return await transcribe_pcm(pcm, cfg, source="gradio")


def transcribe_file(
    file_obj,
    language,
    diarization,
    speaker_match,
    speaker_group,
    emotion,
    events,
    punctuation,
    hotwords,
    fallback,
    raw,
):
    try:
        result = _run(_transcribe_path(
            _file_path(file_obj),
            language,
            diarization,
            speaker_match,
            speaker_group,
            emotion,
            events,
            punctuation,
            hotwords,
            fallback,
            raw,
        ))
        summary = (
            f"模型：{result.get('model', MODEL_NAME)}\n"
            f"音频时长：{result.get('duration', 0):.2f}s\n"
            f"处理耗时：{result.get('processing_time', 0):.2f}s\n"
            f"段落数：{result.get('paragraph_count', 0)}\n"
            f"句子数：{result.get('sentence_count', 0)}"
        )
        return (
            result.get("text", ""),
            _format_paragraphs(result),
            _format_timeline(result),
            summary,
            _json_dumps(result),
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def submit_job(
    file_obj,
    url,
    language,
    diarization,
    speaker_match,
    speaker_group,
    emotion,
    events,
    punctuation,
    hotwords,
):
    try:
        tm = _TASK_MANAGER
        if tm is None:
            raise gr.Error("任务管理器尚未初始化")
        cfg = build_config(
            language=language or "auto",
            diarization=diarization,
            speaker_match=speaker_match,
            speaker_group=speaker_group or None,
            emotion=emotion,
            events=events,
            punctuation=punctuation,
            hotwords=hotwords or None,
        )
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
            task = _run(tm.submit_url(url=url, **kwargs))
        else:
            src = _file_path(file_obj)
            suffix = os.path.splitext(src)[1] or ".wav"
            with open(src, "rb") as f:
                tmp = _run(save_temp_upload(f.read(), suffix))
            task = _run(tm.submit_file(file_path=tmp, **kwargs))
        return f"任务已提交：{task.task_id}", _json_dumps(task.to_dict())
    except Exception as exc:
        raise gr.Error(str(exc))


def list_jobs():
    tm = _TASK_MANAGER
    if tm is None:
        return [], _json_dumps({"error": "任务管理器尚未初始化"})
    jobs = [t.to_dict() for t in tm.list_tasks()]
    rows = []
    for t in jobs:
        rows.append([
            t.get("task_id"),
            t.get("status"),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.get("created_at", 0))),
            t.get("duration_seconds", ""),
        ])
    return rows, _json_dumps({"jobs": jobs, "total": len(jobs)})


def get_job(task_id: str):
    if not task_id:
        raise gr.Error("请输入 task_id")
    tm = _TASK_MANAGER
    if tm is None:
        raise gr.Error("任务管理器尚未初始化")
    task = tm.get_task(task_id)
    if not task:
        raise gr.Error(f"任务不存在：{task_id}")
    data = task.to_dict()
    result = data.get("result") or {}
    return result.get("text", ""), _format_timeline(result), _json_dumps(data)


def delete_job(task_id: str):
    if not task_id:
        raise gr.Error("请输入 task_id")
    tm = _TASK_MANAGER
    if tm is None:
        raise gr.Error("任务管理器尚未初始化")
    if not tm.delete_task(task_id):
        raise gr.Error(f"任务不存在：{task_id}")
    return f"已删除任务：{task_id}"


def _speaker_group_choices() -> list[str]:
    return [item["group_id"] for item in list_groups()]


def _group_rows() -> list[list[Any]]:
    rows = []
    for item in list_groups():
        speakers = item.get("speakers") or []
        rows.append([
            item.get("group_id", ""),
            item.get("speaker_count", len(speakers)),
            "、".join(speakers),
        ])
    return rows


def _speaker_rows(group_id: str | None) -> list[list[str]]:
    if not group_id:
        return []
    return [[name] for name in list_speakers(group_id)]


def _speaker_names(group_id: str | None) -> list[str]:
    if not group_id:
        return []
    return list_speakers(group_id)


def _group_status_text(group_id: str | None) -> str:
    if not group_id:
        return "尚未选择声纹组。请先创建声纹组，或从下拉框选择已有声纹组。"
    speakers = list_speakers(group_id)
    names = "、".join(speakers) if speakers else "暂无"
    return f"当前声纹组：{group_id}\n说话人数：{len(speakers)}\n组内说话人：{names}"


def _resolve_group_value(preferred: str | None = None) -> str | None:
    choices = _speaker_group_choices()
    if preferred and preferred in choices:
        return preferred
    return choices[0] if choices else None


def refresh_all_speaker_ui(current_group: str | None = None):
    value = _resolve_group_value(current_group)
    group_choices = _speaker_group_choices()
    speaker_names = _speaker_names(value)
    speaker_value = speaker_names[0] if speaker_names else None
    return (
        gr.update(choices=group_choices, value=value),
        gr.update(choices=group_choices, value=value),
        gr.update(choices=group_choices, value=value),
        _group_rows(),
        _group_status_text(value),
        gr.update(choices=speaker_names, value=speaker_value),
        _speaker_rows(value),
    )


def select_speaker_group(group_id: str | None):
    value = _resolve_group_value(group_id)
    group_choices = _speaker_group_choices()
    speaker_names = _speaker_names(value)
    speaker_value = speaker_names[0] if speaker_names else None
    return (
        _group_status_text(value),
        gr.update(choices=speaker_names, value=speaker_value),
        _speaker_rows(value),
        gr.update(choices=group_choices, value=value),
        gr.update(choices=group_choices, value=value),
    )


def create_speaker_group_ui():
    group_id = create_group()
    (
        file_group,
        job_group,
        manage_group,
        group_rows,
        group_status,
        speaker_dropdown,
        speaker_rows,
    ) = refresh_all_speaker_ui(group_id)
    return (
        file_group,
        job_group,
        manage_group,
        group_rows,
        group_status,
        speaker_dropdown,
        speaker_rows,
        f"已创建声纹组：{group_id}",
    )


def delete_speaker_group_ui(group_id: str | None):
    if not group_id:
        raise gr.Error("请先选择要删除的声纹组")
    if not remove_group(group_id):
        raise gr.Error(f"声纹组不存在：{group_id}")
    (
        file_group,
        job_group,
        manage_group,
        group_rows,
        group_status,
        speaker_dropdown,
        speaker_rows,
    ) = refresh_all_speaker_ui()
    return (
        file_group,
        job_group,
        manage_group,
        group_rows,
        group_status,
        speaker_dropdown,
        speaker_rows,
        f"已删除声纹组：{group_id}",
    )


def register_ui_speaker(group_id: str, name: str, file_obj):
    if not group_id:
        raise gr.Error("请先创建或选择声纹组")
    if not name:
        raise gr.Error("请输入说话人名称")
    try:
        registry = ModelRegistry.get_instance()
        pcm = _run(convert_to_pcm(_file_path(file_obj)))
        embedding = extract_embedding(registry.get_aux("sv"), pcm)
        if embedding is None:
            raise gr.Error("提取声纹失败，请检查音频质量")
        register_speaker(group_id, name, embedding)
        group_choices = _speaker_group_choices()
        speakers = _speaker_names(group_id)
        return (
            f"已注册说话人：{name}",
            gr.update(choices=speakers, value=name),
            _speaker_rows(group_id),
            _group_rows(),
            _group_status_text(group_id),
            gr.update(choices=group_choices, value=group_id),
            gr.update(choices=group_choices, value=group_id),
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def list_ui_speakers(group_id: str | None):
    if not group_id:
        raise gr.Error("请先选择声纹组")
    speakers = _speaker_names(group_id)
    return (
        _group_status_text(group_id),
        gr.update(choices=speakers, value=speakers[0] if speakers else None),
        _speaker_rows(group_id),
    )


def delete_ui_speaker(group_id: str | None, name: str | None):
    if not group_id or not name:
        raise gr.Error("请先选择声纹组和说话人")
    if not remove_speaker(group_id, name):
        raise gr.Error(f"说话人不存在：{name}")
    speakers = _speaker_names(group_id)
    group_choices = _speaker_group_choices()
    return (
        f"已删除说话人：{name}",
        gr.update(choices=speakers, value=speakers[0] if speakers else None),
        _speaker_rows(group_id),
        _group_rows(),
        _group_status_text(group_id),
        gr.update(choices=group_choices, value=group_id),
        gr.update(choices=group_choices, value=group_id),
    )


def _format_realtime(snapshot: dict) -> str:
    """把 RealtimeSession.snapshot() 渲染成可读文本。"""
    if not snapshot:
        return "等待录音..."
    parts = []
    finals = snapshot.get("finals") or []
    for idx, item in enumerate(finals, start=1):
        speaker = item.get("speaker") or {}
        spk_label = ""
        if isinstance(speaker, dict):
            name = speaker.get("name") or speaker.get("id")
            if name:
                spk_label = f"[{name}] "
        elif speaker:
            spk_label = f"[{speaker}] "
        emo = f" <{item.get('emotion')}>" if item.get("emotion") else ""
        events = f" 事件:{','.join(item.get('events') or [])}" if item.get("events") else ""
        parts.append(f"{idx}. {spk_label}{item.get('text', '')}{emo}{events}")
    partial = snapshot.get("partial") or ""
    partial_line = f"\n\n… {partial}" if partial else ""
    err = snapshot.get("error") or ""
    err_line = f"\n\n⚠️ {err}" if err else ""
    return "\n".join(parts) + partial_line + err_line if parts else (partial_line.lstrip() or err_line or "等待录音...")


def _realtime_outputs(session: RealtimeSession, status: str, *, force: bool = False):
    snapshot = session.snapshot()
    rendered = _format_realtime(snapshot)
    event_json = _json_dumps(snapshot.get("last_event") or {})

    output = rendered if force or rendered != session.last_render else gr.skip()
    raw = event_json if force or event_json != session.last_event_json else gr.skip()
    session.last_render = rendered
    session.last_event_json = event_json
    return output, session, status, raw


async def realtime_stream(audio, session: RealtimeSession, mode, hotwords,
                          spk, speaker_match, speaker_group,
                          emo, events, punctuation):
    """gr.Audio(streaming=True) 的 handler。

    Gradio 在录音开始后按 stream_every 间隔调用：
      - 第一次进入时 session.ws 还没建立 → 建立连接并发送配置帧
      - 每次进入拿到增量音频 → 转 PCM → 发 WS → 返回当前累积结果
      - 用户停止录音 → audio 为 None → 关闭 WS，返回最终结果
    """
    if audio is None:
        if session.ws:
            await session.stop()
        return _realtime_outputs(session, "已停止", force=True)

    sample_rate, data = audio
    try:
        if session.ws is None:
            await session.connect(
                mode=mode or "2pass",
                hotwords=hotwords or "",
                spk=bool(spk),
                speaker_match=bool(speaker_match),
                speaker_group=speaker_group or "",
                emo=bool(emo),
                events=bool(events),
                punctuation=bool(punctuation),
            )
        await session.send_audio_array(int(sample_rate or 16000), data)
    except Exception as exc:
        session.error = str(exc)

    status = "录音中…" if session.ws and not session.error else f"错误：{session.error}"
    return _realtime_outputs(session, status)


def realtime_reset():
    """重置按钮：返回新 session + 清空显示。"""
    return "", RealtimeSession(), "已清空，可重新录音", "{}"


def service_status():
    registry = ModelRegistry.get_instance()
    data = {
        "status": "ok",
        "model": MODEL_NAME,
        "device": registry.device,
        "models_loaded": registry.loaded_models(),
        "capabilities": capabilities_for_model(),
    }
    text = (
        f"状态：运行中\n"
        f"模型：{data['model']}\n"
        f"设备：{data['device']}\n"
        f"已加载模型：{', '.join(data['models_loaded']) or '尚未加载'}"
    )
    return text, _json_dumps(data)


def model_matrix():
    return _json_dumps({"current": capabilities_for_model(), "models": list_model_capabilities()})


def api_reference_text() -> str:
    return """
## 标准接口

`POST /api/v1/transcriptions`

必定返回 `text`、`paragraph_count`、`sentence_count`、`paragraphs[]`、`sentences[]`，句子时间为秒。

常用参数：

| 参数 | 说明 |
|---|---|
| `language` | 语言提示，默认 `auto` |
| `diarization` | 是否启用说话人分离 |
| `speaker_match` | 是否启用声纹组匹配 |
| `speaker_group` | 声纹组 ID |
| `emotion` | 情感识别 |
| `events` | 事件检测，SenseVoice 支持 |
| `punctuation` | 标点恢复 |
| `hotwords` | 热词 JSON |

## 异步任务

`POST /api/v1/transcription-jobs`

适合长音频和远程 URL。查询：`GET /api/v1/transcription-jobs/{task_id}`。

## 声纹管理

`POST /api/v1/speaker-groups` 创建声纹组。

`POST /api/v1/speaker-groups/{group_id}/speakers` 注册说话人。

## 实时 WebSocket

Swagger `/docs` 不展示 WebSocket 路由；实时接口以本节为准。

连接地址：

```text
ws://host:17767/api/v1/realtime/transcriptions
```

启用 `API_TOKEN` 时：

```text
ws://host:17767/api/v1/realtime/transcriptions?token=your-token
```

连接建立后先发送 `session.start` 配置事件。参数结构与 HTTP API 一致：

```json
{
  "type": "session.start",
  "mode": "2pass",
  "audio_fs": 16000,
  "wav_format": "pcm",
  "chunk_size": [5, 10, 5],
  "chunk_interval": 10,
  "itn": true,
  "features": {
    "diarization": true,
    "speaker_match": {
      "enabled": true,
      "group_id": "grp_xxx"
    },
    "emotion": true,
    "events": true,
    "punctuation": true,
    "raw": false
  },
  "options": {
    "language": "auto",
    "hotwords": {"FunASR": 20}
  },
  "fallback": "auto"
}
```

随后持续发送二进制音频帧：`16 kHz / mono / signed int16 little-endian PCM`。
停止录音时发送：

```json
{"type": "audio.end"}
```

服务端事件：

- `session.started`：配置已生效。
- `transcript.delta`：实时中间文本，只在 `online` / `2pass` 中返回。
- `transcript.segment`：VAD 断句后的最终片段，只在 `offline` / `2pass` 中返回。
- `error`：配置或推理错误。

三种模式：

| 模式 | 返回逻辑 | 增强字段 |
|---|---|---|
| `online` | 只返回 `transcript.delta`，延迟最低 | 不返回说话人、声纹、情感、事件 |
| `offline` | VAD 断句后返回 `transcript.segment` | 最终段可返回说话人、声纹、情感、事件、标点 |
| `2pass` | 先返回 `transcript.delta`，断句后返回 `transcript.segment` 修正 | 最终段可返回全部增强字段 |

说话人一致性只在单个 WebSocket 连接会话内维护。开启 `diarization` 后，
服务端会用会话级 `SpeakerTracker` 尽量保持第一句和第一百句的匿名
`speaker_0` 指向同一人；开启 `speaker_match` 并指定 `group_id` 后，
最终段会继续匹配注册声纹并返回姓名和分数。

## OpenAI 兼容

`POST /v1/audio/transcriptions`

默认 `response_format=json` 只返回 `{ "text": "..." }`。需要完整时间轴请用标准接口，或使用 `response_format=verbose_json`。

## 完整文档

查看仓库 `API.md`，或打开 `/docs` 查看 Swagger。
"""


def build_gradio_app(task_manager: TaskManager | None = None):
    set_task_manager(task_manager)
    initial_group = _resolve_group_value()
    initial_group_choices = _speaker_group_choices()
    initial_speakers = _speaker_names(initial_group)
    initial_speaker = initial_speakers[0] if initial_speakers else None

    with gr.Blocks(title="FunASR All-in-One", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
# FunASR All-in-One

当前模型：**{MODEL_NAME}**

标准 API 会统一返回全文、段落、句子时间轴；额外能力由参数控制。
"""
        )

        with gr.Tab("文件转写"):
            with gr.Row():
                audio_file = gr.File(
                    label="上传音频文件",
                    file_types=SUPPORTED_AUDIO_FILE_TYPES,
                    type="filepath",
                )
                with gr.Column():
                    language = gr.Dropdown(["auto", "zh", "en", "ja", "ko", "yue"], value="auto", label="语言")
                    diarization = gr.Checkbox(label="说话人分离", value=False)
                    speaker_match = gr.Checkbox(label="声纹匹配", value=False)
                    speaker_group = gr.Dropdown(
                        choices=initial_group_choices,
                        value=initial_group,
                        label="声纹组",
                        interactive=True,
                    )
                    emotion = gr.Checkbox(label="情感识别", value=False)
                    events = gr.Checkbox(label="事件检测", value=False)
                    punctuation = gr.Checkbox(label="标点恢复", value=True)
                    hotwords = gr.Textbox(label="热词 JSON", placeholder='{"FunASR": 20}')
                    fallback = gr.Radio(["error", "auto"], value="error", label="不支持能力处理")
                    raw = gr.Checkbox(label="返回原始 FunASR 输出", value=False)
                    run_btn = gr.Button("开始转写", variant="primary")

            text_out = gr.Textbox(label="完整文本", lines=6)
            paragraphs_out = gr.Textbox(label="段落", lines=8)
            timeline_out = gr.Textbox(label="句子时间轴", lines=12)
            summary_out = gr.Textbox(label="摘要信息", lines=5)
            json_out = gr.Code(label="完整 JSON", language="json")
            run_btn.click(
                transcribe_file,
                [
                    audio_file,
                    language,
                    diarization,
                    speaker_match,
                    speaker_group,
                    emotion,
                    events,
                    punctuation,
                    hotwords,
                    fallback,
                    raw,
                ],
                [text_out, paragraphs_out, timeline_out, summary_out, json_out],
            )

        with gr.Tab("异步任务"):
            with gr.Row():
                job_file = gr.File(
                    label="上传长音频",
                    file_types=SUPPORTED_AUDIO_FILE_TYPES,
                    type="filepath",
                )
                job_url = gr.Textbox(label="远程音频 URL")
            with gr.Row():
                job_language = gr.Dropdown(["auto", "zh", "en", "ja", "ko", "yue"], value="auto", label="语言")
                job_diarization = gr.Checkbox(label="说话人分离", value=False)
                job_speaker_match = gr.Checkbox(label="声纹匹配", value=False)
                job_speaker_group = gr.Dropdown(
                    choices=initial_group_choices,
                    value=initial_group,
                    label="声纹组",
                    interactive=True,
                )
            with gr.Row():
                job_emotion = gr.Checkbox(label="情感识别", value=False)
                job_events = gr.Checkbox(label="事件检测", value=False)
                job_punctuation = gr.Checkbox(label="标点恢复", value=True)
                job_hotwords = gr.Textbox(label="热词 JSON")
            submit_btn = gr.Button("提交任务", variant="primary")
            job_msg = gr.Textbox(label="提交结果")
            job_json = gr.Code(label="任务 JSON", language="json")
            submit_btn.click(
                submit_job,
                [
                    job_file,
                    job_url,
                    job_language,
                    job_diarization,
                    job_speaker_match,
                    job_speaker_group,
                    job_emotion,
                    job_events,
                    job_punctuation,
                    job_hotwords,
                ],
                [job_msg, job_json],
            )

            refresh_jobs = gr.Button("刷新任务列表")
            jobs_table = gr.Dataframe(headers=["task_id", "状态", "创建时间", "处理耗时"], label="任务列表")
            jobs_raw = gr.Code(label="任务列表 JSON", language="json")
            refresh_jobs.click(list_jobs, outputs=[jobs_table, jobs_raw])

            task_id = gr.Textbox(label="task_id")
            with gr.Row():
                get_btn = gr.Button("查询任务")
                del_btn = gr.Button("删除任务")
            task_text = gr.Textbox(label="任务文本", lines=5)
            task_timeline = gr.Textbox(label="任务句子时间轴", lines=8)
            task_json = gr.Code(label="任务详情 JSON", language="json")
            del_msg = gr.Textbox(label="删除结果")
            get_btn.click(get_job, task_id, [task_text, task_timeline, task_json])
            del_btn.click(delete_job, task_id, del_msg)

        with gr.Tab("声纹管理"):
            with gr.Row():
                manage_group = gr.Dropdown(
                    choices=initial_group_choices,
                    value=initial_group,
                    label="1. 选择声纹组",
                    interactive=True,
                )
                create_group_btn = gr.Button("新建声纹组", variant="primary")
                refresh_groups_btn = gr.Button("刷新")
                delete_group_btn = gr.Button("删除当前声纹组", variant="stop")

            group_status = gr.Textbox(
                label="当前声纹组",
                value=_group_status_text(initial_group),
                lines=3,
                interactive=False,
            )
            groups_table = gr.Dataframe(
                headers=["声纹组 ID", "说话人数", "说话人"],
                value=_group_rows(),
                label="声纹组列表",
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                with gr.Column():
                    speaker_name = gr.Textbox(label="2. 新增说话人名称")
                    speaker_audio = gr.File(
                        label="说话人参考音频",
                        file_types=SUPPORTED_AUDIO_FILE_TYPES,
                        type="filepath",
                    )
                    reg_speaker_btn = gr.Button("注册到当前声纹组", variant="primary")
                with gr.Column():
                    speaker_select = gr.Dropdown(
                        choices=initial_speakers,
                        value=initial_speaker,
                        label="3. 选择已有说话人",
                        interactive=True,
                    )
                    speakers_table = gr.Dataframe(
                        headers=["说话人"],
                        value=_speaker_rows(initial_group),
                        label="组内说话人",
                        interactive=False,
                    )
                    with gr.Row():
                        list_speakers_btn = gr.Button("刷新组内列表")
                        delete_speaker_btn = gr.Button("删除所选说话人")

            speaker_msg = gr.Textbox(label="操作结果", interactive=False)
            manage_group.change(
                select_speaker_group,
                manage_group,
                [group_status, speaker_select, speakers_table, speaker_group, job_speaker_group],
            )
            create_group_btn.click(
                create_speaker_group_ui,
                outputs=[
                    speaker_group,
                    job_speaker_group,
                    manage_group,
                    groups_table,
                    group_status,
                    speaker_select,
                    speakers_table,
                    speaker_msg,
                ],
            )
            refresh_groups_btn.click(
                refresh_all_speaker_ui,
                manage_group,
                [
                    speaker_group,
                    job_speaker_group,
                    manage_group,
                    groups_table,
                    group_status,
                    speaker_select,
                    speakers_table,
                ],
            )
            delete_group_btn.click(
                delete_speaker_group_ui,
                manage_group,
                [
                    speaker_group,
                    job_speaker_group,
                    manage_group,
                    groups_table,
                    group_status,
                    speaker_select,
                    speakers_table,
                    speaker_msg,
                ],
            )
            reg_speaker_btn.click(
                register_ui_speaker,
                [manage_group, speaker_name, speaker_audio],
                [
                    speaker_msg,
                    speaker_select,
                    speakers_table,
                    groups_table,
                    group_status,
                    speaker_group,
                    job_speaker_group,
                ],
            )
            list_speakers_btn.click(
                list_ui_speakers,
                manage_group,
                [group_status, speaker_select, speakers_table],
            )
            delete_speaker_btn.click(
                delete_ui_speaker,
                [manage_group, speaker_select],
                [
                    speaker_msg,
                    speaker_select,
                    speakers_table,
                    groups_table,
                    group_status,
                    speaker_group,
                    job_speaker_group,
                ],
            )

        with gr.Tab("实时录音"):
            gr.Markdown(
                """
### 实时录音转写（流式）

浏览器麦克风实时识别，复用 `/api/v1/realtime/transcriptions` WebSocket。

**麦克风限制**：浏览器安全策略要求 HTTPS 或 `localhost` 访问才能开麦克风。
局域网（如 `http://192.168.x.x:17767/ui`）会被浏览器禁用麦克风，需用 `localhost` 或配 HTTPS。

**模式说明**：
- `2pass`（推荐）：实时输出 + 离线修正
- `online`：纯流式，最低延迟，无标点/情感/事件
- `offline`：VAD 断句后整句识别，含标点/情感/事件

本页会在 WebSocket 连接后发送 `session.start` 配置事件，参数结构与 HTTP API 的
`features/options/fallback` 保持一致。说话人、声纹、情感、事件和标点只在
`offline` 或 `2pass` 的最终段 `transcript.segment` 中返回。
"""
            )
            rt_session = gr.State(value=RealtimeSession())
            with gr.Row():
                with gr.Column(scale=1):
                    rt_mode = gr.Radio(
                        ["2pass", "online", "offline"],
                        value="2pass",
                        label="识别模式",
                    )
                    rt_hotwords = gr.Textbox(
                        label="热词 JSON",
                        placeholder='{"FunASR": 20}',
                    )
                    rt_spk = gr.Checkbox(label="说话人分离", value=False)
                    rt_speaker_match = gr.Checkbox(label="声纹匹配", value=False)
                    rt_speaker_group = gr.Dropdown(
                        choices=initial_group_choices,
                        value=initial_group,
                        label="声纹组",
                        interactive=True,
                    )
                    rt_emo = gr.Checkbox(label="情感识别", value=False)
                    rt_events = gr.Checkbox(label="事件检测", value=False)
                    rt_punctuation = gr.Checkbox(label="标点恢复", value=True)
                with gr.Column(scale=2):
                    rt_mic = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="麦克风",
                        waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                    )
                    rt_status = gr.Textbox(label="连接状态", value="等待录音", interactive=False)
                    rt_clear_btn = gr.Button("清空结果")

            rt_output = gr.Textbox(
                label="累计识别结果（最终段持续累加，末尾为实时中间结果）",
                lines=15,
                value="等待录音...",
            )
            rt_raw = gr.Code(label="最近一帧 JSON", language="json")

            # Gradio 4+ streaming: handler 每次 chunk 触发
            gr.on(
                rt_mic.stream,
                fn=realtime_stream,
                inputs=[
                    rt_mic,
                    rt_session,
                    rt_mode,
                    rt_hotwords,
                    rt_spk,
                    rt_speaker_match,
                    rt_speaker_group,
                    rt_emo,
                    rt_events,
                    rt_punctuation,
                ],
                outputs=[rt_output, rt_session, rt_status, rt_raw],
            )
            rt_clear_btn.click(
                realtime_reset,
                outputs=[rt_output, rt_session, rt_status, rt_raw],
            )

        with gr.Tab("服务状态"):
            status_btn = gr.Button("刷新状态", variant="primary")
            status_text = gr.Textbox(label="状态", lines=5)
            status_json = gr.Code(label="状态 JSON", language="json")
            status_btn.click(service_status, outputs=[status_text, status_json])
            matrix_btn = gr.Button("查看模型能力矩阵")
            matrix_json = gr.Code(label="模型能力矩阵", language="json")
            matrix_btn.click(model_matrix, outputs=matrix_json)

        with gr.Tab("接口说明"):
            gr.Markdown(api_reference_text())

    return demo
