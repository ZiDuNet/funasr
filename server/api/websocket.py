"""WebSocket 流式识别 — 支持 offline / online / 2pass + 说话人分离 + 情感 + 事件"""

import json
import logging
import os

from fastapi import WebSocket, WebSocketDisconnect

from server.core.schemas import build_config
from server.core.inference import (
    infer_vad, infer_asr_online, infer_asr_offline_ws,
)
from server.core.audio import pcm_duration_ms
from server.core.postprocess import clean_text, extract_metadata
from server.core.speaker_db import match_segments, SpeakerTracker
from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)
_API_TOKEN = os.environ.get("API_TOKEN", "").strip()

WS_PROTOCOL_DOC = {
    "websocket": {
        "url": "/api/v1/realtime/transcriptions",
        "auth": "启用 API_TOKEN 时，在查询参数追加 ?token=your-token",
        "audio_frame": "binary 16 kHz / mono / signed int16 little-endian PCM",
        "flow": [
            "建立 WebSocket 连接",
            "发送 session.start JSON 配置帧",
            "持续发送 PCM 二进制音频帧",
            "停止录音时发送 audio.end JSON 控制帧",
            "服务端返回 session.started、transcript.delta、transcript.segment 或 error",
        ],
    },
    "browser_ui": {
        "path": "/",
        "note": "WebUI 只有根路径；实时录音 Tab 直连 WebSocket，音频帧直接以 PCM 二进制发送。",
    },
    "session_start_example": {
        "type": "session.start",
        "mode": "2pass",
        "audio_fs": 16000,
        "wav_format": "pcm",
        "chunk_size": [0, 10, 5],
        "chunk_interval": 10,
        "encoder_chunk_look_back": 4,
        "decoder_chunk_look_back": 1,
        "itn": True,
        "features": {
            "diarization": True,
            "speaker_match": {
                "enabled": True,
                "group_id": "grp_xxx",
            },
            "emotion": True,
            "events": True,
            "punctuation": True,
            "raw": False,
        },
        "options": {
            "language": "auto",
            "hotwords": {"FunASR": 20},
        },
        "fallback": "auto",
    },
    "stop_example": {"type": "audio.end"},
    "events": {
        "session.started": "配置已生效，返回本次会话模式、音频格式和能力开关。",
        "transcript.delta": "online/2pass 的实时中间文本，用于一边说一边出字。",
        "transcript.segment": "offline/2pass 的最终片段，VAD 断句后返回，可包含说话人、声纹、情感、事件、标点。",
        "error": "配置或推理错误。",
    },
    "modes": {
        "online": "只返回 transcript.delta，延迟低；不返回说话人、声纹、情感、事件、完整标点。",
        "offline": "VAD 断句后返回 transcript.segment；可返回增强字段，但没有实时中间文本。",
        "2pass": "先返回 transcript.delta，断句后返回 transcript.segment 修正结果；推荐会议录音使用。",
    },
    "speaker_consistency": (
        "说话人编号一致性只在单个 WebSocket 会话内维护。开启 diarization 后，服务端会用会话级 "
        "SpeakerTracker 尽量保持 speaker_0/speaker_1 跨多句一致；开启 speaker_match 并提供 "
        "group_id 后，最终段会尝试匹配注册声纹并返回姓名和分数。"
    ),
}


def _parse_chunk_size(value) -> list[int]:
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError("chunk_size must be a list or comma-separated string")
    if len(value) != 3:
        raise ValueError("chunk_size must contain 3 integers")
    return [int(x) for x in value]


def _apply_config_frame(state: dict, cfg: dict) -> None:
    """Apply the WS session.start config frame."""
    event_type = cfg.get("type")
    if event_type == "audio.end":
        state["is_speaking"] = False
        state["status_asr_online"]["is_final"] = True
        return

    if event_type and event_type != "session.start":
        raise ValueError("first/control text frame type must be session.start or audio.end")

    for key in ("is_speaking", "wav_name", "chunk_interval",
                "audio_fs", "wav_format", "mode"):
        if key in cfg:
            state[key] = cfg[key]

    if "mode" in cfg and state["mode"] not in {"online", "offline", "2pass"}:
        raise ValueError("mode must be one of: online, offline, 2pass")

    if "chunk_size" in cfg:
        state["status_asr_online"]["chunk_size"] = _parse_chunk_size(cfg["chunk_size"])

    if "is_speaking" in cfg:
        v = bool(cfg["is_speaking"])
        state["is_speaking"] = v
        state["status_asr_online"]["is_final"] = not v

    if "encoder_chunk_look_back" in cfg:
        state["status_asr_online"]["encoder_chunk_look_back"] = int(
            cfg["encoder_chunk_look_back"]
        )
    if "decoder_chunk_look_back" in cfg:
        state["status_asr_online"]["decoder_chunk_look_back"] = int(
            cfg["decoder_chunk_look_back"]
        )

    config_keys = {
        "features", "options", "fallback", "language", "diarization",
        "speaker_diarization", "speaker_match", "speaker_group", "emotion",
        "events", "punctuation", "hotwords", "raw",
    }
    if not any(key in cfg for key in config_keys):
        return

    config_json = json.dumps(cfg, ensure_ascii=False)
    tc = build_config(
        config_json=config_json,
        language=cfg.get("language"),
        diarization=cfg.get("diarization"),
        speaker_diarization=cfg.get("speaker_diarization"),
        speaker_match=cfg.get("speaker_match"),
        speaker_group=cfg.get("speaker_group"),
        emotion=cfg.get("emotion"),
        events=cfg.get("events"),
        punctuation=cfg.get("punctuation"),
        hotwords=cfg.get("hotwords"),
        raw=cfg.get("raw"),
        fallback=cfg.get("fallback"),
    )

    state["language"] = tc.language
    state["fallback"] = tc.fallback
    state["speaker_diarization"] = tc.features.diarization
    state["speaker_match"] = tc.features.speaker_match
    state["speaker_group"] = tc.features.speaker_group
    state["emotion"] = tc.features.emotion
    state["events"] = tc.features.events
    state["punctuation"] = tc.features.punctuation
    state["raw"] = tc.features.raw
    state["requested_features"] = tc.requested_features()

    if tc.hotwords:
        state["status_asr"]["hotword"] = tc.hotwords
        state["status_asr_online"]["hotword"] = tc.hotwords
    if "itn" in cfg:
        state["itn"] = bool(cfg["itn"])
    state["status_asr"]["language"] = tc.language
    state["status_asr"]["use_itn"] = state["itn"]


def _milliseconds_to_seconds(value) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return round(number / 1000.0, 3)


def _build_ws_canonical(rec: dict, sentence_info: list | None,
                        state: dict, audio_duration: float) -> dict:
    raw_text = rec.get("text", "")
    text = clean_text(raw_text)
    top_metadata = extract_metadata(raw_text)
    sentences = []
    for idx, seg in enumerate(sentence_info or []):
        seg_text = seg.get("text") or seg.get("sentence", "")
        seg_metadata = extract_metadata(seg_text)
        item = {
            "id": idx,
            "paragraph_id": 0,
            "start": _milliseconds_to_seconds(seg.get("start")),
            "end": _milliseconds_to_seconds(seg.get("end")),
            "text": clean_text(seg_text),
        }
        if seg_metadata.get("language"):
            item["language"] = seg_metadata["language"]
        if "itn" in seg_metadata:
            item["itn"] = seg_metadata["itn"]
        if state.get("speaker_diarization") and "spk" in seg:
            speaker = {"id": f"speaker_{seg['spk']}"}
            if "speaker" in seg:
                speaker["name"] = seg["speaker"]
            if "speaker_score" in seg:
                speaker["score"] = seg["speaker_score"]
            if state.get("speaker_group"):
                speaker["group_id"] = state["speaker_group"]
            item["speaker"] = speaker
        if state.get("emotion"):
            item["emotion"] = seg_metadata.get("emotion") or top_metadata.get("emotion")
        if state.get("events"):
            item["events"] = seg_metadata.get("events") or top_metadata.get("events", [])
        sentences.append(item)

    if not sentences and text:
        sentences.append({
            "id": 0,
            "paragraph_id": 0,
            "start": 0.0,
            "end": round(audio_duration, 3),
            "text": text,
        })

    paragraph = {
        "id": 0,
        "start": sentences[0]["start"] if sentences else 0.0,
        "end": sentences[-1]["end"] if sentences else 0.0,
        "text": "".join(s["text"] for s in sentences).strip(),
        "sentence_ids": [s["id"] for s in sentences],
    }

    canonical = {
        "object": "realtime.transcription.segment",
        "text": text,
        "duration": round(audio_duration, 3),
        "paragraph_count": 1 if sentences else 0,
        "sentence_count": len(sentences),
        "paragraphs": [paragraph] if sentences else [],
        "sentences": sentences,
    }
    if top_metadata.get("language"):
        canonical["language"] = top_metadata["language"]
    canonical["itn"] = top_metadata.get("itn", state.get("itn", True))
    if state.get("emotion"):
        canonical["emotion"] = top_metadata.get("emotion")
    if state.get("events"):
        canonical["events"] = top_metadata.get("events", [])
    if state.get("speaker_group"):
        canonical["speaker_group"] = state["speaker_group"]
    return canonical


def register_ws_endpoint(app):
    @app.get(
        "/api/v1/realtime/transcriptions/protocol",
        tags=["Realtime WebSocket"],
        summary="实时转写 WebSocket 协议说明",
        description=(
            "Swagger/OpenAPI 不能直接描述 WebSocket 操作；本端点用于在 /docs 中展示 "
            "`/api/v1/realtime/transcriptions` 的请求流程、参数和返回事件。"
        ),
        responses={
            200: {
                "description": "WebSocket 协议说明",
                "content": {
                    "application/json": {
                        "example": WS_PROTOCOL_DOC,
                    }
                },
            }
        },
    )
    async def ws_protocol_doc():
        return WS_PROTOCOL_DOC

    @app.websocket("/api/v1/realtime/transcriptions")
    async def ws_endpoint(websocket: WebSocket):
        if _API_TOKEN and websocket.query_params.get("token", "") != _API_TOKEN:
            await websocket.close(code=1008)
            return

        await websocket.accept(subprotocol="binary")
        logger.info("WebSocket 新连接")

        state = {
            "mode": "2pass",
            "wav_name": "microphone",
            "is_speaking": True,
            "chunk_interval": 10,
            "audio_fs": 16000,
            "wav_format": "pcm",
            "itn": True,
            "language": "auto",
            "fallback": "error",
            "speaker_diarization": False,
            "speaker_match": False,
            "speaker_group": "",             # 声纹组 ID
            "emotion": False,                # 情感识别
            "events": False,                 # 事件检测
            "punctuation": True,              # 最终段标点，取决于离线模型能力
            "raw": False,
            "requested_features": ["asr", "sentence_timestamps", "paragraphs"],
            "vad_pre_idx": 0,                # VAD 预处理音频累计时长
            "status_asr": {"batch_size_s": 300, "language": "auto", "use_itn": True},
            "status_asr_online": {
                "cache": {},
                "is_final": False,
                "chunk_size": [0, 10, 5],
                "encoder_chunk_look_back": 4,
                "decoder_chunk_look_back": 1,
            },
            "status_vad": {"cache": {}, "is_final": False, "max_single_segment_time": 15000},
            "status_punc": {"cache": {}},
        }

        # 跨段说话人追踪器（流式中维护全局 speaker_id 一致性）
        speaker_tracker = None

        frames, frames_asr, frames_asr_online = [], [], []
        speech_start, speech_end_i = False, -1
        partial_text_online = ""

        try:
            while True:
                msg = await websocket.receive()

                if msg["type"] == "websocket.receive":
                    if "text" in msg:
                        try:
                            cfg = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue
                        try:
                            _apply_config_frame(state, cfg)
                        except Exception as exc:
                            await websocket.send_json({
                                "type": "error",
                                "code": "invalid_config",
                                "message": str(exc),
                            })
                            continue
                        if cfg.get("type") == "session.start":
                            await websocket.send_json({
                                "type": "session.started",
                                "mode": state["mode"],
                                "audio": {
                                    "format": state["wav_format"],
                                    "sample_rate": state["audio_fs"],
                                    "channels": 1,
                                },
                                "features": {
                                    "diarization": state["speaker_diarization"],
                                    "speaker_match": state["speaker_match"],
                                    "speaker_group": state["speaker_group"],
                                    "emotion": state["emotion"],
                                    "events": state["events"],
                                    "punctuation": state["punctuation"],
                                    "raw": state["raw"],
                                },
                            })

                    elif "bytes" in msg:
                        pcm = msg["bytes"]

                        if "chunk_size" not in state["status_asr_online"]:
                            logger.warning("chunk_size 未设置，跳过音频帧")
                            continue

                        try:
                            state["status_vad"]["chunk_size"] = int(
                                state["status_asr_online"]["chunk_size"][1]
                                * 60 / state["chunk_interval"]
                            )
                        except Exception:
                            pass

                        frames.append(pcm)
                        duration_ms = pcm_duration_ms(pcm, fs=state["audio_fs"])
                        state["vad_pre_idx"] += duration_ms

                        # ── online ASR ──────────────────────────
                        frames_asr_online.append(pcm)
                        state["status_asr_online"]["is_final"] = (speech_end_i != -1)

                        if (len(frames_asr_online) % state["chunk_interval"] == 0) or state["status_asr_online"]["is_final"]:
                            if state["mode"] in ("2pass", "online"):
                                audio_in = b"".join(frames_asr_online)
                                try:
                                    rec = await infer_asr_online(audio_in, state["status_asr_online"])
                                    if rec.get("text"):
                                        if not (state["mode"] == "2pass" and state["status_asr_online"].get("is_final")):
                                            partial_text_online = rec["text"]
                                            mode_label = "2pass-online" if "2pass" in state["mode"] else state["mode"]
                                            await websocket.send_json({
                                                "type": "transcript.delta",
                                                "mode": mode_label,
                                                "text": rec["text"],
                                                "wav_name": state["wav_name"],
                                                "is_final": False,
                                            })
                                except Exception as e:
                                    logger.error(f"流式 ASR 错误: {e}")
                            frames_asr_online = []

                        if speech_start:
                            frames_asr.append(pcm)

                        # ── VAD ─────────────────────────────────
                        try:
                            speech_start_i, speech_end_i = await infer_vad(pcm, state["status_vad"])
                        except Exception:
                            speech_start_i, speech_end_i = -1, -1

                        if speech_start_i != -1:
                            speech_start = True
                            beg_bias = ((state["vad_pre_idx"] - speech_start_i) // duration_ms
                                        ) if duration_ms > 0 else 0
                            frames_pre = frames[-beg_bias:] if beg_bias > 0 else []
                            frames_asr = []
                            frames_asr.extend(frames_pre)

                        # ── 离线触发（VAD 断句 or 结束）─────────
                        if (speech_end_i != -1) or (not state["is_speaking"]):
                            if state["mode"] in ("2pass", "offline"):
                                audio_in = b"".join(frames_asr)
                                if len(audio_in) > 0:
                                    try:
                                        # 离线 ASR（支持 speaker_diarization）
                                        rec = await infer_asr_offline_ws(
                                            audio_in, state["status_asr"],
                                            with_spk=state.get("speaker_diarization", False),
                                        )
                                        text = rec.get("text", "")

                                        if text:
                                            mode_label = "2pass-offline" if "2pass" in state["mode"] else state["mode"]
                                            # ── 说话人处理 ──
                                            sentence_info = rec.get("sentence_info")
                                            if state.get("speaker_diarization") and sentence_info:
                                                registry = ModelRegistry.get_instance()

                                                # 1. 跨段一致性追踪（全局 speaker_id）
                                                if speaker_tracker is None:
                                                    speaker_tracker = SpeakerTracker(
                                                        registry.get_aux("sv"))
                                                sentence_info = speaker_tracker.track(
                                                    sentence_info, audio_in)

                                                # 2. 声纹匹配（speaker_match + speaker_group → 注册名）
                                                if state.get("speaker_match") and state.get("speaker_group"):
                                                    for si in sentence_info:
                                                        if "spk" in si:
                                                            si["speaker_id"] = si["spk"]
                                                    match_segments(
                                                        sentence_info, audio_in,
                                                        state["speaker_group"],
                                                        registry.get_aux("sv"),
                                                    )
                                                    # 清理临时字段
                                                    for si in sentence_info:
                                                        si.pop("speaker_id", None)

                                            canonical = _build_ws_canonical(
                                                rec,
                                                sentence_info,
                                                state,
                                                pcm_duration_ms(audio_in, fs=state["audio_fs"]) / 1000.0,
                                            )
                                            resp = {
                                                "type": "transcript.segment",
                                                "mode": mode_label,
                                                "text": text,                    # 原始文本（含标签）
                                                "clean_text": clean_text(text),  # 清洗后纯文本
                                                "wav_name": state["wav_name"],
                                                "is_final": True,
                                                "timestamp": rec.get("timestamp"),
                                                "sentence_info": sentence_info,
                                                "features": {
                                                    "requested": state["requested_features"],
                                                    "applied": [
                                                        "asr",
                                                        "sentence_timestamps",
                                                        "paragraphs",
                                                        *(
                                                            ["diarization"]
                                                            if state.get("speaker_diarization")
                                                            else []
                                                        ),
                                                        *(
                                                            ["speaker_match"]
                                                            if state.get("speaker_match")
                                                            and state.get("speaker_group")
                                                            else []
                                                        ),
                                                        *(
                                                            ["emotion"]
                                                            if state.get("emotion")
                                                            else []
                                                        ),
                                                        *(
                                                            ["events"]
                                                            if state.get("events")
                                                            else []
                                                        ),
                                                        *(
                                                            ["punctuation"]
                                                            if state.get("punctuation")
                                                            else []
                                                        ),
                                                    ],
                                                    "warnings": [],
                                                },
                                                **canonical,
                                            }
                                            if state.get("raw"):
                                                resp["raw"] = rec
                                            await websocket.send_json(resp)
                                    except Exception as e:
                                        logger.error(f"离线 ASR 错误: {e}")

                            frames_asr, frames_asr_online = [], []
                            speech_start = False
                            state["status_asr_online"]["cache"] = {}

                            if not state["is_speaking"]:
                                state["vad_pre_idx"], frames = 0, []
                                state["status_vad"]["cache"] = {}
                                speech_end_i = -1
                            else:
                                frames = frames[-20:]

                elif msg["type"] == "websocket.disconnect":
                    logger.info("WebSocket 断开")
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket 断开")
        except Exception as e:
            logger.error(f"WebSocket 错误: {e}")
