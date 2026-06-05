"""WebSocket 流式识别 — 支持 offline / online / 2pass + 说话人分离 + 情感 + 事件"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from server.core.inference import (
    infer_vad, infer_asr_online, infer_asr_offline_ws, infer_punc,
)
from server.core.audio import pcm_duration_ms
from server.core.postprocess import clean_text, extract_emotion, extract_events
from server.core.speaker_db import match_segments
from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def register_ws_endpoint(app):
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
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
            "speaker_diarization": False,
            "speaker_group": "",             # 声纹组 ID
            "emotion": False,                # 情感识别
            "events": False,                 # 事件检测
            "vad_pre_idx": 0,                # VAD 预处理音频累计时长
            "status_asr": {"batch_size_s": 300},
            "status_asr_online": {"cache": {}, "is_final": False},
            "status_vad": {"cache": {}, "is_final": False},
            "status_punc": {"cache": {}},
        }

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

                        for key in ("is_speaking", "wav_name", "chunk_interval",
                                     "audio_fs", "wav_format", "mode"):
                            if key in cfg:
                                state[key] = cfg[key]

                        if "is_speaking" in cfg:
                            v = bool(cfg["is_speaking"])
                            state["is_speaking"] = v
                            state["status_asr_online"]["is_final"] = not v

                        if "chunk_size" in cfg:
                            cs = cfg["chunk_size"]
                            if isinstance(cs, str):
                                cs = [int(x.strip()) for x in cs.split(",") if x.strip()]
                            state["status_asr_online"]["chunk_size"] = [int(x) for x in cs]

                        if "hotwords" in cfg:
                            state["status_asr"]["hotword"] = cfg["hotwords"]
                            state["status_asr_online"]["hotword"] = cfg["hotwords"]

                        if "speaker_diarization" in cfg:
                            state["speaker_diarization"] = bool(cfg["speaker_diarization"])
                        if "speaker_group" in cfg:
                            state["speaker_group"] = str(cfg["speaker_group"])
                        if "emotion" in cfg:
                            state["emotion"] = bool(cfg["emotion"])
                        if "events" in cfg:
                            state["events"] = bool(cfg["events"])
                        if "itn" in cfg:
                            state["itn"] = bool(cfg["itn"])

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

                                        # 标点（SenseVoice 自带，但流式阶段也可能需要）
                                        if text and state.get("punc_enabled", True):
                                            try:
                                                punc_res = await infer_punc(text, state["status_punc"])
                                                if punc_res.get("text"):
                                                    text = punc_res["text"]
                                            except Exception:
                                                pass

                                        if text:
                                            mode_label = "2pass-offline" if "2pass" in state["mode"] else state["mode"]
                                            # 声纹匹配（需同时启用 speaker_diarization + speaker_group）
                                            sentence_info = rec.get("sentence_info")
                                            if (state.get("speaker_diarization")
                                                    and state.get("speaker_group")
                                                    and sentence_info):
                                                # match_segments 使用 speaker_id 字段
                                                for si in sentence_info:
                                                    if "spk" in si:
                                                        si["speaker_id"] = si["spk"]
                                                registry = ModelRegistry.get_instance()
                                                match_segments(
                                                    sentence_info, audio_in,
                                                    state["speaker_group"],
                                                    registry.get_aux("sv"),
                                                )

                                            resp = {
                                                "mode": mode_label,
                                                "text": clean_text(text),
                                                "wav_name": state["wav_name"],
                                                "is_final": True,
                                                "timestamp": rec.get("timestamp"),
                                                "sentence_info": sentence_info,
                                            }
                                            # 情感/事件仅在请求时返回
                                            if state.get("emotion"):
                                                emo = extract_emotion(text)
                                                if emo:
                                                    resp["emotion"] = emo
                                            if state.get("events"):
                                                evt = extract_events(text)
                                                if evt:
                                                    resp["events"] = evt
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

        except WebSocketDisconnect:
            logger.info("WebSocket 断开")
        except Exception as e:
            logger.error(f"WebSocket 错误: {e}")
