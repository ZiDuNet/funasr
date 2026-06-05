"""WebSocket 流式识别 — 支持 offline / online / 2pass"""

import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from server.core.inference import (
    infer_vad, infer_asr_online, infer_asr_offline_ws, infer_punc,
)
from server.core.audio import pcm_duration_ms

logger = logging.getLogger(__name__)


def register_ws_endpoint(app):
    """在 FastAPI app 上注册 WebSocket 端点"""

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept(subprotocol="binary")
        logger.info("WebSocket 新连接")

        # 每连接状态
        state = {
            "mode": "2pass",
            "wav_name": "microphone",
            "is_speaking": True,
            "chunk_interval": 10,
            "audio_fs": 16000,
            "vad_pre_idx": 0,
            "status_asr": {},           # 离线 ASR 参数
            "status_asr_online": {       # 流式 ASR 参数
                "cache": {},
                "is_final": False,
            },
            "status_vad": {              # VAD 参数
                "cache": {},
                "is_final": False,
            },
            "status_punc": {             # 标点参数
                "cache": {},
            },
        }

        frames = []
        frames_asr = []
        frames_asr_online = []
        speech_start = False
        speech_end_i = -1

        try:
            while True:
                msg = await websocket.receive()

                if msg["type"] == "websocket.receive":
                    if "text" in msg:
                        # ── JSON 配置消息 ──
                        try:
                            cfg = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue

                        if "is_speaking" in cfg:
                            state["is_speaking"] = bool(cfg["is_speaking"])
                            state["status_asr_online"]["is_final"] = not state["is_speaking"]

                        if "chunk_interval" in cfg:
                            state["chunk_interval"] = int(cfg["chunk_interval"])

                        if "wav_name" in cfg:
                            state["wav_name"] = cfg["wav_name"]

                        if "chunk_size" in cfg:
                            cs = cfg["chunk_size"]
                            if isinstance(cs, str):
                                cs = [int(x.strip()) for x in cs.split(",") if x.strip()]
                            state["status_asr_online"]["chunk_size"] = [int(x) for x in cs]

                        if "encoder_chunk_look_back" in cfg:
                            state["status_asr_online"]["encoder_chunk_look_back"] = cfg["encoder_chunk_look_back"]

                        if "decoder_chunk_look_back" in cfg:
                            state["status_asr_online"]["decoder_chunk_look_back"] = cfg["decoder_chunk_look_back"]

                        if "hotwords" in cfg:
                            state["status_asr"]["hotword"] = cfg["hotwords"]
                            state["status_asr_online"]["hotword"] = cfg["hotwords"]

                        if "mode" in cfg:
                            state["mode"] = cfg["mode"]

                        if "audio_fs" in cfg:
                            state["audio_fs"] = int(cfg["audio_fs"])

                    elif "bytes" in msg:
                        # ── 二进制音频消息 ──
                        pcm = msg["bytes"]

                        if "chunk_size" not in state["status_asr_online"]:
                            logger.warning("chunk_size 未设置，跳过音频帧")
                            continue

                        # 设置 VAD chunk_size
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

                        # online ASR
                        frames_asr_online.append(pcm)
                        state["status_asr_online"]["is_final"] = (speech_end_i != -1)

                        if (len(frames_asr_online) % state["chunk_interval"] == 0) or state["status_asr_online"]["is_final"]:
                            if state["mode"] in ("2pass", "online"):
                                audio_in = b"".join(frames_asr_online)
                                try:
                                    rec = await infer_asr_online(audio_in, state["status_asr_online"])
                                    if rec.get("text"):
                                        if not (state["mode"] == "2pass" and state["status_asr_online"].get("is_final", False)):
                                            mode = "2pass-online" if "2pass" in state["mode"] else state["mode"]
                                            await websocket.send_json({
                                                "mode": mode,
                                                "text": rec["text"],
                                                "wav_name": state["wav_name"],
                                                "is_final": False,
                                            })
                                except Exception as e:
                                    logger.error(f"流式 ASR 错误: {e}")
                            frames_asr_online = []

                        if speech_start:
                            frames_asr.append(pcm)

                        # VAD
                        try:
                            speech_start_i, speech_end_i = await infer_vad(pcm, state["status_vad"])
                        except Exception:
                            speech_start_i, speech_end_i = -1, -1

                        if speech_start_i != -1:
                            speech_start = True
                            if duration_ms > 0:
                                beg_bias = (state["vad_pre_idx"] - speech_start_i) // duration_ms
                            else:
                                beg_bias = 0
                            frames_pre = frames[-beg_bias:] if beg_bias > 0 else []
                            frames_asr = []
                            frames_asr.extend(frames_pre)

                        # 离线触发点
                        if (speech_end_i != -1) or (not state["is_speaking"]):
                            if state["mode"] in ("2pass", "offline"):
                                audio_in = b"".join(frames_asr)
                                if len(audio_in) > 0:
                                    try:
                                        # 离线 ASR
                                        rec = await infer_asr_offline_ws(audio_in, state["status_asr"])
                                        text = rec.get("text", "")

                                        # 标点
                                        if text:
                                            try:
                                                punc_res = await infer_punc(text, state["status_punc"])
                                                if punc_res.get("text"):
                                                    text = punc_res["text"]
                                            except Exception:
                                                pass

                                        if text:
                                            mode = "2pass-offline" if "2pass" in state["mode"] else state["mode"]
                                            await websocket.send_json({
                                                "mode": mode,
                                                "text": text,
                                                "wav_name": state["wav_name"],
                                                "is_final": True,
                                                "timestamp": rec.get("timestamp"),
                                            })
                                    except Exception as e:
                                        logger.error(f"离线 ASR 错误: {e}")

                            frames_asr = []
                            speech_start = False
                            frames_asr_online = []
                            state["status_asr_online"]["cache"] = {}

                            if not state["is_speaking"]:
                                state["vad_pre_idx"] = 0
                                frames = []
                                state["status_vad"]["cache"] = {}
                                speech_end_i = -1
                            else:
                                frames = frames[-20:]

        except WebSocketDisconnect:
            logger.info("WebSocket 断开")
        except Exception as e:
            logger.error(f"WebSocket 错误: {e}")
