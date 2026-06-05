"""统一推理层 — 线程池 + Semaphore 并发控制"""

import asyncio
import functools
import logging

from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


async def run_blocking(fn, *args, sem: asyncio.Semaphore | None = None, **kwargs):
    """把阻塞推理函数丢到线程池执行，避免卡事件循环。

    Args:
        fn: 阻塞函数（如 model.generate）
        sem: 可选的 Semaphore，用于限流
    """
    registry = ModelRegistry.get_instance()
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    if sem is None:
        return await loop.run_in_executor(registry.executor, call)
    async with sem:
        return await loop.run_in_executor(registry.executor, call)


def _generate_sync(model, audio_input, **kwargs):
    """同步推理包装"""
    return model.generate(input=audio_input, **kwargs)


async def infer_offline(
    audio_input,
    model_name: str = "sensevoice",
    speaker_diarization: bool = False,
    punctuation: bool = True,
    **generate_kwargs,
) -> dict:
    """离线推理（OpenAI API / HTTP REST 使用）

    根据 speaker_diarization 参数决定是否加载和调用声纹模型。
    """
    registry = ModelRegistry.get_instance()

    # 说话人分离：加载带 spk_model 的版本
    # 官方确认：SenseVoice + cam++ 无需额外 punc_model 就能做
    model_name = "sensevoice_spk" if speaker_diarization else "sensevoice"
    model = registry.get(model_name)
    result_list = await run_blocking(
        _generate_sync, model, audio_input,
        batch_size_s=300, **generate_kwargs,
        sem=registry.sem_asr_offline,
    )

    return result_list[0] if result_list else {}


async def infer_vad(audio_input, status_dict: dict) -> tuple:
    """VAD 推理（WebSocket 使用）"""
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get("vad"), audio_input, **status_dict,
        sem=registry.sem_vad,
    )
    segments = out[0].get("value", []) if out else []
    speech_start = -1
    speech_end = -1
    if len(segments) == 1:
        if segments[0][0] != -1:
            speech_start = segments[0][0]
        if segments[0][1] != -1:
            speech_end = segments[0][1]
    return speech_start, speech_end


async def infer_asr_online(audio_input, status_dict: dict) -> dict:
    """流式 ASR 推理（WebSocket online 使用）"""
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get("streaming"), audio_input, **status_dict,
        sem=registry.sem_asr_online,
    )
    return out[0] if out else {}


async def infer_asr_offline_ws(audio_input, status_dict: dict) -> dict:
    """离线 ASR 推理（WebSocket offline / 2pass 使用）"""
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get("sensevoice"), audio_input, **status_dict,
        sem=registry.sem_asr_offline,
    )
    return out[0] if out else {}


async def infer_punc(text_input, status_dict: dict) -> dict:
    """标点推理"""
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get("punc"), text_input, **status_dict,
        sem=registry.sem_punc,
    )
    return out[0] if out else {}
