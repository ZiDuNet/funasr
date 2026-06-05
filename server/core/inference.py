"""统一推理层 — 线程池 + Semaphore 并发控制"""

import asyncio
import functools
import logging

from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


async def run_blocking(fn, *args, sem: asyncio.Semaphore | None = None, **kwargs):
    """阻塞推理丢线程池，异步返回不卡事件循环"""
    registry = ModelRegistry.get_instance()
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    if sem is None:
        return await loop.run_in_executor(registry.executor, call)
    async with sem:
        return await loop.run_in_executor(registry.executor, call)


def _generate_sync(model, audio_input, **kwargs):
    return model.generate(input=audio_input, **kwargs)


async def infer_offline(audio_input, with_spk: bool = False, **kwargs) -> dict:
    """离线 ASR 推理"""
    registry = ModelRegistry.get_instance()
    model = registry.get(with_spk=with_spk)
    kwargs.setdefault("batch_size_s", 300)
    kwargs.setdefault("merge_vad", True)
    kwargs.setdefault("merge_length_s", 15)
    kwargs.setdefault("batch_size_threshold_s", 60)
    kwargs.setdefault("language", "auto")
    kwargs.setdefault("use_itn", True)
    result_list = await run_blocking(
        _generate_sync, model, audio_input,
        sem=registry.sem_asr_offline, **kwargs,
    )
    return result_list[0] if result_list else {}


async def infer_vad(audio_input, status_dict: dict) -> tuple:
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get_aux("vad"), audio_input, **status_dict,
        sem=registry.sem_vad,
    )
    segments = out[0].get("value", []) if out else []
    s, e = -1, -1
    if len(segments) == 1:
        if segments[0][0] != -1: s = segments[0][0]
        if segments[0][1] != -1: e = segments[0][1]
    return s, e


async def infer_asr_online(audio_input, status_dict: dict) -> dict:
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get_streaming(), audio_input, **status_dict,
        sem=registry.sem_asr_online,
    )
    return out[0] if out else {}


async def infer_asr_offline_ws(audio_input, status_dict: dict, with_spk: bool = False) -> dict:
    """WebSocket 离线阶段推理"""
    registry = ModelRegistry.get_instance()
    model = registry.get(with_spk=with_spk)
    gen = status_dict.copy()
    gen.setdefault("batch_size_s", 300)
    gen.setdefault("merge_vad", True)
    gen.setdefault("merge_length_s", 15)
    gen.setdefault("batch_size_threshold_s", 60)
    gen.setdefault("language", "auto")
    gen.setdefault("use_itn", True)
    out = await run_blocking(
        _generate_sync, model, audio_input,
        sem=registry.sem_asr_offline, **gen,
    )
    return out[0] if out else {}


async def infer_punc(text_input, status_dict: dict) -> dict:
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get_aux("punc"), text_input, **status_dict,
        sem=registry.sem_punc,
    )
    return out[0] if out else {}


async def infer_emotion(audio_input) -> dict:
    registry = ModelRegistry.get_instance()
    out = await run_blocking(
        _generate_sync, registry.get_aux("emotion"), audio_input,
        granularity="utterance", sem=registry.sem_emotion,
    )
    return out[0] if out else {}
