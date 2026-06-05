"""模型注册表 — 配置文件中选模型，API 不暴露"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from funasr import AutoModel

from server.models.config import (
    DEVICE, NGPU, NCPU, WORKER_THREADS, PRELOAD_ALL, ENABLE_STREAMING,
    CONCURRENT_VAD, CONCURRENT_ASR_ONLINE, CONCURRENT_ASR_OFFLINE,
    CONCURRENT_PUNC, CONCURRENT_SV, CONCURRENT_EMOTION,
    ASR_CONFIG, ASR_CONFIG_WITH_SPK, STREAMING_CONFIG,
    VAD_CONFIG, PUNC_CONFIG, SV_CONFIG, EMOTION_CONFIG,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表：配置文件中选定模型，所有 API 共享"""

    _instance = None

    def __init__(self):
        self.device = DEVICE
        self._models: dict[str, AutoModel] = {}
        self.executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)
        self.sem_vad = asyncio.Semaphore(CONCURRENT_VAD)
        self.sem_asr_online = asyncio.Semaphore(CONCURRENT_ASR_ONLINE)
        self.sem_asr_offline = asyncio.Semaphore(CONCURRENT_ASR_OFFLINE)
        self.sem_punc = asyncio.Semaphore(CONCURRENT_PUNC)
        self.sem_sv = asyncio.Semaphore(CONCURRENT_SV)
        self.sem_emotion = asyncio.Semaphore(CONCURRENT_EMOTION)

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _base_kwargs(self) -> dict:
        return {"ngpu": NGPU, "ncpu": NCPU, "device": DEVICE,
                "disable_pbar": True, "disable_log": True, "disable_update": True}

    def _load(self, key: str, config: dict) -> AutoModel:
        if key not in self._models:
            logger.info(f"加载模型: {key} ...")
            cfg = config.copy()
            cfg.update(self._base_kwargs())
            self._models[key] = AutoModel(**cfg)
            logger.info(f"模型加载完成: {key}")
        return self._models[key]

    # ── 核心方法 ────────────────────────────────

    def get(self, with_spk: bool = False) -> AutoModel:
        """获取离线 ASR 模型（配置文件中选定的那个）"""
        cfg = ASR_CONFIG_WITH_SPK if with_spk else ASR_CONFIG
        return self._load("asr", cfg)

    def get_streaming(self) -> AutoModel:
        return self._load("streaming", STREAMING_CONFIG)

    def get_aux(self, name: str) -> AutoModel:
        """获取辅助模型: vad / punc / sv / emotion"""
        configs = {"vad": VAD_CONFIG, "punc": PUNC_CONFIG,
                   "sv": SV_CONFIG, "emotion": EMOTION_CONFIG}
        if name not in configs:
            raise ValueError(f"未知辅助模型: {name}")
        return self._load(name, configs[name])

    # ── 预加载 ──────────────────────────────────

    def preload(self):
        """启动时加载模型"""
        logger.info(f"预加载模型 (device={DEVICE}, model={MODEL_NAME}) ...")
        self.get()                          # 离线 ASR
        self.get(with_spk=True)             # 离线 ASR + 说话人
        if ENABLE_STREAMING:
            self.get_streaming()            # 流式 ASR
        self.get_aux("vad")                 # VAD
        self.get_aux("punc")                # 标点
        self.get_aux("sv")                  # 声纹（注册/匹配需要）
        self.get_aux("emotion")             # 情感
        logger.info(f"所有模型加载完成！共 {len(self._models)} 个")

    # ── 状态 ────────────────────────────────────
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
