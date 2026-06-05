"""模型注册表 — 单例模式，所有 API 共享"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from funasr import AutoModel

from server.models.config import (
    DEVICE, NGPU, NCPU, WORKER_THREADS,
    CONCURRENT_VAD, CONCURRENT_ASR_ONLINE, CONCURRENT_ASR_OFFLINE,
    CONCURRENT_PUNC, CONCURRENT_SV, CONCURRENT_EMOTION,
    MODEL_CONFIGS, STREAMING_MODEL, VAD_MODEL, PUNC_MODEL, SV_MODEL, EMOTION_MODEL,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表，懒加载 + 共享"""

    _instance = None

    def __init__(self):
        self.device = DEVICE
        self.ngpu = NGPU
        self.ncpu = NCPU

        # 已加载的模型实例
        self._models: dict[str, AutoModel] = {}

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)

        # 并发 Semaphore
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
        """AutoModel 公共参数"""
        return {
            "ngpu": self.ngpu,
            "ncpu": self.ncpu,
            "device": self.device,
            "disable_pbar": True,
            "disable_log": True,
            "disable_update": True,
        }

    # ── 同步加载 ──────────────────────────────────────

    def _load_sensevoice(self) -> AutoModel:
        """加载 SenseVoiceSmall（离线 ASR + 情感 + 事件）"""
        if "sensevoice" not in self._models:
            logger.info("加载模型: SenseVoiceSmall ...")
            cfg = MODEL_CONFIGS["sensevoice"].copy()
            cfg.update(self._base_kwargs())
            self._models["sensevoice"] = AutoModel(**cfg)
            logger.info("SenseVoiceSmall 加载完成")
        return self._models["sensevoice"]

    def _load_streaming(self) -> AutoModel:
        """加载 paraformer-zh-streaming（流式 ASR）"""
        if "streaming" not in self._models:
            logger.info("加载模型: paraformer-zh-streaming ...")
            cfg = STREAMING_MODEL.copy()
            cfg.update(self._base_kwargs())
            self._models["streaming"] = AutoModel(**cfg)
            logger.info("paraformer-zh-streaming 加载完成")
        return self._models["streaming"]

    def _load_vad(self) -> AutoModel:
        """加载 fsmn-vad"""
        if "vad" not in self._models:
            logger.info("加载模型: fsmn-vad ...")
            cfg = VAD_MODEL.copy()
            cfg.update(self._base_kwargs())
            self._models["vad"] = AutoModel(**cfg)
            logger.info("fsmn-vad 加载完成")
        return self._models["vad"]

    def _load_punc(self) -> AutoModel:
        """加载 ct-punc（标点恢复）"""
        if "punc" not in self._models:
            logger.info("加载模型: ct-punc ...")
            cfg = PUNC_MODEL.copy()
            cfg.update(self._base_kwargs())
            self._models["punc"] = AutoModel(**cfg)
            logger.info("ct-punc 加载完成")
        return self._models["punc"]

    def _load_sv(self) -> AutoModel:
        """加载 cam++（说话人分离）"""
        if "sv" not in self._models:
            logger.info("加载模型: cam++ (speaker verification) ...")
            cfg = SV_MODEL.copy()
            cfg.update(self._base_kwargs())
            self._models["sv"] = AutoModel(**cfg)
            logger.info("cam++ 加载完成")
        return self._models["sv"]

    def _load_emotion(self) -> AutoModel:
        """加载 emotion2vec（独立情感识别，比 SenseVoice 自带更准）"""
        if "emotion" not in self._models:
            logger.info("加载模型: emotion2vec_plus_large ...")
            cfg = EMOTION_MODEL.copy()
            cfg.update(self._base_kwargs())
            self._models["emotion"] = AutoModel(**cfg)
            logger.info("emotion2vec 加载完成")
        return self._models["emotion"]

    def _load_funasr_nano(self) -> AutoModel:
        """加载 Fun-ASR-Nano（31 语言，LLM-based，自带标点）"""
        if "funasr_nano" not in self._models:
            logger.info("加载模型: Fun-ASR-Nano (31语言) ...")
            cfg = MODEL_CONFIGS["fun-asr-nano"].copy()
            cfg.update(self._base_kwargs())
            self._models["funasr_nano"] = AutoModel(**cfg)
            logger.info("Fun-ASR-Nano 加载完成")
        return self._models["funasr_nano"]

    # ── 获取模型（同步，需在启动时或线程池中调用）──────────

    def get(self, name: str) -> AutoModel:
        loaders = {
            "sensevoice": self._load_sensevoice,
            "streaming": self._load_streaming,
            "vad": self._load_vad,
            "punc": self._load_punc,
            "sv": self._load_sv,
            "emotion": self._load_emotion,
            "funasr_nano": self._load_funasr_nano,
        }
        if name not in loaders:
            raise ValueError(f"未知模型: {name}，可用: {list(loaders.keys())}")
        return loaders[name]()

    # ── 预加载 ────────────────────────────────────────

    def preload_all(self):
        """启动时预加载所有模型"""
        logger.info(f"开始预加载所有模型 (device={self.device}) ...")
        self.get("sensevoice")
        self.get("streaming")
        self.get("vad")
        self.get("punc")
        self.get("sv")
        self.get("emotion")
        logger.info("所有模型加载完成！")

    def preload_core(self):
        """只加载核心离线模型（SenseVoice + VAD）"""
        logger.info(f"预加载核心模型 (device={self.device}) ...")
        self.get("sensevoice")
        self.get("vad")
        self.get("emotion")
        logger.info("核心模型加载完成")

    # ── 状态查询 ───────────────────────────────────────

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
