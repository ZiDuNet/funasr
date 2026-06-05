"""模型注册表 — 配置文件中选模型，API 不暴露"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from funasr import AutoModel

from server.models.config import (
    DEVICE, NGPU, NCPU, WORKER_THREADS, PRELOAD_ALL, ENABLE_STREAMING,
    CONCURRENT_VAD, CONCURRENT_ASR_ONLINE, CONCURRENT_ASR_OFFLINE,
    CONCURRENT_PUNC, CONCURRENT_SV, CONCURRENT_EMOTION,
    ASR_CONFIG, ASR_CONFIG_WITH_SPK, STREAMING_CONFIG,
    VAD_CONFIG, PUNC_CONFIG, SV_CONFIG, EMOTION_CONFIG,
    MODEL_NAME, MODEL_CACHE,
)

logger = logging.getLogger(__name__)


def _model_id(config: dict) -> str:
    """从配置中提取模型 ID（用于缓存检测）"""
    return config.get("model", "")


def _is_cached(model_id: str) -> bool:
    """检查模型是否已存在于本地缓存"""
    if not model_id:
        return False
    # ModelScope 缓存结构: ~/.cache/modelscope/models/<org>/<model>/
    # 或 hub 子目录
    cache_dir = os.path.join(MODEL_CACHE, "models")
    if not os.path.isdir(cache_dir):
        return False
    # 尝试匹配 "org/model" 格式
    parts = model_id.split("/")
    if len(parts) >= 2:
        # 检查 iic/SenseVoiceSmall → models/iic/SenseVoiceSmall
        # 检查 Qwen/Qwen3-ASR-1.7B → models/Qwen/Qwen3-ASR-1.7B
        target = os.path.join(cache_dir, *parts)
        if os.path.isdir(target):
            return True
        # 也可能在小写目录下
        target_lower = os.path.join(cache_dir, parts[0].lower(), parts[1])
        if os.path.isdir(target_lower):
            return True
    # 单段名称（如 "paraformer-zh"、"fsmn-vad"）
    for org_dir in os.listdir(cache_dir):
        org_path = os.path.join(cache_dir, org_dir)
        if os.path.isdir(org_path):
            candidate = os.path.join(org_path, model_id)
            if os.path.isdir(candidate):
                return True
    return False


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
            model_id = _model_id(config)
            cached = _is_cached(model_id)
            status = "从缓存加载" if cached else "下载中（首次使用）"
            logger.info(f"加载模型: {key} [{model_id}] — {status}")
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
        """启动时加载模型（已缓存的秒加载，未缓存的自动下载）"""
        # 先报告缓存状态
        all_configs = [
            ("离线ASR", ASR_CONFIG),
            ("流式ASR", STREAMING_CONFIG),
            ("VAD", VAD_CONFIG),
            ("标点", PUNC_CONFIG),
            ("声纹", SV_CONFIG),
            ("情感", EMOTION_CONFIG),
        ]
        logger.info(f"预加载模型 (device={DEVICE}, model={MODEL_NAME})")
        logger.info("缓存检测:")
        for label, cfg in all_configs:
            mid = _model_id(cfg)
            cached = "✅ 已缓存" if _is_cached(mid) else "⬇️ 需下载"
            logger.info(f"  {label}: {mid} — {cached}")

        # 加载
        self.get()                          # 离线 ASR
        self.get(with_spk=True)             # 离线 ASR + 说话人
        if ENABLE_STREAMING:
            self.get_streaming()            # 流式 ASR
        self.get_aux("vad")                 # VAD
        self.get_aux("punc")                # 标点
        self.get_aux("sv")                  # 声纹
        self.get_aux("emotion")             # 情感
        logger.info(f"所有模型加载完成！共 {len(self._models)} 个")

    # ── 状态 ────────────────────────────────────

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
