"""声纹数据库 — 多租户隔离，JSON 文件存储"""

import json
import os
import logging
import numpy as np
from uuid import uuid4

from server.models.config import SPEAKERS_DIR

logger = logging.getLogger(__name__)


def _ensure_group_dir(group_id: str) -> str:
    d = os.path.join(SPEAKERS_DIR, group_id)
    os.makedirs(d, exist_ok=True)
    return d


def _db_path(group_id: str) -> str:
    return os.path.join(_ensure_group_dir(group_id), "db.json")


def _load_db(group_id: str) -> dict:
    """加载某个 group 的声纹库"""
    path = _db_path(group_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_db(group_id: str, data: dict):
    """保存声纹库"""
    path = _db_path(group_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def generate_group_id() -> str:
    """生成新的租户 group ID"""
    return f"grp_{uuid4().hex[:12]}"


def register_speaker(group_id: str, name: str, embedding: list) -> str:
    """注册说话人（如 group 不存在则自动创建）

    返回 group_id
    """
    db = _load_db(group_id)
    db[name] = embedding
    _save_db(group_id, db)
    logger.info(f"声纹注册: group={group_id}, name={name}")
    return group_id


def remove_speaker(group_id: str, name: str) -> bool:
    """删除某个说话人"""
    db = _load_db(group_id)
    if name not in db:
        return False
    del db[name]
    _save_db(group_id, db)
    return True


def list_speakers(group_id: str) -> list[str]:
    """列出某 group 的所有说话人"""
    db = _load_db(group_id)
    return list(db.keys())


def list_groups() -> list[dict]:
    """列出所有 group"""
    if not os.path.isdir(SPEAKERS_DIR):
        return []
    groups = []
    for gid in os.listdir(SPEAKERS_DIR):
        gpath = os.path.join(SPEAKERS_DIR, gid)
        if os.path.isdir(gpath):
            speakers = list_speakers(gid)
            groups.append({"group_id": gid, "speaker_count": len(speakers), "speakers": speakers})
    return groups


def match_speaker(group_id: str, embedding: list | None, threshold: float = 0.2) -> str | None:
    """在 group 中匹配说话人

    返回匹配到的说话人名字，或 None
    """
    if embedding is None:
        return None
    db = _load_db(group_id)
    if not db:
        return None

    try:
        vec = np.array(embedding, dtype=np.float32).flatten()
    except Exception:
        return None

    from scipy.spatial.distance import cosine

    best_name, best_score = None, 0.0
    for name, ref in db.items():
        try:
            ref_vec = np.array(ref, dtype=np.float32)
            sim = 1.0 - cosine(vec, ref_vec)
            if sim > best_score and sim > threshold:
                best_score = sim
                best_name = name
        except Exception:
            continue

    return best_name


def extract_embedding(model, audio_input) -> list | None:
    """从音频提取声纹 embedding（256维向量）"""
    try:
        out = model.generate(input=audio_input, embedding=True)
        embedding = out[0]["spk_embedding"][0].cpu().numpy()
        return embedding.tolist()
    except Exception as e:
        logger.error(f"提取声纹失败: {e}")
        return None
