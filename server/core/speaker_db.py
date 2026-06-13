"""声纹数据库 — 多租户隔离，JSON 文件存储"""

import json
import os
import logging
import shutil
import numpy as np
from pathlib import Path
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


def create_group(group_id: str | None = None) -> str:
    """Create an empty speaker group and return its ID."""
    gid = group_id or generate_group_id()
    _save_db(gid, _load_db(gid))
    return gid


def remove_group(group_id: str) -> bool:
    """删除整个声纹组目录。"""
    if not group_id:
        return False
    root = Path(SPEAKERS_DIR).resolve()
    target = (root / group_id).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("非法声纹组 ID")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    logger.info(f"声纹组删除: group={group_id}")
    return True


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


def best_speaker_candidate(group_id: str, embedding: list | None) -> dict | None:
    """返回声纹组里最接近的候选人，不做阈值过滤。"""
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

    best_name, best_score = None, float("-inf")
    for name, ref in db.items():
        try:
            ref_vec = np.array(ref, dtype=np.float32).flatten()
            sim = 1.0 - cosine(vec, ref_vec)
            if not np.isfinite(sim):
                continue
            if sim > best_score:
                best_score = sim
                best_name = name
        except Exception:
            continue

    if best_name is None:
        return None
    return {"name": best_name, "score": round(float(best_score), 4)}


def match_speaker_detail(group_id: str, embedding: list | None,
                         threshold: float = 0.2) -> dict | None:
    """在 group 中匹配说话人，返回名称和分数。"""
    candidate = best_speaker_candidate(group_id, embedding)
    if candidate and candidate["score"] > threshold:
        return candidate
    return None


def match_speaker(group_id: str, embedding: list | None, threshold: float = 0.2) -> str | None:
    """在 group 中匹配说话人，返回匹配到的名字或 None。"""
    detail = match_speaker_detail(group_id, embedding, threshold=threshold)
    return detail["name"] if detail else None


def extract_embedding(model, audio_input) -> list | None:
    """从音频提取声纹 embedding（256维向量）"""
    try:
        out = model.generate(input=audio_input, embedding=True)
        embedding = out[0]["spk_embedding"][0].cpu().numpy()
        return embedding.tolist()
    except Exception as e:
        logger.error(f"提取声纹失败: {e}")
        return None


def _segment_speaker_key(seg: dict):
    spk_id = seg.get("speaker_id")
    if spk_id is None:
        spk_id = seg.get("spk")
    if spk_id is None:
        return None
    try:
        number = float(spk_id)
        if number.is_integer():
            return int(number)
    except (TypeError, ValueError):
        pass
    return str(spk_id)


def _slice_pcm_by_ms(pcm_bytes: bytes, start_ms, end_ms) -> bytes:
    bytes_per_ms = 32000 / 1000  # 16kHz, 16bit, mono
    try:
        start = max(0.0, float(start_ms or 0))
        end = max(0.0, float(end_ms or 0))
    except (TypeError, ValueError):
        return b""
    if end <= start:
        return b""
    start_byte = max(0, int(start * bytes_per_ms))
    end_byte = min(len(pcm_bytes), int(end * bytes_per_ms))
    if end_byte <= start_byte:
        return b""
    return pcm_bytes[start_byte:end_byte]


def match_segments(segments: list[dict], pcm_bytes: bytes,
                   group_id: str, sv_model, threshold: float = 0.2) -> dict:
    """对 segments 中的每个匿名说话人提取声纹并匹配数据库。

    匹配到的添加 ``speaker`` 字段（注册名），未匹配的保留数字 ``speaker_id``。
    兼容 FunASR 的 ``spk`` 和本项目标准化后的 ``speaker_id`` 字段。

    同一个匿名说话人的多段音频会先合并再提取声纹，避免第一段太短或
    静音过多导致整个说话人匹配失败。

    Args:
        segments: 转写结果中的 segments 列表（会被原地修改）
        pcm_bytes: 16kHz 16bit 单声道 PCM 原始音频
        group_id: 声纹组 ID
        sv_model: 声纹模型（AutoModel 实例）
    """
    summary = {
        "group_id": group_id,
        "threshold": threshold,
        "reference_speaker_count": len(_load_db(group_id)),
        "total_segments": len(segments),
        "matched_segment_count": 0,
        "matched_speaker_count": 0,
        "unmatched_speaker_count": 0,
        "speakers": {},
    }
    if not segments:
        return summary

    chunks_by_speaker: dict = {}
    segments_by_speaker: dict = {}

    for seg in segments:
        spk_id = _segment_speaker_key(seg)
        if spk_id is None:
            continue
        seg["speaker_id"] = spk_id
        segments_by_speaker.setdefault(spk_id, []).append(seg)
        seg_audio = _slice_pcm_by_ms(pcm_bytes, seg.get("start"), seg.get("end"))
        if seg_audio:
            chunks_by_speaker.setdefault(spk_id, []).append(seg_audio)

    for spk_id, speaker_segments in segments_by_speaker.items():
        chunks = chunks_by_speaker.get(spk_id) or []
        audio = b"".join(chunks)
        speaker_summary = {
            "audio_seconds": round(len(audio) / 32000, 3),
            "segment_count": len(speaker_segments),
            "matched": False,
            "name": None,
            "score": None,
            "best_candidate": None,
            "reason": "",
        }

        if summary["reference_speaker_count"] == 0:
            speaker_summary["reason"] = "声纹组为空"
        elif len(audio) < 1600:
            speaker_summary["reason"] = "可用于匹配的说话人音频太短"
        else:
            try:
                embedding = extract_embedding(sv_model, audio)
                if embedding is None:
                    speaker_summary["reason"] = "当前说话人音频提取声纹失败"
                    candidate = None
                else:
                    candidate = best_speaker_candidate(group_id, embedding)
                speaker_summary["best_candidate"] = candidate
                matched = candidate if candidate and candidate["score"] > threshold else None
                if matched:
                    speaker_summary.update({
                        "matched": True,
                        "name": matched["name"],
                        "score": matched["score"],
                        "reason": "已命中",
                    })
                    for seg in speaker_segments:
                        seg["speaker"] = matched["name"]
                        seg["speaker_score"] = matched["score"]
                elif candidate:
                    speaker_summary["reason"] = (
                        f"最高相似度 {candidate['score']} 未达到阈值 {threshold}"
                    )
                else:
                    speaker_summary["reason"] = "未找到可比较的声纹候选"
            except Exception as e:
                speaker_summary["reason"] = f"声纹匹配失败: {e}"
                logger.warning(f"声纹匹配失败 speaker_id={spk_id}: {e}")

        if speaker_summary["matched"]:
            summary["matched_speaker_count"] += 1
            summary["matched_segment_count"] += len(speaker_segments)
        else:
            summary["unmatched_speaker_count"] += 1
        summary["speakers"][str(spk_id)] = speaker_summary

    return summary


class SpeakerTracker:
    """流式跨段说话人追踪 — 维护全局 speaker_id 一致性

    解决流式 2pass/offline 模式中每段独立聚类导致的
    说话人编号跳跃问题（第一段的 speaker 0 可能不是
    第二段的 speaker 0）。

    在 WebSocket 连接生命周期内维护一个全局 embedding 库，
    每段新 embedding 先和已知说话人比对（余弦相似度），
    匹配上则继承已有 ID，否则分配新 ID。
    """

    def __init__(self, sv_model, threshold: float = 0.6):
        self.next_id = 0
        self.centroids: dict[int, np.ndarray] = {}  # global_id → centroid
        self.threshold = threshold
        self.sv_model = sv_model

    def track(self, segments: list[dict], pcm_bytes: bytes) -> list[dict]:
        """重新分配一致的 speaker_id

        Args:
            segments: 单段离线结果中的 sentence_info 列表
                      (含 spk, start, end 字段; start/end 毫秒)
            pcm_bytes: 当前 VAD 段的 16kHz 16bit PCM 原始音频

        Returns:
            修改后的 segments，speaker_id 为全局一致编号
        """
        if not segments:
            return segments

        from scipy.spatial.distance import cosine

        bytes_per_ms = 32000 / 1000
        # 当前段内的本地 spk → 新 embedding
        local_embeddings: dict[int, np.ndarray] = {}

        for seg in segments:
            spk = seg.get("spk")
            if spk is None:
                continue
            if spk in local_embeddings:
                continue

            start_byte = int(seg.get("start", 0) * bytes_per_ms)
            end_byte = int(seg.get("end", 0) * bytes_per_ms)
            seg_audio = pcm_bytes[start_byte:end_byte]

            if len(seg_audio) < 1600:
                continue

            try:
                emb = extract_embedding(self.sv_model, seg_audio)
                if emb is not None:
                    local_embeddings[spk] = np.array(emb, dtype=np.float32)
            except Exception as e:
                logger.warning(f"SpeakerTracker 提取声纹失败: {e}")

        # 对每个本地 spk，匹配全局已知说话人或分配新 ID
        spk_remap: dict[int, int] = {}
        for local_spk, emb in local_embeddings.items():
            best_id, best_sim = None, self.threshold
            for gid, centroid in self.centroids.items():
                sim = 1.0 - cosine(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_id = gid

            if best_id is not None:
                # 匹配到已知说话人：更新 centroid
                self.centroids[best_id] = (
                    self.centroids[best_id] * 0.7 + emb * 0.3
                )
                spk_remap[local_spk] = best_id
            else:
                # 新说话人
                gid = self.next_id
                self.next_id += 1
                self.centroids[gid] = emb
                spk_remap[local_spk] = gid

        # 重映射
        for seg in segments:
            old_spk = seg.get("spk")
            if old_spk is not None and old_spk in spk_remap:
                seg["spk"] = spk_remap[old_spk]

        return segments
