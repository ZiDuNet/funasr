"""后处理工具 — 情感/事件标签、文本清洗"""

import re

# ── 情感标签 ────────────────────────────────────────
EMO_DICT = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

# ── 事件标签 ────────────────────────────────────────
EVENT_DICT = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
}

# ── 语言标签 ────────────────────────────────────────
LANG_DICT = {
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
}

EMOJI_DICT = {
    "<|nospeech|><|Event_UNK|>": "❓",
    **{k: "" for k in LANG_DICT},
    **EMO_DICT,
    **EVENT_DICT,
    "<|EMO_UNKNOWN|>": "",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

EMO_SET = {"😊", "😔", "😡", "😰", "🤢", "😮"}
EVENT_SET = {"🎼", "👏", "😀", "😭", "🤧", "😷"}

CONTROL_TAGS = {
    "zh", "en", "yue", "ja", "ko", "nospeech",
    "happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised",
    "bgm", "speech", "applause", "laughter", "cry", "sneeze", "breath", "cough",
    "emo_unknown", "sing", "speech_noise", "withitn", "woitn", "gbg", "event_unk",
}

CONTROL_TAG_RE = re.compile(r"<\s*\|?\s*([^<>]*?)\s*\|?\s*>")
LANG_TAGS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
EMOTION_TAGS = {
    "happy": "HAPPY",
    "sad": "SAD",
    "angry": "ANGRY",
    "neutral": "NEUTRAL",
    "fearful": "FEARFUL",
    "disgusted": "DISGUSTED",
    "surprised": "SURPRISED",
    "emo_unknown": "EMO_UNKNOWN",
}
EVENT_TAGS = {
    "bgm": "BGM",
    "speech": "Speech",
    "applause": "Applause",
    "laughter": "Laughter",
    "cry": "Cry",
    "sneeze": "Sneeze",
    "breath": "Breath",
    "cough": "Cough",
    "sing": "Sing",
    "speech_noise": "Speech_Noise",
    "event_unk": "Event_UNK",
}


def _normalize_tag_name(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _strip_control_tag(match: re.Match) -> str:
    return "" if _normalize_tag_name(match.group(1)) in CONTROL_TAGS else match.group(0)


def _canonical_tag_names(text: str) -> set[str]:
    names = set()
    for match in CONTROL_TAG_RE.finditer(text or ""):
        name = _normalize_tag_name(match.group(1))
        if name in CONTROL_TAGS:
            names.add(name)
    return names


def extract_metadata(text: str) -> dict:
    """把模型富文本控制标签拆成统一 JSON 元信息。"""
    names = _canonical_tag_names(text)
    metadata: dict[str, object] = {}

    for name in names:
        if name in LANG_TAGS:
            metadata["language"] = name
            break

    for name, label in EMOTION_TAGS.items():
        if name in names:
            metadata["emotion"] = label
            break

    events = []
    for name, label in EVENT_TAGS.items():
        if name in names:
            events.append(label)
    if events:
        metadata["events"] = events

    if "withitn" in names:
        metadata["itn"] = True
    elif "woitn" in names:
        metadata["itn"] = False

    if names:
        metadata["raw_tags"] = sorted(names)
    return metadata


def clean_text(text: str) -> str:
    """去掉 SenseVoice 控制标签，返回纯文本。

    FunASR/SenseVoice 常见输出是 ``<|zh|>``，但带说话人分离的路径有时会把
    控制 token 拆成 ``< | zh | >`` 或 ``< | S pe ech | >``。这里按已知控制
    标签白名单清理，避免误删普通尖括号内容。
    """
    text = CONTROL_TAG_RE.sub(_strip_control_tag, str(text or ""))
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    return text.strip()


def extract_emotion(text: str) -> str | None:
    """从 SenseVoice 输出中提取情感标签，返回原始标签名（如 HAPPY、SAD、NEUTRAL）"""
    return extract_metadata(text).get("emotion")


def extract_events(text: str) -> list[str]:
    """从 SenseVoice 输出中提取事件标签，返回原始标签名列表（如 BGM、Applause）"""
    return extract_metadata(text).get("events", [])


def rich_transcription_postprocess(text: str) -> str:
    """SenseVoice 富文本后处理（简化版）

    保留情感和事件 emoji，去掉语言标签和控制标签。
    """
    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")

    # 统计各特殊标签出现次数
    sptk_dict = {}
    for sptk in EMOJI_DICT:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")

    # 找出主要情感
    emo = "<|NEUTRAL|>"
    for e in EMO_DICT:
        tag_count = sptk_dict.get(e, 0)
        emo_count = sptk_dict.get(emo, 0)
        if tag_count > emo_count:
            emo = e

    # 构建事件 emoji 前缀
    event_prefix = ""
    for e in EVENT_DICT:
        if sptk_dict.get(e, 0) > 0 and EVENT_DICT[e]:
            event_prefix += EVENT_DICT[e]

    # 构建情感 emoji
    emo_emoji = EMO_DICT.get(emo, "")

    text = text.strip()
    if event_prefix:
        text = event_prefix + text
    if emo_emoji:
        text = text + emo_emoji

    return text
