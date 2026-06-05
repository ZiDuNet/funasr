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


def clean_text(text: str) -> str:
    """去掉所有 <|...|> 标签，返回纯文本"""
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def extract_emotion(text: str) -> str | None:
    """从 SenseVoice 输出中提取情感标签，返回原始标签名（如 HAPPY、SAD）"""
    for tag, emoji in EMO_DICT.items():
        if tag in text and emoji:  # emoji 非空表示有效情感（排除 NEUTRAL）
            return tag.strip("<|").strip("|>")
    return None


def extract_events(text: str) -> list[str]:
    """从 SenseVoice 输出中提取事件标签，返回原始标签名列表（如 BGM、Applause）"""
    events = []
    for tag, emoji in EVENT_DICT.items():
        if tag in text and emoji:  # emoji 非空表示有效事件（排除 Speech/Breath）
            events.append(tag.strip("<|").strip("|>"))
    return events


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
