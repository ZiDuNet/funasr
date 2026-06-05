"""MCP 服务 — 使用 FastMCP 暴露语音识别工具"""

import os
import re
import logging

from fastmcp import FastMCP

from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

mcp = FastMCP("funasr", instructions="FunASR 语音识别服务，支持 50+ 语言。可指定语言和模型。")


def _do_transcribe(audio_path: str, language: str = "auto",
                   speaker_diarization: bool = False) -> str:
    """内部统一的转写函数"""
    registry = ModelRegistry.get_instance()
    model = registry.get(with_spk=speaker_diarization)

    result = model.generate(
        input=audio_path,
        batch_size_s=300,
        language=language,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
        batch_size_threshold_s=60,
    )
    if not result:
        return "转写结果为空"

    raw = result[0]
    text = re.sub(r"<\|[^|]*\|>", "", raw.get("text", "")).strip()
    if not text:
        return "未检测到语音内容"

    output = f"转写结果:\n{text}"
    if "sentence_info" in raw:
        output += "\n\n分段信息:"
        for seg in raw["sentence_info"]:
            seg_text = re.sub(r"<\|[^|]*\|>", "", seg.get("text", "")).strip()
            ts = f"{seg.get('start', 0)/1000:.1f}s-{seg.get('end', 0)/1000:.1f}s"
            spk = ""
            if "spk" in seg:
                spk = f" [说话人{seg['spk']}]"
            output += f"\n  [{ts}]{spk} {seg_text}"
    return output


@mcp.tool
def transcribe_audio(audio_path: str, language: str = "auto",
                     speaker_diarization: bool = False) -> str:
    """转写音频文件为文本。支持 5-31 语言（视部署模型而定）。

    Args:
        audio_path: 音频文件路径（wav/mp3/flac/m4a 等）
        language: 语言提示（auto/zh/en/ja/ko/yue），默认自动检测
        speaker_diarization: 是否启用说话人分离，默认关闭
    """
    if not os.path.exists(audio_path):
        return f"错误：文件不存在: {audio_path}"
    return _do_transcribe(audio_path, language=language,
                          speaker_diarization=speaker_diarization)


@mcp.tool
def transcribe_url(url: str, language: str = "auto",
                   speaker_diarization: bool = False) -> str:
    """下载远程音频文件并转写为文本。

    Args:
        url: 远程音频文件 URL
        language: 语言提示，默认自动检测
        speaker_diarization: 是否启用说话人分离
    """
    import tempfile
    import httpx

    try:
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
    except Exception as e:
        return f"下载失败: {e}"

    suffix = os.path.splitext(url.split("?")[0])[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        return _do_transcribe(tmp_path, language=language,
                              speaker_diarization=speaker_diarization)
    finally:
        os.unlink(tmp_path)


def get_mcp_app():
    return mcp.http_app(path="/")
