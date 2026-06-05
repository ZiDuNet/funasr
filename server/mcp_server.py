"""MCP 服务 — 使用 FastMCP 暴露语音识别工具"""

import os
import logging

from fastmcp import FastMCP

from server.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

mcp = FastMCP("funasr", instructions="FunASR 语音识别服务，支持 50+ 语言")


@mcp.tool
def transcribe_audio(audio_path: str, language: str = "auto") -> str:
    """转写音频文件为文本。支持 50+ 语言，自动检测语言。

    Args:
        audio_path: 音频文件路径（wav/mp3/flac/m4a 等）
        language: 语言提示（auto/zh/en/ja/ko/yue），默认自动检测
    """
    if not os.path.exists(audio_path):
        return f"错误：文件不存在: {audio_path}"

    import re
    registry = ModelRegistry.get_instance()
    model = registry.get("sensevoice")

    result = model.generate(input=audio_path, batch_size=1, language=language)
    if not result:
        return "转写结果为空"

    text = re.sub(r"<\|[^|]*\|>", "", result[0].get("text", "")).strip()

    output = f"转写结果: {text}"
    if "sentence_info" in result[0]:
        output += "\n\n分段信息:"
        for seg in result[0]["sentence_info"]:
            seg_text = re.sub(r"<\|[^|]*\|>", "", seg.get("text", "")).strip()
            spk = f" [说话人 {seg.get('spk')}]" if "spk" in seg else ""
            output += f"\n  [{seg.get('start', 0)/1000:.1f}s - {seg.get('end', 0)/1000:.1f}s]{spk} {seg_text}"

    return output


@mcp.tool
def transcribe_url(url: str, language: str = "auto") -> str:
    """下载远程音频文件并转写为文本。

    Args:
        url: 远程音频文件 URL
        language: 语言提示，默认自动检测
    """
    import re
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
        return transcribe_audio(tmp_path, language)
    finally:
        os.unlink(tmp_path)


def get_mcp_app():
    """获取 MCP HTTP ASGI 应用"""
    return mcp.http_app(path="/")
