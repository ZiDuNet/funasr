"""音频工具 — ffmpeg 转码、PCM 工具"""

import asyncio
import os
import uuid
import logging
import wave

logger = logging.getLogger(__name__)


async def convert_to_pcm(input_path: str, output_path: str | None = None) -> bytes:
    """使用 ffmpeg 将音频文件转为 16kHz 单声道 PCM"""
    cmd = [
        "ffmpeg", "-nostdin", "-i", input_path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000",
        "-loglevel", "error",
    ]
    if output_path:
        cmd.extend(["-y", output_path])
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 转码失败: {stderr.decode()}")
        with open(output_path, "rb") as f:
            return f.read()
    else:
        cmd.extend(["-"])
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 转码失败: {stderr.decode()}")
        return stdout


async def save_temp_upload(content: bytes, suffix: str, temp_dir: str = "/tmp/funasr") -> str:
    """保存上传文件到临时目录"""
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(content)
    return path


def pcm_duration_ms(pcm_bytes: bytes, fs: int = 16000, ch: int = 1, sampwidth: int = 2) -> int:
    """计算 PCM 时长（毫秒）"""
    if not pcm_bytes:
        return 0
    bytes_per_ms = (fs * ch * sampwidth) / 1000.0
    if bytes_per_ms <= 0:
        return 0
    return int(len(pcm_bytes) / bytes_per_ms)


def save_wav(out_path: str, audio_bytes: bytes, fs: int = 16000, ch: int = 1, sampwidth: int = 2):
    """保存 PCM 为 WAV"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sampwidth)
        wf.setframerate(fs)
        wf.writeframes(audio_bytes)
