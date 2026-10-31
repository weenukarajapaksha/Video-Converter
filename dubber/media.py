"""Thin, typed wrappers around ffprobe / ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from dubber.errors import InputValidationError, MediaError
from dubber.schemas import VideoMetadata

MP4_FAMILY = "mp4"  # ffprobe reports MP4/MOV as "mov,mp4,m4a,3gp,3g2,mj2"


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise MediaError(f"{binary} not found on PATH - install FFmpeg")
    return path


def run_ffmpeg(*args: str | Path) -> None:
    cmd = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")


def _ffprobe_json(path: Path) -> dict:
    cmd = [_require("ffprobe"), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        raise InputValidationError(f"not a readable media file: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = stream.get(key, "0/0")
        num, _, den = rate.partition("/")
        if den and int(den) != 0 and int(num) != 0:
            return round(float(Fraction(int(num), int(den))), 3)
    raise InputValidationError("could not determine frame rate")


def _float(value) -> float | None:
    return float(value) if value not in (None, "N/A") else None


def _int(value) -> int | None:
    return int(value) if value not in (None, "N/A") else None


def probe(path: Path) -> VideoMetadata:
    data = _ffprobe_json(Path(path))
    fmt = data["format"]
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if video is None:
        raise InputValidationError("file has no video stream")

    container_duration = _float(fmt.get("duration")) or 0.0
    return VideoMetadata(
        container=fmt["format_name"],
        duration_s=round(container_duration, 3),
        video_duration_s=round(_float(video.get("duration")) or container_duration, 3),
        audio_duration_s=round(_float(audio.get("duration")) or container_duration, 3)
        if audio else None,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=_fps(video),
        video_codec=video["codec_name"],
        pix_fmt=video.get("pix_fmt"),
        video_bitrate=_int(video.get("bit_rate")),
        audio_codec=audio["codec_name"] if audio else None,
        audio_sample_rate=_int(audio.get("sample_rate")) if audio else None,
        audio_channels=_int(audio.get("channels")) if audio else None,
        size_bytes=int(fmt.get("size", Path(path).stat().st_size)),
    )


def validate(meta: VideoMetadata, min_s: float, max_s: float) -> None:
    if MP4_FAMILY not in meta.container.split(","):
        raise InputValidationError(f"expected an MP4 file, got container '{meta.container}'")
    if not meta.has_audio:
        raise InputValidationError("video has no audio track - nothing to dub")
    if not (min_s <= meta.duration_s <= max_s):
        raise InputValidationError(
            f"duration {meta.duration_s:.1f}s is outside the supported range "
            f"{min_s:.0f}-{max_s:.0f}s"
        )


def time_stretch(src: Path, dst: Path, tempo: float, sample_rate: int) -> Path:
    """Change speed without changing pitch (Rubber Band, formant-preserving).

    ``tempo`` > 1 speeds up. Output is resampled to ``sample_rate`` mono PCM.
    """
    filters = [] if abs(tempo - 1.0) < 1e-3 else [
        f"rubberband=tempo={tempo:.4f}:pitchq=quality:formant=preserved"]
    filters.append(f"aresample={sample_rate}")
    run_ffmpeg("-i", src, "-af", ",".join(filters), "-ac", "1", "-c:a", "pcm_s16le", dst)
    return dst


def extract_audio(src: Path, dst: Path, sample_rate: int | None = None,
                  channels: int | None = None) -> Path:
    """Decode the first audio stream to 16-bit PCM WAV (lossless intermediate)."""
    args: list[str | Path] = ["-i", src, "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if channels:
        args += ["-ac", str(channels)]
    run_ffmpeg(*args, dst)
    return dst
