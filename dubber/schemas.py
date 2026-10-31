"""Data contracts passed between pipeline stages (persisted as JSON artifacts)."""

from __future__ import annotations

from pydantic import BaseModel


class VideoMetadata(BaseModel):
    container: str
    duration_s: float  # container duration
    video_duration_s: float
    audio_duration_s: float | None
    width: int
    height: int
    fps: float
    video_codec: str
    pix_fmt: str | None
    video_bitrate: int | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    size_bytes: int

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"
