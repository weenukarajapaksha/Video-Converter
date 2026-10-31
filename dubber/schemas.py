"""Data contracts passed between pipeline stages (persisted as JSON artifacts)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class Segment(BaseModel):
    """One utterance of the source dialogue (transcript.json)."""

    id: int
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    text: str
    words: list[Word] = Field(default_factory=list)
    confidence: float | None = None  # mean word probability

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    language: str = "zh"
    segments: list[Segment]
    speech_ratio: float | None = None  # fraction of the timeline that contains speech

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})


Gender = Literal["male", "female", "unknown"]


class SpeakerProfile(BaseModel):
    """Voice assignment for one diarized speaker (voices.json). Editable in the UI."""

    speaker: str
    gender: Gender = "unknown"
    median_f0_hz: float | None = None
    speaking_time_s: float = 0.0
    voice: str  # provider-specific voice id, e.g. "Puck" or "en-US-GuyNeural"
    style: str = ""  # free-text delivery hint for style-capable TTS


class VoiceMap(BaseModel):
    provider: str
    speakers: dict[str, SpeakerProfile]


class TranslatedSegment(BaseModel):
    """One line of dubbed dialogue (translation.json). Editable before dubbing."""

    id: int
    start: float
    end: float
    speaker: str
    source_text: str
    text: str
    emotion: str = "neutral"
    # Time the line may occupy: its own slot plus any silence before the next line.
    max_duration_s: float


class TranslationIssue(BaseModel):
    id: int
    category: str  # accuracy | idiom | tone | fluency | consistency | length
    comment: str
    before: str
    after: str


class TranslationReview(BaseModel):
    """Output of the critic pass (LLM-as-judge, 1-10 scales)."""

    accuracy: int
    naturalness: int
    tone: int
    summary: str
    issues: list[TranslationIssue] = Field(default_factory=list)


class Translation(BaseModel):
    source_language: str = "Chinese"
    target_language: str = "English"
    model: str
    segments: list[TranslatedSegment]
    review: TranslationReview | None = None


class SynthesizedLine(BaseModel):
    id: int
    path: str  # relative to the job dir
    start: float  # placement on the output timeline
    raw_duration_s: float
    final_duration_s: float
    tempo: float = 1.0  # time-stretch factor applied (1.0 = none)
    rewrites: int = 0  # how many times the line was shortened by the LLM
    text: str


class DubTrack(BaseModel):
    lines: list[SynthesizedLine]
