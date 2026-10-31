"""Text-to-speech providers.

Both providers return mono float32 audio at ``SR`` with leading/trailing silence
trimmed, so measured durations are the real speaking time used for sync.
Results are cached by content hash: re-running after editing one line only
re-synthesizes that line.
"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from dubber import llm
from dubber.errors import DubberError, TransientError
from dubber.pricing import gemini_tts_cost

SR = 24000  # Gemini TTS native rate; Edge output is resampled to match


@dataclass
class Speech:
    path: Path
    duration: float
    cost_usd: float = 0.0
    cached: bool = False


def trim_silence(audio: np.ndarray, threshold_db: float = -42.0, pad_s: float = 0.04,
                 sr: int = SR) -> np.ndarray:
    if audio.size == 0:
        return audio
    frame = int(0.01 * sr)
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms) > threshold_db)[0]
    if loud.size == 0:
        return audio
    pad = int(pad_s * sr)
    return audio[max(0, loud[0] * frame - pad): min(len(audio), (loud[-1] + 1) * frame + pad)]


class TTSProvider:
    name: str

    def synthesize(self, text: str, voice: str, emotion: str, style: str,
                   language: str, cache_dir: Path) -> Speech:
        key = hashlib.sha1(f"{self.name}|{voice}|{emotion}|{style}|{language}|{text}"
                           .encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"{self.name}_{key}.wav"
        if path.exists():
            return Speech(path, sf.info(path).duration, cached=True)
        audio, cost = self._generate(text, voice, emotion, style, language)
        audio = trim_silence(audio)
        if audio.size < SR * 0.1:
            raise TransientError(f"{self.name} returned no speech for: {text!r}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, SR, subtype="PCM_16")
        return Speech(path, len(audio) / SR, cost)

    def _generate(self, text, voice, emotion, style, language) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class GeminiTTS(TTSProvider):
    """Gemini native TTS: controllable delivery via natural-language instructions."""

    name = "gemini"
    model = "gemini-2.5-flash-preview-tts"

    @staticmethod
    def prompt(text: str, emotion: str, style: str) -> str:
        how = "naturally, in a conversational tone" if emotion == "neutral" else \
            f"with a {emotion} tone"
        if style:
            how += f" ({style})"
        return f"Say {how}:\n{text}"

    def _generate(self, text, voice, emotion, style, language):
        from google.genai import types

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        )
        for attempt in range(4):  # absorb short per-minute rate limits locally
            try:
                resp = llm.client().models.generate_content(
                    model=self.model, contents=self.prompt(text, emotion, style), config=config)
                break
            except Exception as exc:  # noqa: BLE001
                err = llm.classify(exc)
                if not isinstance(err, TransientError) or attempt == 3:
                    raise err from exc
                time.sleep(8 * (attempt + 1))
        try:
            pcm = resp.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError) as exc:
            raise TransientError("Gemini TTS response contained no audio") from exc
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        tokens = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
        return audio, gemini_tts_cost(self.model, tokens or 0, len(audio) / SR)


class EdgeTTS(TTSProvider):
    """Microsoft Edge neural voices: free, fast, no key; no emotion control."""

    name = "edge"

    def _generate(self, text, voice, emotion, style, language):
        import edge_tts

        async def fetch() -> bytes:
            chunks = []
            async for msg in edge_tts.Communicate(text, voice).stream():
                if msg["type"] == "audio":
                    chunks.append(msg["data"])
            return b"".join(chunks)

        try:
            mp3 = asyncio.run(fetch())
        except Exception as exc:  # noqa: BLE001 - network/service errors are transient
            raise TransientError(f"edge-tts failed: {exc}") from exc
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
             "-ac", "1", "-ar", str(SR), "pipe:1"], input=mp3, capture_output=True)
        if proc.returncode != 0:
            raise DubberError(f"could not decode edge-tts audio: {proc.stderr[-500:]!r}")
        return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0, 0.0


PROVIDERS: dict[str, type[TTSProvider]] = {"gemini": GeminiTTS, "edge": EdgeTTS}


def get_provider(name: str) -> TTSProvider:
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise DubberError(f"unknown TTS provider {name!r}; choose from {list(PROVIDERS)}")
