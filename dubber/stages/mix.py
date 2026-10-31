"""Task 5b: mix English dialogue onto the music/SFX stem.

* The dub is gain-matched to the loudness of the original Chinese dialogue, so
  the voice/music balance of the source is preserved.
* The background is gently ducked (-4 dB, 80 ms ramps) while English is spoken,
  which masks any residual Chinese bleed left by source separation.
* The final mix is loudness-normalized to the original soundtrack's level.
"""

from __future__ import annotations

import json
import re
import subprocess

import numpy as np
import soundfile as sf

from dubber import media
from dubber.artifacts import (BACKGROUND, DUB_MIX, DUB_TRACK, DUB_VOICE, ORIGINAL_AUDIO,
                              TRANSCRIPT, VOCALS)
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import DubTrack, Transcript

DUCK_DB = -4.0
RAMP_S = 0.08


def active_rms(audio: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> float:
    parts = [audio[int(s * sr):int(e * sr)] for s, e in spans]
    parts = [p for p in parts if p.size]
    if not parts:
        return 0.0
    cat = np.concatenate(parts)
    return float(np.sqrt(np.mean(cat ** 2)))


def duck_envelope(n: int, sr: int, spans: list[tuple[float, float]], db: float = DUCK_DB,
                  ramp_s: float = RAMP_S) -> np.ndarray:
    """Gain curve: 1.0 normally, ``db`` during spans, with linear ramps at the edges."""
    target = np.zeros(n, dtype=np.float32)
    for s, e in spans:
        target[max(0, int((s - ramp_s) * sr)):min(n, int((e + ramp_s) * sr))] = 1.0
    ramp = max(1, int(ramp_s * sr))
    kernel = np.ones(ramp, dtype=np.float32) / ramp
    depth = np.convolve(target, kernel, mode="same")
    return (1.0 - depth * (1.0 - 10 ** (db / 20))).astype(np.float32)


def integrated_loudness(path) -> float | None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
         "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr)
    if not match:
        return None
    value = float(json.loads(match.group(0))["input_i"])
    return value if np.isfinite(value) and value > -70 else None


class MixStage(Stage):
    name = "mix"
    outputs = (DUB_MIX,)
    weight = 0.5

    def run(self, job: Job, progress: ProgressFn) -> None:
        background, sr = sf.read(job.path(BACKGROUND), dtype="float32", always_2d=True)
        voice, vsr = sf.read(job.path(DUB_VOICE), dtype="float32")
        vocals, _ = sf.read(job.path(VOCALS), dtype="float32", always_2d=True)
        assert vsr == sr, f"sample rate mismatch {vsr} != {sr}"

        n = len(background)
        voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]
        track = job.read_model(DUB_TRACK, DubTrack)
        dub_spans = [(ln.start, ln.start + ln.final_duration_s) for ln in track.lines]
        src_spans = [(s.start, s.end) for s in job.read_model(TRANSCRIPT, Transcript).segments]

        progress(0.2, "matching dialogue level")
        src_rms = active_rms(vocals.mean(axis=1), sr, src_spans)
        dub_rms = active_rms(voice, sr, dub_spans)
        gain = float(np.clip(src_rms / dub_rms, 0.25, 4.0)) if src_rms and dub_rms else 1.0

        env = duck_envelope(n, sr, dub_spans)
        mix = background * env[:, None] + (voice * gain)[:, None]
        raw = job.path("dub_mix_raw.wav")
        sf.write(raw, mix, sr, subtype="FLOAT")

        progress(0.6, "loudness normalization")
        target = integrated_loudness(job.path(ORIGINAL_AUDIO)) or -16.0
        target = float(np.clip(target, -30.0, -9.0))
        media.run_ffmpeg("-i", raw, "-af",
                         f"loudnorm=I={target:.1f}:TP=-1.5:LRA=11,aresample={sr}",
                         "-c:a", "pcm_s16le", job.path(DUB_MIX))
        raw.unlink(missing_ok=True)
        progress(1.0, f"voice gain {20 * np.log10(gain):+.1f} dB, target {target:.1f} LUFS")
