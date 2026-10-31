"""Benchmark Demucs models: speed vs separation quality (SDR against ground truth).

    python scripts/bench_separation.py samples/two_speakers_music.mp4
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubber.stages.separate import separate_vocals  # noqa: E402

MODELS = ["htdemucs", "htdemucs_ft"]


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(reference), len(estimate))
    ref, est = reference[:n], estimate[:n]
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12) + 1e-12))


def main(video: Path) -> None:
    truth = video.with_name(video.stem + "_truth")
    speech, sr = sf.read(truth / "speech.wav", dtype="float32")
    music, _ = sf.read(truth / "music.wav", dtype="float32")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mix.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-ar", str(sr),
                        "-ac", "2", str(wav)], check=True)
        mix, _ = sf.read(wav, dtype="float32", always_2d=True)

    duration = len(mix) / sr
    print(f"{'model':<14}{'time':>8}{'x realtime':>12}{'vocal SDR':>12}{'music SDR':>12}")
    for name in MODELS:
        separate_vocals(mix[: sr * 5], sr, name)  # warm-up: model download + CUDA init
        t = time.perf_counter()
        vocals = separate_vocals(mix, sr, name)
        elapsed = time.perf_counter() - t
        v, b = vocals.mean(axis=1), (mix - vocals).mean(axis=1)
        print(f"{name:<14}{elapsed:>7.1f}s{duration / elapsed:>11.1f}x"
              f"{sdr(speech, v):>10.1f}dB{sdr(music, b):>10.1f}dB")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/two_speakers_music.mp4"))
