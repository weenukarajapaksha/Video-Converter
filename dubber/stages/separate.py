"""Split the soundtrack into a speech stem and a music/SFX stem with Demucs.

The vocal stem feeds ASR + diarization (much more robust under background music),
and the background stem is what the English dialogue is mixed onto, so the
original Chinese voices disappear while music and effects are preserved.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import soundfile as sf

from dubber import ml
from dubber.artifacts import BACKGROUND, ORIGINAL_AUDIO, VOCALS
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage

CHUNK_S = 60  # process long audio in chunks to report progress and cap memory


def separate_vocals(mix: np.ndarray, sr: int, model_name: str,
                    progress: Callable[[float, str], None] = lambda f, m: None) -> np.ndarray:
    """Return the vocal stem for ``mix`` of shape (samples, channels)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    x = torch.from_numpy(np.ascontiguousarray(mix.T))  # (channels, samples)
    with ml.gpu_scope():
        model = get_model(model_name)
        model.eval()
        dev = ml.device()
        vocal_idx = model.sources.index("vocals")

        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        normed = (x - mean) / std

        step = CHUNK_S * sr
        vocals = torch.zeros_like(x)
        for lo in range(0, x.shape[1], step):
            hi = min(lo + step, x.shape[1])
            with torch.inference_mode():
                est = apply_model(model, normed[None, :, lo:hi], device=dev, shifts=0,
                                  split=True, overlap=0.25, progress=False)[0]
            vocals[:, lo:hi] = est[vocal_idx].cpu() * std + mean
            progress(hi / x.shape[1], f"separated {hi / sr:.0f}s")
        del model
    return vocals.numpy().T


class SeparateStage(Stage):
    name = "separate"
    outputs = (VOCALS, BACKGROUND)
    weight = 2.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        model_name = job.option("separation_model", get_settings().separation_model)
        mix, sr = sf.read(job.path(ORIGINAL_AUDIO), dtype="float32", always_2d=True)
        progress(0.02, f"loading {model_name}")
        vocals = separate_vocals(mix, sr, model_name,
                                 lambda f, m: progress(0.05 + 0.9 * f, m))
        # Background = mixture minus speech: keeps everything that isn't voice and is
        # exactly complementary to the vocal stem (no phase artefacts from re-summing).
        sf.write(job.path(VOCALS), vocals, sr, subtype="PCM_16")
        sf.write(job.path(BACKGROUND), np.clip(mix - vocals, -1, 1), sr, subtype="PCM_16")
        progress(1.0, "stems written")
