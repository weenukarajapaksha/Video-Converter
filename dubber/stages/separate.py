"""Split the soundtrack into a speech stem and a music/SFX stem with Demucs.

The vocal stem feeds ASR + diarization (much more robust under background music),
and the background stem is what the English dialogue is mixed onto, so the
original Chinese voices disappear while music and effects are preserved.
"""

import numpy as np
import soundfile as sf

from dubber import ml
from dubber.artifacts import BACKGROUND, ORIGINAL_AUDIO, VOCALS
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage

# Fine-tuned bag of 4 htdemucs models: best open-source vocal SDR that fits in 4 GB.
MODEL = "htdemucs_ft"
CHUNK_S = 60  # process long audio in chunks to report progress and cap memory


class SeparateStage(Stage):
    name = "separate"
    outputs = (VOCALS, BACKGROUND)
    weight = 2.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        audio, sr = sf.read(job.path(ORIGINAL_AUDIO), dtype="float32", always_2d=True)
        mix = torch.from_numpy(audio.T.copy())  # (channels, samples)

        with ml.gpu_scope():
            progress(0.02, f"loading {MODEL}")
            model = get_model(MODEL)
            model.eval()
            dev = ml.device()
            vocal_idx = model.sources.index("vocals")

            ref = mix.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            normed = (mix - mean) / std

            step = CHUNK_S * sr
            vocals = torch.zeros_like(mix)
            for i, lo in enumerate(range(0, mix.shape[1], step)):
                hi = min(lo + step, mix.shape[1])
                with torch.inference_mode():
                    est = apply_model(model, normed[None, :, lo:hi], device=dev, shifts=1,
                                      split=True, overlap=0.25, progress=False)[0]
                vocals[:, lo:hi] = est[vocal_idx].cpu() * std + mean
                progress(0.05 + 0.9 * hi / mix.shape[1], f"separated {hi / sr:.0f}s")
            del model

        # Background = mixture minus speech: keeps everything that isn't voice and is
        # exactly complementary to the vocal stem (no phase artefacts from re-summing).
        background = mix - vocals
        sf.write(job.path(VOCALS), vocals.numpy().T, sr, subtype="PCM_16")
        sf.write(job.path(BACKGROUND), np.clip(background.numpy().T, -1, 1), sr,
                 subtype="PCM_16")
        progress(1.0, "stems written")
