"""Speaker diarization with pyannote 3.1, merged with ASR word timings."""

from __future__ import annotations

import logging

from dubber import ml
from dubber.align import Turn, assign_speakers
from dubber.artifacts import TRANSCRIPT, VOCALS
from dubber.config import get_settings
from dubber.errors import DubberError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Transcript
from dubber.stages.transcribe import SR, TranscribeStage, load_mono_16k

log = logging.getLogger(__name__)

DIARIZATION = "diarization.json"
PIPELINE_ID = "pyannote/speaker-diarization-3.1"
# Gated repos the pipeline pulls in; each needs its terms accepted on huggingface.co.
GATED_MODELS = {"pyannote/speaker-diarization-3.1": "config.yaml",
                "pyannote/segmentation-3.0": "config.yaml"}


def _allow_pyannote_checkpoints() -> None:
    """PyTorch >= 2.6 loads checkpoints with ``weights_only=True``, which rejects any
    class not explicitly allow-listed. pyannote 3.x checkpoints pickle a few of their
    own metadata classes; allow exactly those instead of disabling the safety check."""
    import torch
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals(
        [torch.torch_version.TorchVersion, Specifications, Problem, Resolution])


def check_model_access(token: str) -> None:
    """Fail fast with an actionable message: pyannote itself swallows 403s and
    later crashes with "'NoneType' object has no attribute 'eval'"."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    for repo, filename in GATED_MODELS.items():
        try:
            hf_hub_download(repo, filename, token=token)
        except GatedRepoError:
            raise DubberError(f"HF_TOKEN's account has not accepted the terms of {repo}: "
                              f"open https://huggingface.co/{repo}, click 'Agree', then "
                              "retry") from None
        except RepositoryNotFoundError:
            raise DubberError(f"HF_TOKEN is invalid or cannot see {repo}") from None


def _hf_hub_compat() -> None:
    """pyannote 3.x passes ``use_auth_token=`` to ``hf_hub_download``; huggingface_hub
    >= 1.0 only accepts ``token=``. Translate the argument instead of pinning an old
    hub version that the other ML libraries (faster-whisper, tokenizers) have outgrown.
    Must run before pyannote is imported, since it binds the function at import time."""
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_dubber_compat", False):
        return

    def hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None:
            kwargs.setdefault("token", use_auth_token)
        return original(*args, **kwargs)

    hf_hub_download._dubber_compat = True
    huggingface_hub.hf_hub_download = hf_hub_download


class DiarizeStage(Stage):
    name = "diarize"
    outputs = (DIARIZATION, TRANSCRIPT)
    weight = 1.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        raw = TranscribeStage.raw(job)
        turns = self._diarize(job, progress) if raw.segments else []
        job.write_json(DIARIZATION, [t.__dict__ for t in turns])

        progress(0.9, "merging speakers with words")
        lines = assign_speakers(raw.segments, turns)
        transcript = Transcript(segments=lines, speech_ratio=raw.speech_ratio)
        job.write_model(TRANSCRIPT, transcript)
        progress(1.0, f"{len(transcript.speakers)} speaker(s), {len(lines)} lines")

    def _diarize(self, job: Job, progress: ProgressFn) -> list[Turn]:
        token = get_settings().hf_token
        if not token:
            raise DubberError("HF_TOKEN is not set - required for pyannote diarization")
        check_model_access(token)
        _hf_hub_compat()  # must precede the first pyannote import
        _allow_pyannote_checkpoints()
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        audio = load_mono_16k(job.path(VOCALS))
        with ml.gpu_scope():
            progress(0.05, "loading pyannote")
            pipe = PyannotePipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
            if pipe is None:
                raise DubberError(f"could not load {PIPELINE_ID}: accept its terms on "
                                  "huggingface.co and check HF_TOKEN")
            pipe.to(torch.device(ml.device()))
            # Passing the waveform in memory avoids torchaudio file decoding entirely.
            waveform = torch.from_numpy(audio)[None]
            options = {}
            if (n := job.option("num_speakers")) is not None:
                options["num_speakers"] = int(n)
            progress(0.2, "diarizing")
            diarization = pipe({"waveform": waveform, "sample_rate": SR}, **options)
            del pipe

        return [Turn(round(seg.start, 3), round(seg.end, 3), speaker)
                for seg, _, speaker in diarization.itertracks(yield_label=True)]
