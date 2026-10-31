"""Speaker diarization with pyannote 3.1, merged with ASR word timings."""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from dubber import ml
from dubber.align import Turn, assign_speakers
from dubber.artifacts import TRANSCRIPT, VOCALS
from dubber.config import get_settings
from dubber.errors import DubberError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Transcript
from dubber.stages.transcribe import TranscribeStage

log = logging.getLogger(__name__)

DIARIZATION = "diarization.json"
PIPELINE_ID = "pyannote/speaker-diarization-3.1"
SR = 16000


def load_mono_16k(path) -> np.ndarray:
    import librosa

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR)


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
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        token = get_settings().hf_token
        if not token:
            raise DubberError("HF_TOKEN is not set - required for pyannote diarization")

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
