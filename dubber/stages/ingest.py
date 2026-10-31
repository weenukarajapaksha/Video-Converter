"""Task 1: validate the input video, extract metadata and the original audio."""

from dubber import media
from dubber.artifacts import METADATA, ORIGINAL_AUDIO
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import VideoMetadata

# 44.1 kHz stereo is what Demucs is trained on; downstream stages resample as needed.
SEPARATION_SR = 44100


class IngestStage(Stage):
    name = "ingest"
    outputs = (METADATA, ORIGINAL_AUDIO)
    weight = 0.5

    def run(self, job: Job, progress: ProgressFn) -> None:
        settings = get_settings()
        progress(0.1, "probing video")
        meta = media.probe(job.input_video)
        media.validate(meta, settings.min_duration_s, settings.max_duration_s)
        job.write_model(METADATA, meta)

        progress(0.4, "extracting audio")
        media.extract_audio(job.input_video, job.path(ORIGINAL_AUDIO),
                            sample_rate=SEPARATION_SR, channels=2)
        progress(1.0, f"{meta.resolution} @ {meta.fps}fps, {meta.duration_s:.1f}s")

    @staticmethod
    def metadata(job: Job) -> VideoMetadata:
        return job.read_model(METADATA, VideoMetadata)
