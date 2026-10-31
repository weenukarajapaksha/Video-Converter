import soundfile as sf
import pytest

from dubber import media
from dubber.errors import InputValidationError
from dubber.pipeline import Job, Pipeline
from dubber.stages import IngestStage
from dubber.stages.ingest import ORIGINAL_AUDIO
from tests.conftest import make_video


def test_probe_reports_video_and_audio_properties(sample_video):
    meta = media.probe(sample_video)
    assert meta.resolution == "640x360"
    assert meta.fps == pytest.approx(30.0)
    assert meta.video_codec == "h264"
    assert meta.audio_codec == "aac"
    assert meta.duration_s == pytest.approx(32.0, abs=0.1)
    assert meta.audio_duration_s == pytest.approx(32.0, abs=0.1)


@pytest.mark.parametrize("seconds, audio, reason", [
    (5, True, "outside the supported range"),
    (31, False, "no audio track"),
])
def test_validate_rejects_unusable_input(tmp_path, seconds, audio, reason):
    meta = media.probe(make_video(tmp_path / "bad.mp4", seconds, audio=audio))
    with pytest.raises(InputValidationError, match=reason):
        media.validate(meta, min_s=30, max_s=180)


def test_probe_rejects_non_media_file(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"definitely not a video")
    with pytest.raises(InputValidationError):
        media.probe(junk)


def test_ingest_stage_extracts_full_length_stereo_audio(sample_video, isolated_settings):
    job = Job.create(isolated_settings.jobs_dir, sample_video)
    Pipeline([IngestStage()]).run(job)

    info = sf.info(job.path(ORIGINAL_AUDIO))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.duration == pytest.approx(32.0, abs=0.1)
    assert IngestStage.metadata(job).fps == pytest.approx(30.0)
