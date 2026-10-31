"""Single-line re-voicing (human edit -> new audio in seconds), with a fake TTS."""

import numpy as np
import pytest
import soundfile as sf

from dubber import artifacts
from dubber.pipeline import Job, Pipeline
from dubber.schemas import (DubTrack, SpeakerProfile, TranslatedSegment, Translation,
                            VoiceMap)
from dubber.stages import synthesize
from dubber.stages.ingest import IngestStage
from dubber.tts import SR, TTSProvider

WORD_S = 0.3  # fake speaking rate: 0.3 s per word


class FakeTTS(TTSProvider):
    name = "fake"

    def _generate(self, text, voice, emotion, style, language):
        t = np.arange(int(len(text.split()) * WORD_S * SR)) / SR
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 0.0


@pytest.fixture
def dubbed_job(tmp_path, sample_video, monkeypatch, isolated_settings):
    monkeypatch.setattr(synthesize, "get_provider", lambda name: FakeTTS())
    job = Job.create(isolated_settings.jobs_dir, sample_video)
    Pipeline([IngestStage()]).run(job)
    segs = [TranslatedSegment(id=0, start=1.0, end=2.5, speaker="SPEAKER_01", source_text="你好",
                              text="Hello there", max_duration_s=2.0),
            TranslatedSegment(id=1, start=5.0, end=6.0, speaker="SPEAKER_01", source_text="走吧",
                              text="Let us go", max_duration_s=1.5)]
    job.write_model(artifacts.TRANSLATION, Translation(model="m", segments=segs))
    job.write_model(artifacts.VOICES, VoiceMap(provider="edge", speakers={
        "SPEAKER_01": SpeakerProfile(speaker="SPEAKER_01", voice="v")}))
    tts, voices = FakeTTS(), job.read_model(artifacts.VOICES, VoiceMap)
    lines = [synthesize.fit_line(job, s, synthesize.speak(tts, job, s, s.text, voices, "English"),
                                 s.text, 0) for s in segs]
    job.write_model(artifacts.DUB_TRACK, DubTrack(lines=lines))
    synthesize.assemble_timeline(job, lines)
    return job


def test_revoice_updates_only_that_line_and_rebuilds_the_track(dubbed_job):
    before = {ln.id: ln for ln in dubbed_job.read_model(artifacts.DUB_TRACK, DubTrack).lines}

    line = synthesize.revoice_line(dubbed_job, 1, text="Come on, we really should go now",
                                   emotion="excited")

    track = {ln.id: ln for ln in dubbed_job.read_model(artifacts.DUB_TRACK, DubTrack).lines}
    seg = next(s for s in dubbed_job.read_model(artifacts.TRANSLATION, Translation).segments
               if s.id == 1)
    assert seg.text == "Come on, we really should go now" and seg.emotion == "excited"
    assert track[0] == before[0]  # untouched line kept as-is
    assert track[1].text == seg.text and line.start == 5.0
    # 7 words * 0.3 s = 2.1 s raw vs a 1.5 s budget: stretched to the 1.25x cap, no rewrite.
    assert line.tempo == pytest.approx(1.25) and line.rewrites == 0
    assert line.final_duration_s == pytest.approx(2.1 / 1.25, abs=0.05)

    voice, sr = sf.read(dubbed_job.path(artifacts.DUB_VOICE))
    assert len(voice) / sr == pytest.approx(IngestStage.metadata(dubbed_job).duration_s, abs=0.01)
    assert np.abs(voice[int(5.1 * sr):int(6.5 * sr)]).max() > 0.1  # new clip placed at 5 s
    assert np.abs(voice[int(3.5 * sr):int(4.5 * sr)]).max() == 0  # silence stays silent


def test_revoice_rejects_unknown_line_and_empty_text(dubbed_job):
    from dubber.errors import DubberError

    with pytest.raises(DubberError, match="unknown line"):
        synthesize.revoice_line(dubbed_job, 99)
    with pytest.raises(DubberError, match="empty"):
        synthesize.revoice_line(dubbed_job, 0, text="   ")
