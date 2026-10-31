"""REST API behaviour, with the ML pipeline replaced by the lightweight ingest stage."""

import time

import pytest
from fastapi.testclient import TestClient

from dubber import artifacts
from dubber.api.app import create_app
from dubber.api.manager import JobManager
from dubber.pipeline import Job
from dubber.schemas import TranslatedSegment, Translation
from dubber.stages.ingest import IngestStage


@pytest.fixture
def client(isolated_settings):
    manager = JobManager(isolated_settings.jobs_dir,
                         stages_factory=lambda lipsync=False: [IngestStage()])
    with TestClient(create_app(manager)) as c:
        yield c
    manager.shutdown()


def _wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["state"] not in {"queued", "running"}:
            return body
        time.sleep(0.1)
    raise TimeoutError(job_id)


def _upload(client, video, **form):
    with open(video, "rb") as fh:
        return client.post("/api/jobs", files={"file": ("clip.mp4", fh, "video/mp4")},
                           data=form)


def test_upload_runs_pipeline_and_exposes_artifacts(client, sample_video):
    resp = _upload(client, sample_video, target_language="English", review="false")
    assert resp.status_code == 201
    job = _wait(client, resp.json()["job_id"])

    assert job["state"] == "done"
    assert job["metadata"]["width"] == 640
    assert artifacts.ORIGINAL_AUDIO in job["artifacts"]
    assert job["stages"]["ingest"]["status"] == "done"
    report = client.get(f"/api/jobs/{job['job_id']}/report").json()
    assert report["stages"][0]["stage"] == "ingest"
    assert report["realtime_factor"] is not None


def test_upload_rejects_non_mp4(client, tmp_path):
    bogus = tmp_path / "clip.avi"
    bogus.write_bytes(b"RIFF")
    with open(bogus, "rb") as fh:
        resp = client.post("/api/jobs", files={"file": ("clip.avi", fh, "video/avi")})
    assert resp.status_code == 415
    assert client.get("/api/jobs").json() == []


def test_file_endpoint_only_serves_whitelisted_artifacts(client, sample_video):
    job_id = _upload(client, sample_video).json()["job_id"]
    _wait(client, job_id)

    ok = client.get(f"/api/jobs/{job_id}/files/{artifacts.INPUT_VIDEO}",
                    headers={"Range": "bytes=0-99"})
    assert ok.status_code == 206 and len(ok.content) == 100
    for name in ["manifest.json", "api_state.json", "..%2F..%2Fpyproject.toml"]:
        assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc/files/input.mp4").status_code == 404


def test_translation_edit_changes_wording_but_never_timing(client, sample_video,
                                                           isolated_settings):
    job_id = _upload(client, sample_video).json()["job_id"]
    _wait(client, job_id)
    job = Job(isolated_settings.jobs_dir / job_id)
    original = Translation(model="test", segments=[
        TranslatedSegment(id=0, start=1.0, end=3.0, speaker="SPEAKER_00",
                          source_text="你今天为什么来这里？", text="Why are you here today?",
                          max_duration_s=2.4),
    ])
    job.write_model(artifacts.TRANSLATION, original)

    edited = original.model_copy(deep=True)
    edited.segments[0].text = "Why'd you come today?"
    edited.segments[0].emotion = "curious"
    edited.segments[0].start, edited.segments[0].end = 9.0, 20.0  # must be ignored
    resp = client.put(f"/api/jobs/{job_id}/translation", json=edited.model_dump())
    assert resp.status_code == 200
    seg = job.read_model(artifacts.TRANSLATION, Translation).segments[0]
    assert (seg.text, seg.emotion) == ("Why'd you come today?", "curious")
    assert (seg.start, seg.end, seg.max_duration_s) == (1.0, 3.0, 2.4)

    unknown = edited.model_dump()
    unknown["segments"][0]["id"] = 99
    assert client.put(f"/api/jobs/{job_id}/translation", json=unknown).status_code == 422
    malformed = {"model": "x", "segments": [{"id": 0}]}
    assert client.put(f"/api/jobs/{job_id}/translation", json=malformed).status_code == 422
    # The edit is remembered so `continue` regenerates the dub.
    assert "synthesize" in client.app.state.manager.api_state(job)["edited"]


def test_voice_catalog_lists_genders(client):
    voices = client.get("/api/voices", params={"provider": "edge"}).json()
    assert {v["gender"] for v in voices} == {"male", "female"}
    assert client.get("/api/voices", params={"provider": "nope"}).status_code == 422


def test_quality_report_turns_signals_into_warnings(isolated_settings, sample_video):
    from dubber.api.app import build_quality

    job = Job.create(isolated_settings.jobs_dir, sample_video)
    job.write_json("asr_stats.json", {"vad_speech_regions": 11, "recovered_regions": 2,
                                      "segments": 3, "low_confidence_segments": [0]})
    job.write_json(artifacts.TRANSCRIPT, {"segments": [
        {"id": 0, "start": 0, "end": 2, "speaker": "SPEAKER_01", "text": "坦黛", "confidence": 0.45},
        {"id": 1, "start": 3, "end": 5, "speaker": "SPEAKER_02", "text": "好", "confidence": 0.95},
    ], "speech_ratio": 0.8})
    job.write_json(artifacts.VOICES, {"provider": "edge", "speakers": {
        f"SPEAKER_0{i}": {"speaker": f"SPEAKER_0{i}", "gender": g, "voice": "v"}
        for i, g in enumerate(["male", "female", "female", "male", "unknown"], 1)}})
    job.write_json(artifacts.TRANSLATION, {"model": "m", "review": None, "segments": [
        {"id": 0, "start": 0, "end": 2, "speaker": "SPEAKER_01", "source_text": "x",
         "text": "a", "max_duration_s": 2.0},
        {"id": 1, "start": 3, "end": 5, "speaker": "SPEAKER_02", "source_text": "y",
         "text": "b", "max_duration_s": 1.0}]})
    job.write_json(artifacts.DUB_TRACK, {"lines": [
        {"id": 0, "final_duration_s": 1.5, "tempo": 1.0, "rewrites": 0},
        {"id": 1, "final_duration_s": 1.3, "tempo": 1.25, "rewrites": 2}]})

    q = build_quality(job)

    assert q["asr"]["recovered_regions"] == 2
    assert [s["id"] for s in q["asr"]["low_confidence"]] == [0]
    assert q["speakers"] == {"count": 5, "genders": {"male": 2, "female": 2, "unknown": 1}}
    assert q["sync"]["overflow"] == [{"id": 1, "by_s": 0.3}]
    assert q["sync"]["stretched"] == 1 and q["sync"]["rewritten"] == 1
    messages = " | ".join(w["message"] for w in q["warnings"])
    for expected in ("low ASR confidence", "Recovered 2", "5 speakers", "overflow",
                     "critic was skipped"):
        assert expected in messages


def test_quality_report_tolerates_missing_artifacts(client, sample_video):
    job_id = _upload(client, sample_video).json()["job_id"]
    _wait(client, job_id)
    body = client.get(f"/api/jobs/{job_id}/quality").json()
    assert body == {"warnings": []}
