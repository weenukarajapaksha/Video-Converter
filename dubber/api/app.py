"""FastAPI application: upload, live progress (SSE), review/edit, downloads, reports."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dubber import artifacts
from dubber.api.manager import TERMINAL, JobManager
from dubber.config import get_settings
from dubber.pipeline import Job
from dubber.schemas import Transcript, Translation, VideoMetadata, VoiceMap
from dubber.voices_catalog import VOICES

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

# Downloadable artifacts: every file name in dubber.artifacts except directories.
DOWNLOADABLE = {
    value for key, value in vars(artifacts).items()
    if key.isupper() and isinstance(value, str) and "." in value
}
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".json": "application/json",
    ".srt": "application/x-subrip",
}


class JobSummary(BaseModel):
    job_id: str
    input_name: str
    created_at: str
    state: str
    stage: str | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    options: dict
    artifacts: list[str]
    stages: dict
    total_duration_s: float
    total_cost_usd: float
    metadata: VideoMetadata | None = None


def create_app(manager: JobManager | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Dubber", version="0.1.0",
                  description="AI video translation & dubbing (Chinese -> English)")
    app.state.manager = manager or JobManager(settings.jobs_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )

    def get_manager(request: Request) -> JobManager:
        return request.app.state.manager

    def get_job(job_id: str, mgr: JobManager = Depends(get_manager)) -> Job:
        try:
            return mgr.job(job_id)
        except KeyError:
            raise HTTPException(404, f"job '{job_id}' not found") from None

    def summary(mgr: JobManager, job: Job) -> JobSummary:
        m = job.manifest
        live = mgr.live(job.id)
        api_state = mgr.api_state(job)
        meta = (job.read_model(artifacts.METADATA, VideoMetadata)
                if job.path(artifacts.METADATA).exists() else None)
        done = mgr.state(job) == "done"
        return JobSummary(
            job_id=m.job_id, input_name=m.input_name, created_at=m.created_at.isoformat(),
            state=mgr.state(job),
            stage=live.stage if live else None,
            progress=1.0 if done else (live.progress if live else 0.0),
            message=live.message if live else "",
            error=api_state.get("error"),
            options=m.options,
            artifacts=sorted(n for n in DOWNLOADABLE if job.path(n).is_file()),
            stages={k: v.model_dump(mode="json") for k, v in m.stages.items()},
            total_duration_s=m.total_duration_s, total_cost_usd=m.total_cost_usd,
            metadata=meta,
        )

    def require_editable(mgr: JobManager, job: Job) -> None:
        if mgr.state(job) not in {"awaiting_review", "done", "failed", "idle"}:
            raise HTTPException(409, "job is running; wait for review or completion")

    # -- jobs --------------------------------------------------------------------
    @app.post("/api/jobs", response_model=JobSummary, status_code=201)
    def create_job(
        file: UploadFile = File(...),
        target_language: str = Form("English"),
        tts_provider: Literal["gemini", "edge"] = Form(settings.tts_provider),
        review: bool = Form(False),
        lipsync: bool = Form(False),
        mgr: JobManager = Depends(get_manager),
    ):
        if not (file.filename or "").lower().endswith(".mp4"):
            raise HTTPException(415, "only .mp4 files are supported")
        with tempfile.TemporaryDirectory() as tmp:
            upload = Path(tmp) / Path(file.filename).name
            with upload.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            options = {"target_language": target_language, "tts_provider": tts_provider,
                       "review": review, "lipsync": lipsync}
            job = Job.create(mgr.jobs_dir, upload, options)
        mgr.submit(job)
        return summary(mgr, job)

    @app.get("/api/jobs", response_model=list[JobSummary])
    def list_jobs(mgr: JobManager = Depends(get_manager)):
        return [summary(mgr, mgr.job(i)) for i in mgr.job_ids()]

    @app.get("/api/jobs/{job_id}", response_model=JobSummary)
    def get_job_summary(job: Job = Depends(get_job), mgr: JobManager = Depends(get_manager)):
        return summary(mgr, job)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(request: Request, job: Job = Depends(get_job),
                         mgr: JobManager = Depends(get_manager)):
        async def stream():
            snapshot = summary(mgr, job).model_dump(mode="json")
            yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
            index = len(mgr.events_since(job.id, 0))
            idle = 0.0
            while not await request.is_disconnected():
                events = mgr.events_since(job.id, index)
                for e in events:
                    yield f"data: {json.dumps(e)}\n\n"
                index += len(events)
                if events:
                    idle = 0.0
                elif idle >= 15:
                    yield ": keep-alive\n\n"
                    idle = 0.0
                await asyncio.sleep(0.25)
                idle += 0.25

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{job_id}/continue", response_model=JobSummary)
    def continue_job(job: Job = Depends(get_job), mgr: JobManager = Depends(get_manager)):
        if mgr.is_busy(job):
            raise HTTPException(409, "job is already running")
        force = mgr.api_state(job).get("edited", [])
        mgr.submit(job, force=force, review=False)
        return summary(mgr, job)

    @app.post("/api/jobs/{job_id}/retry", response_model=JobSummary)
    def retry_job(job: Job = Depends(get_job), mgr: JobManager = Depends(get_manager)):
        if mgr.is_busy(job):
            raise HTTPException(409, "job is already running")
        mgr.submit(job, force=mgr.api_state(job).get("edited", []))
        return summary(mgr, job)

    # -- editable artifacts --------------------------------------------------------
    def _read(job: Job, name: str, model):
        if not job.path(name).exists():
            raise HTTPException(404, f"{name} not available yet")
        return job.read_model(name, model)

    @app.get("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def get_transcript(job: Job = Depends(get_job)):
        return _read(job, artifacts.TRANSCRIPT, Transcript)

    @app.put("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def put_transcript(body: Transcript, job: Job = Depends(get_job),
                       mgr: JobManager = Depends(get_manager)):
        require_editable(mgr, job)
        _read(job, artifacts.TRANSCRIPT, Transcript)
        job.write_model(artifacts.TRANSCRIPT, body)
        mgr.mark_edited(job, "translate")
        return body

    @app.get("/api/jobs/{job_id}/translation", response_model=Translation)
    def get_translation(job: Job = Depends(get_job)):
        return _read(job, artifacts.TRANSLATION, Translation)

    @app.put("/api/jobs/{job_id}/translation", response_model=Translation)
    def put_translation(body: Translation, job: Job = Depends(get_job),
                        mgr: JobManager = Depends(get_manager)):
        require_editable(mgr, job)
        current = _read(job, artifacts.TRANSLATION, Translation)
        edits = {s.id: s for s in body.segments}
        unknown = set(edits) - {s.id for s in current.segments}
        if unknown:
            raise HTTPException(422, f"unknown segment ids: {sorted(unknown)}")
        for seg in current.segments:
            if seg.id in edits:  # only wording and delivery are editable; timing is not
                text = edits[seg.id].text.strip()
                if not text:
                    raise HTTPException(422, f"segment {seg.id}: text must not be empty")
                seg.text, seg.emotion = text, edits[seg.id].emotion
        job.write_model(artifacts.TRANSLATION, current)
        mgr.mark_edited(job, "synthesize")
        return current

    @app.get("/api/jobs/{job_id}/voices", response_model=VoiceMap)
    def get_voices(job: Job = Depends(get_job)):
        return _read(job, artifacts.VOICES, VoiceMap)

    @app.put("/api/jobs/{job_id}/voices", response_model=VoiceMap)
    def put_voices(body: VoiceMap, job: Job = Depends(get_job),
                   mgr: JobManager = Depends(get_manager)):
        require_editable(mgr, job)
        current = _read(job, artifacts.VOICES, VoiceMap)
        if set(body.speakers) != set(current.speakers):
            raise HTTPException(422, "speaker set must match the diarized speakers")
        valid = {v["id"] for v in VOICES.get(body.provider, [])}
        bad = [p.voice for p in body.speakers.values() if valid and p.voice not in valid]
        if bad:
            raise HTTPException(422, f"unknown voices for {body.provider}: {bad}")
        job.write_model(artifacts.VOICES, body)
        mgr.mark_edited(job, "synthesize")
        return body

    # -- downloads & reports -------------------------------------------------------
    @app.get("/api/jobs/{job_id}/files/{name}")
    def get_file(name: str, download: bool = False, job: Job = Depends(get_job)):
        if name not in DOWNLOADABLE:
            raise HTTPException(404, "unknown artifact")
        path = job.path(name)
        if not path.is_file():
            raise HTTPException(404, f"{name} not available")
        stem = Path(job.manifest.input_name).stem
        return FileResponse(
            path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
            filename=f"{stem}_{name}" if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/api/jobs/{job_id}/report")
    def get_report(job: Job = Depends(get_job)):
        m = job.manifest
        by_stage: dict[str, float] = defaultdict(float)
        by_provider: dict[str, float] = defaultdict(float)
        for c in m.costs:
            by_stage[c.stage] += c.usd
            by_provider[c.provider] += c.usd
        meta = (job.read_model(artifacts.METADATA, VideoMetadata)
                if job.path(artifacts.METADATA).exists() else None)
        minutes = meta.duration_s / 60 if meta else None
        return {
            "job_id": m.job_id,
            "video_duration_s": meta.duration_s if meta else None,
            "stages": [
                {"stage": name, "status": rec.status, "attempts": rec.attempts,
                 "duration_s": rec.duration_s, "cost_usd": round(by_stage.get(name, 0.0), 6),
                 "error": rec.error}
                for name, rec in m.stages.items()
            ],
            "costs": [c.model_dump() for c in m.costs],
            "cost_by_provider": {k: round(v, 6) for k, v in by_provider.items()},
            "total_cost_usd": m.total_cost_usd,
            "total_duration_s": m.total_duration_s,
            "cost_per_minute_usd": round(m.total_cost_usd / minutes, 6) if minutes else None,
            "realtime_factor": round(m.total_duration_s / meta.duration_s, 3) if meta else None,
        }

    @app.get("/api/voices")
    def list_voices(provider: Literal["gemini", "edge"] = "gemini"):
        return VOICES[provider]

    @app.get("/api/health")
    def health():
        return {"status": "ok", "terminal_states": sorted(TERMINAL)}

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


app = create_app()
