"""Background job execution and live state for the API.

Jobs run one at a time on a single worker thread (the GPU has 4 GB). Live progress
is kept in memory; the durable part (API state, edit flags) is persisted next to the
manifest in ``api_state.json`` so it survives a server restart.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dubber.pipeline import Job, Pipeline, PipelineEvent
from dubber.pipeline.job import StageStatus
from dubber.pipeline.stage import Stage
from dubber.stages import REVIEW_STAGE, build_stages

log = logging.getLogger(__name__)

API_STATE = "api_state.json"
TERMINAL = {"done", "failed", "awaiting_review"}


@dataclass
class LiveState:
    state: str = "queued"  # queued | running | awaiting_review | done | failed | idle
    stage: str | None = None
    progress: float = 0.0
    message: str = ""
    events: list[dict] = field(default_factory=list)


class JobManager:
    def __init__(self, jobs_dir: Path,
                 stages_factory: Callable[..., list[Stage]] = build_stages):
        self.jobs_dir = Path(jobs_dir)
        self.stages_factory = stages_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._lock = threading.Lock()
        self._live: dict[str, LiveState] = {}

    # -- persistence -------------------------------------------------------------
    def job(self, job_id: str) -> Job:
        root = self.jobs_dir / job_id
        if not job_id or Path(job_id).name != job_id or not (root / "manifest.json").exists():
            raise KeyError(job_id)
        return Job(root)

    def job_ids(self) -> list[str]:
        if not self.jobs_dir.exists():
            return []
        ids = [p.name for p in self.jobs_dir.iterdir() if (p / "manifest.json").exists()]
        return sorted(ids, reverse=True)

    def api_state(self, job: Job) -> dict:
        path = job.path(API_STATE)
        if path.exists():
            return json.loads(path.read_text("utf-8"))
        return {"state": "idle", "edited": [], "error": None}

    def _save_api_state(self, job: Job, **changes) -> dict:
        data = self.api_state(job) | changes
        tmp = job.path(API_STATE + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, job.path(API_STATE))
        return data

    def mark_edited(self, job: Job, earliest_stage: str) -> None:
        edited = set(self.api_state(job).get("edited", []))
        self._save_api_state(job, edited=sorted(edited | {earliest_stage}))

    # -- status ------------------------------------------------------------------
    def live(self, job_id: str) -> LiveState | None:
        return self._live.get(job_id)

    def state(self, job: Job) -> str:
        live = self._live.get(job.id)
        if live and live.state in {"queued", "running"}:
            return live.state
        saved = self.api_state(job).get("state", "idle")
        if saved in {"idle", "failed"}:
            # Jobs run or resumed from the CLI only update the manifest; derive their
            # state from it so they show correctly in the UI.
            return self._manifest_state(job) or saved
        return saved

    @staticmethod
    def _manifest_state(job: Job) -> str | None:
        records = job.manifest.stages.values()
        if any(r.status == "failed" for r in records):
            return "failed"
        if records and all(r.status == "done" for r in records) \
                and job.path("output.mp4").exists():
            return "done"
        return None

    def is_busy(self, job: Job) -> bool:
        return self.state(job) in {"queued", "running"}

    def any_busy(self) -> bool:
        return any(live.state in {"queued", "running"} for live in self._live.values())

    def run_exclusive(self, fn: Callable[[], object]) -> object:
        """Run short work (e.g. re-voicing one line) on the single pipeline worker so it
        never overlaps a running job, and wait for its result."""
        return self._executor.submit(fn).result()

    def events_since(self, job_id: str, index: int) -> list[dict]:
        live = self._live.get(job_id)
        return live.events[index:] if live else []

    # -- execution ---------------------------------------------------------------
    def submit(self, job: Job, force: Iterable[str] = (), review: bool | None = None) -> None:
        if self.is_busy(job):
            raise RuntimeError("job is already queued or running")
        review = bool(job.option("review", False)) if review is None else review
        with self._lock:
            self._live[job.id] = LiveState(state="queued", message="waiting for worker")
        self._push(job.id, {"status": "queued", "stage": None, "progress": 0.0,
                            "message": "waiting for worker"})
        self._save_api_state(job, state="queued", error=None)
        self._executor.submit(self._run, job.id, list(force), review)

    def _run(self, job_id: str, force: list[str], review: bool) -> None:
        job = self.job(job_id)
        live = self._live[job_id]
        live.state = "running"
        self._save_api_state(job, state="running")
        try:
            stages = self.stages_factory(lipsync=bool(job.option("lipsync", False)))
            names = {s.name for s in stages}
            stop_after = REVIEW_STAGE if review and REVIEW_STAGE in names else None
            pipeline = Pipeline(stages, on_event=self._on_event)
            finished = pipeline.run(job, force=[f for f in force if f in names],
                                    stop_after=stop_after)
            state = "done" if finished else "awaiting_review"
            # Reload: the pipeline thread wrote the manifest through its own Job object.
            self._save_api_state(Job(job.root), state=state, edited=[], error=None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            log.exception("job %s failed", job_id)
            state = "failed"
            self._save_api_state(job, state=state, error=f"{type(exc).__name__}: {exc}")
        live.state = state
        self._push(job_id, {"status": state, "stage": live.stage, "progress":
                            1.0 if state == "done" else live.progress, "message": state})

    def _on_event(self, e: PipelineEvent) -> None:
        self._push(e.job_id, {"status": e.status, "stage": e.stage, "progress": e.progress,
                              "message": e.message})

    def _push(self, job_id: str, event: dict) -> None:
        with self._lock:
            live = self._live.setdefault(job_id, LiveState())
            if event.get("stage"):
                live.stage = event["stage"]
            live.progress = event["progress"]
            live.message = event["message"]
            live.events.append({**event, "seq": len(live.events), "state": live.state})

    def completed_stages(self, job: Job) -> list[str]:
        return [name for name, rec in job.manifest.stages.items()
                if rec.status == StageStatus.DONE]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
