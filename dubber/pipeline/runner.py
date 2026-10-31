"""Executes stages in order with checkpointing, retries and progress events."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from dubber.pipeline.job import Job, StageStatus, _now
from dubber.pipeline.stage import Stage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineEvent:
    job_id: str
    stage: str
    status: str  # running | progress | done | skipped | retrying | failed
    progress: float  # overall pipeline progress in [0, 1]
    message: str = ""


class Pipeline:
    def __init__(
        self,
        stages: Sequence[Stage],
        on_event: Callable[[PipelineEvent], None] | None = None,
        backoff_s: float = 2.0,
    ):
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")
        self.stages = list(stages)
        self.on_event = on_event or (lambda e: None)
        self.backoff_s = backoff_s
        total = sum(s.weight for s in self.stages) or 1.0
        self._offsets: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for s in self.stages:
            self._offsets[s.name] = (acc / total, s.weight / total)
            acc += s.weight

    def run(self, job: Job, force: Iterable[str] = ()) -> None:
        """Run all stages. Completed stages are skipped unless named in ``force``.

        Once any stage actually executes, every downstream stage re-runs as well,
        because its inputs may have changed.
        """
        force = set(force)
        unknown = force - set(self._offsets)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")

        dirty = False
        for stage in self.stages:
            if not dirty and stage.name not in force and stage.is_complete(job):
                self._emit(job, stage, "skipped", 1.0, "already complete")
                continue
            dirty = True
            self._run_stage(job, stage)

    def _run_stage(self, job: Job, stage: Stage) -> None:
        record = job.record(stage.name)
        record.attempts = 0
        for attempt in range(1, stage.max_attempts + 1):
            record.status = StageStatus.RUNNING
            record.attempts = attempt
            record.started_at = _now()
            record.finished_at = record.duration_s = record.error = None
            job.clear_costs(stage.name)
            job.save()
            self._emit(job, stage, "running", 0.0, f"attempt {attempt}")

            t0 = time.perf_counter()
            try:
                stage.run(job, lambda frac, msg: self._emit(job, stage, "progress", frac, msg))
            except stage.retry_on as exc:
                self._finish(job, record, t0, exc)
                if attempt == stage.max_attempts:
                    self._emit(job, stage, "failed", 0.0, str(exc))
                    raise
                delay = self.backoff_s * 2 ** (attempt - 1)
                log.warning("%s failed (%s); retrying in %.1fs", stage.name, exc, delay)
                self._emit(job, stage, "retrying", 0.0, f"{exc} - retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            except BaseException as exc:
                self._finish(job, record, t0, exc)
                self._emit(job, stage, "failed", 0.0, str(exc))
                raise

            self._finish(job, record, t0, None)
            self._emit(job, stage, "done", 1.0, f"{record.duration_s:.1f}s")
            return

    @staticmethod
    def _finish(job: Job, record, t0: float, exc: BaseException | None) -> None:
        record.finished_at = _now()
        record.duration_s = round(time.perf_counter() - t0, 3)
        record.status = StageStatus.FAILED if exc else StageStatus.DONE
        record.error = f"{type(exc).__name__}: {exc}" if exc else None
        job.save()

    def _emit(self, job: Job, stage: Stage, status: str, frac: float, message: str) -> None:
        offset, share = self._offsets[stage.name]
        overall = offset + share * max(0.0, min(1.0, frac))
        self.on_event(PipelineEvent(job.id, stage.name, status, round(overall, 4), message))
