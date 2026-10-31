"""Checkpointing, resume and retry behaviour of the stage runner."""

import pytest

from dubber.errors import InputValidationError, TransientError
from dubber.pipeline import Job, Pipeline, Stage
from dubber.pipeline.job import StageStatus


class Recorder(Stage):
    """Writes one output file; can be told to fail N times first."""

    def __init__(self, name: str, fail_times: int = 0, exc=TransientError, attempts: int = 3):
        self.name = name
        self.outputs = (f"{name}.txt",)
        self.max_attempts = attempts
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0

    def run(self, job, progress):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"boom {self.calls}")
        job.path(f"{self.name}.txt").write_text("ok")
        job.add_cost(self.name, "fake", "call", 1, "call", 0.01)


@pytest.fixture
def job(tmp_path, sample_video):
    return Job.create(tmp_path / "jobs", sample_video)


def test_transient_failures_are_retried_and_costs_not_double_counted(job):
    stage = Recorder("a", fail_times=2)
    Pipeline([stage], backoff_s=0).run(job)

    rec = job.manifest.stages["a"]
    assert stage.calls == 3
    assert rec.status == StageStatus.DONE and rec.attempts == 3
    assert job.manifest.total_cost_usd == pytest.approx(0.01)


def test_permanent_errors_fail_fast_and_are_persisted(job):
    stage = Recorder("a", fail_times=5, exc=InputValidationError)
    with pytest.raises(InputValidationError):
        Pipeline([stage], backoff_s=0).run(job)

    reloaded = Job(job.root)
    assert stage.calls == 1
    assert reloaded.manifest.stages["a"].status == StageStatus.FAILED
    assert "boom 1" in reloaded.manifest.stages["a"].error


def test_resume_skips_completed_stages_and_reruns_downstream(job):
    a, b = Recorder("a"), Recorder("b", fail_times=1, attempts=1)
    with pytest.raises(TransientError):
        Pipeline([a, b], backoff_s=0).run(job)

    # New process: reload the job from disk and resume.
    resumed = Job(job.root)
    a2, b2 = Recorder("a"), Recorder("b")
    Pipeline([a2, b2]).run(resumed)
    assert (a2.calls, b2.calls) == (0, 1)

    # Forcing an upstream stage invalidates everything after it.
    a3, b3 = Recorder("a"), Recorder("b")
    Pipeline([a3, b3]).run(resumed, force=["a"])
    assert (a3.calls, b3.calls) == (1, 1)


def test_deleted_artifact_makes_stage_rerun(job):
    Pipeline([Recorder("a")]).run(job)
    job.path("a.txt").unlink()
    again = Recorder("a")
    Pipeline([again]).run(Job(job.root))
    assert again.calls == 1


def test_progress_events_are_monotonic_per_run(job):
    events = []
    Pipeline([Recorder("a"), Recorder("b")], on_event=events.append).run(job)
    done = [e.progress for e in events if e.status == "done"]
    assert done == [0.5, 1.0]
