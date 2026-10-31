from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from dubber.errors import TransientError
from dubber.pipeline.job import Job, StageStatus

# progress(fraction_of_this_stage in [0, 1], human readable message)
ProgressFn = Callable[[float, str], None]


class Stage(ABC):
    """One step of the pipeline.

    Subclasses declare the artifact files they produce in ``outputs``; a stage is
    considered complete (and skipped on resume) only if its manifest record says
    ``done`` *and* all of those files still exist.
    """

    name: ClassVar[str]
    outputs: ClassVar[tuple[str, ...]] = ()
    # Relative share of total pipeline time, used for the overall progress bar.
    weight: ClassVar[float] = 1.0
    max_attempts: ClassVar[int] = 1
    retry_on: ClassVar[tuple[type[BaseException], ...]] = (TransientError,)

    def is_complete(self, job: Job) -> bool:
        record = job.manifest.stages.get(self.name)
        if record is None or record.status != StageStatus.DONE:
            return False
        return all(job.path(name).exists() for name in self.outputs)

    @abstractmethod
    def run(self, job: Job, progress: ProgressFn) -> None: ...
