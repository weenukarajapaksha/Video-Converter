"""A job is a self-contained workspace directory plus a JSON manifest.

Every stage writes its artifacts into the job directory and records its status,
timing and API cost in ``manifest.json``. Because state lives on disk, a crashed or
failed run can be resumed later and completed stages are skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

M = TypeVar("M", bound=BaseModel)

MANIFEST = "manifest.json"
INPUT_VIDEO = "input.mp4"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    error: str | None = None


class CostEntry(BaseModel):
    stage: str
    provider: str
    item: str
    quantity: float
    unit: str
    usd: float


class Manifest(BaseModel):
    job_id: str
    created_at: datetime = Field(default_factory=_now)
    input_name: str
    options: dict = Field(default_factory=dict)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    costs: list[CostEntry] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.usd for c in self.costs), 6)

    @property
    def total_duration_s(self) -> float:
        return round(sum(r.duration_s or 0.0 for r in self.stages.values()), 3)


class Job:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = Manifest.model_validate_json((self.root / MANIFEST).read_text("utf-8"))

    @classmethod
    def create(cls, jobs_dir: Path, video: Path, options: dict | None = None) -> Job:
        job_id = f"{_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        root = Path(jobs_dir) / job_id
        root.mkdir(parents=True)
        shutil.copy2(video, root / INPUT_VIDEO)
        manifest = Manifest(job_id=job_id, input_name=Path(video).name, options=options or {})
        _atomic_write(root / MANIFEST, manifest.model_dump_json(indent=2))
        return cls(root)

    @property
    def id(self) -> str:
        return self.manifest.job_id

    @property
    def input_video(self) -> Path:
        return self.root / INPUT_VIDEO

    def path(self, name: str) -> Path:
        return self.root / name

    def record(self, stage: str) -> StageRecord:
        return self.manifest.stages.setdefault(stage, StageRecord())

    def save(self) -> None:
        _atomic_write(self.root / MANIFEST, self.manifest.model_dump_json(indent=2))

    def add_cost(
        self, stage: str, provider: str, item: str, quantity: float, unit: str, usd: float
    ) -> None:
        self.manifest.costs.append(
            CostEntry(stage=stage, provider=provider, item=item, quantity=quantity, unit=unit,
                      usd=usd)
        )

    def clear_costs(self, stage: str) -> None:
        self.manifest.costs = [c for c in self.manifest.costs if c.stage != stage]

    # -- typed artifact helpers -------------------------------------------------
    def write_model(self, name: str, model: BaseModel) -> Path:
        path = self.path(name)
        _atomic_write(path, model.model_dump_json(indent=2))
        return path

    def read_model(self, name: str, model_type: type[M]) -> M:
        return model_type.model_validate_json(self.path(name).read_text("utf-8"))

    def write_json(self, name: str, data: object) -> Path:
        path = self.path(name)
        _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
        return path


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
