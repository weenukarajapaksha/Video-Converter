"""Command-line entry point: ``dubber probe`` / ``dubber run`` / ``dubber resume``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from dubber import media
from dubber.config import get_settings
from dubber.pipeline import Job, Pipeline, PipelineEvent
from dubber.stages import build_stages

app = typer.Typer(add_completion=False, help="AI video dubbing: Chinese -> English")
console = Console()


@app.command()
def probe(video: Path) -> None:
    """Print video metadata."""
    meta = media.probe(video)
    table = Table(show_header=False)
    for key, value in meta.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


def _execute(job: Job, force: list[str]) -> None:
    with Progress(TextColumn("[bold]{task.description}"), BarColumn(),
                  TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(),
                  TextColumn("{task.fields[msg]}"), console=console) as bar:
        task = bar.add_task("starting", total=1.0, msg="")

        def on_event(e: PipelineEvent) -> None:
            bar.update(task, completed=e.progress, description=e.stage, msg=e.message)
            if e.status in {"done", "skipped", "failed", "retrying"}:
                bar.console.log(f"{e.stage}: {e.status} {e.message}")

        Pipeline(build_stages(), on_event=on_event).run(job, force=force)

    m = job.manifest
    console.print(f"[green]Job {m.job_id} finished[/] in {m.total_duration_s:.1f}s, "
                  f"API cost ${m.total_cost_usd:.4f}  ->  {job.root}")


@app.command()
def run(video: Path, verbose: bool = typer.Option(False, "-v")) -> None:
    """Dub a video end-to-end."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    job = Job.create(get_settings().jobs_dir, video)
    _execute(job, [])


@app.command()
def resume(
    job_id: str,
    force: list[str] = typer.Option([], "--force", "-f", help="stage(s) to re-run"),
) -> None:
    """Resume a failed/interrupted job; completed stages are skipped."""
    _execute(Job(get_settings().jobs_dir / job_id), force)


if __name__ == "__main__":
    app()
