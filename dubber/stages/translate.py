"""Task 3: context-aware, duration-aware translation of the transcript."""

from dubber import llm
from dubber.align import time_budgets
from dubber.artifacts import TRANSCRIPT, TRANSLATION
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Transcript, TranslatedSegment, Translation
from dubber.stages.ingest import IngestStage
from dubber.translate import translate_lines


class TranslateStage(Stage):
    name = "translate"
    outputs = (TRANSLATION,)
    weight = 1.0
    max_attempts = 4  # API rate limits are common on free tiers

    def run(self, job: Job, progress: ProgressFn) -> None:
        settings = get_settings()
        model = job.option("translation_model", settings.translation_model)
        target = job.option("target_language", settings.target_language)
        transcript = job.read_model(TRANSCRIPT, Transcript)
        segments = transcript.segments
        budgets = time_budgets(segments, IngestStage.metadata(job).duration_s)

        usage = llm.Usage()
        lines = translate_lines(
            segments, budgets, model, target, usage,
            on_chunk=lambda done, total: progress(done / total, f"{done}/{total} lines"),
        ) if segments else {}
        job.add_cost(self.name, "gemini", model, usage.input_tokens + usage.output_tokens,
                     "tokens", usage.cost(model))

        job.write_model(TRANSLATION, Translation(
            target_language=target, model=model,
            segments=[TranslatedSegment(
                id=s.id, start=s.start, end=s.end, speaker=s.speaker, source_text=s.text,
                text=lines[s.id].text, emotion=lines[s.id].emotion, max_duration_s=b,
            ) for s, b in zip(segments, budgets)],
        ))
