"""Task 3: context-aware, duration-aware translation with a self-critique pass.

1. Draft: the whole conversation is adapted line by line under word budgets.
2. Critique: a second LLM pass acts as a QA reviewer (LLM-as-judge). It scores
   accuracy / naturalness / tone and proposes fixes for flagged lines only.
3. Revise: fixes that respect the time budget replace the draft lines. The
   review (scores + every before/after change) is saved for the report and UI.
"""

import logging

from dubber import llm
from dubber.align import time_budgets
from dubber.artifacts import TRANSCRIPT, TRANSLATION
from dubber.config import get_settings
from dubber.errors import DubberError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import (Transcript, TranslatedSegment, Translation, TranslationIssue,
                            TranslationReview)
from dubber.stages.ingest import IngestStage
from dubber.translate import LineOut, critique, translate_lines

log = logging.getLogger(__name__)


class TranslateStage(Stage):
    name = "translate"
    outputs = (TRANSLATION,)
    weight = 1.0
    max_attempts = 4  # API rate limits are common on free tiers

    def run(self, job: Job, progress: ProgressFn) -> None:
        settings = get_settings()
        model = job.option("translation_model", settings.translation_model)
        target = job.option("target_language", settings.target_language)
        segments = job.read_model(TRANSCRIPT, Transcript).segments
        budgets = time_budgets(segments, IngestStage.metadata(job).duration_s)

        usage = llm.Usage()
        lines: dict[int, LineOut] = {}
        review: TranslationReview | None = None
        if segments:
            lines = translate_lines(
                segments, budgets, model, target, usage,
                on_chunk=lambda done, total: progress(0.6 * done / total,
                                                      f"drafted {done}/{total} lines"))
            if job.option("critic", True):
                progress(0.65, "critic reviewing translation")
                review = self._review(segments, budgets, lines, model, target, usage)

        job.add_cost(self.name, "gemini", usage.model or model,
                     usage.input_tokens + usage.output_tokens, "tokens", usage.cost(model))
        job.write_model(TRANSLATION, Translation(
            target_language=target, model=usage.model or model, review=review,
            segments=[TranslatedSegment(
                id=s.id, start=s.start, end=s.end, speaker=s.speaker, source_text=s.text,
                text=lines[s.id].text, emotion=lines[s.id].emotion, max_duration_s=b,
            ) for s, b in zip(segments, budgets)],
        ))
        if review:
            progress(1.0, f"scores A{review.accuracy}/N{review.naturalness}/T{review.tone}, "
                          f"{len(review.issues)} line(s) revised")

    @staticmethod
    def _review(segments, budgets, lines, model, target, usage) -> TranslationReview | None:
        try:
            out = critique(segments, budgets, lines, model, target, usage)
        except DubberError as exc:  # the draft is usable on its own; don't fail the job
            log.warning("translation critic skipped: %s", exc)
            return None
        issues = []
        for issue in out.issues:
            before = lines[issue.id].text
            lines[issue.id] = lines[issue.id].model_copy(update={"text": issue.revised_text})
            issues.append(TranslationIssue(id=issue.id, category=issue.category,
                                           comment=issue.comment, before=before,
                                           after=issue.revised_text))
        return TranslationReview(accuracy=out.accuracy, naturalness=out.naturalness,
                                 tone=out.tone, summary=out.summary, issues=issues)
