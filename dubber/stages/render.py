"""Task 6: mux the dubbed soundtrack (and English subtitles) into the final MP4."""

from __future__ import annotations

from dubber import media
from dubber.artifacts import (DUB_MIX, DUB_TRACK, OUTPUT_VIDEO, REPORT, SUBTITLES_EN,
                              SUBTITLES_ZH, TRANSCRIPT, TRANSLATION)
from dubber.errors import MediaError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import DubTrack, Transcript, Translation, VideoMetadata
from dubber.stages.ingest import IngestStage
from dubber.subtitles import to_srt

LANG_CODES = {"english": "eng", "spanish": "spa", "french": "fra", "japanese": "jpn",
              "german": "deu", "korean": "kor", "sinhala": "sin", "hindi": "hin"}


def video_codec_args(meta: VideoMetadata) -> list[str]:
    """Stream-copy H.264 (bit-exact visuals, zero generation loss); else re-encode."""
    if meta.video_codec == "h264" and meta.pix_fmt in ("yuv420p", "yuvj420p"):
        return ["-c:v", "copy"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]


class RenderStage(Stage):
    name = "render"
    outputs = (OUTPUT_VIDEO, SUBTITLES_EN, SUBTITLES_ZH, REPORT)
    weight = 0.5

    def run(self, job: Job, progress: ProgressFn) -> None:
        meta = IngestStage.metadata(job)
        translation = job.read_model(TRANSLATION, Translation)
        transcript = job.read_model(TRANSCRIPT, Transcript)
        track = {ln.id: ln for ln in job.read_model(DUB_TRACK, DubTrack).lines}

        # English cues follow the *dubbed audio* timing, not the Chinese timing.
        job.path(SUBTITLES_EN).write_text(to_srt(
            (s.start, s.start + track[s.id].final_duration_s if s.id in track else s.end, s.text)
            for s in translation.segments), encoding="utf-8")
        job.path(SUBTITLES_ZH).write_text(to_srt(
            ((s.start, s.end, s.text) for s in transcript.segments), cjk=True), encoding="utf-8")

        progress(0.3, "muxing video")
        lang = LANG_CODES.get(translation.target_language.lower(), "und")
        # Soft (toggleable) subtitle track, only when there is dialogue.
        subs = (["-i", job.path(SUBTITLES_EN)], ["-map", "2:s:0", "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={lang}"]) if translation.segments else ([], [])
        media.run_ffmpeg(
            "-i", job.input_video, "-i", job.path(DUB_MIX), *subs[0],
            "-map", "0:v:0", "-map", "1:a:0", *subs[1],
            *video_codec_args(meta),
            "-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", f"language={lang}",
            "-t", f"{meta.video_duration_s:.3f}", "-movflags", "+faststart",
            job.path(OUTPUT_VIDEO),
        )

        out = media.probe(job.path(OUTPUT_VIDEO))
        drift = abs(out.duration_s - meta.duration_s)
        if drift > 0.25:
            raise MediaError(f"output duration {out.duration_s:.2f}s differs from input "
                             f"{meta.duration_s:.2f}s")
        job.write_json(REPORT, build_report(job, out, track))
        progress(1.0, f"{out.resolution} {out.video_codec}/{out.audio_codec}, "
                      f"Δduration {drift * 1000:.0f} ms")


def build_report(job: Job, out: VideoMetadata, track: dict) -> dict:
    m = job.manifest
    lines = list(track.values())
    return {
        "job_id": m.job_id,
        "input": m.input_name,
        "output": {"duration_s": out.duration_s, "video_codec": out.video_codec,
                   "audio_codec": out.audio_codec, "resolution": out.resolution, "fps": out.fps},
        "sync": {
            "lines": len(lines),
            "rewritten_lines": sum(1 for ln in lines if ln.rewrites),
            "stretched_lines": sum(1 for ln in lines if ln.tempo > 1.0),
            "max_tempo": max((ln.tempo for ln in lines), default=1.0),
        },
        "stages": {name: {"status": r.status, "duration_s": r.duration_s, "attempts": r.attempts}
                   for name, r in m.stages.items()},
        "costs": [c.model_dump() for c in m.costs],
        "total_cost_usd": m.total_cost_usd,
        "total_processing_s": m.total_duration_s,
    }
