"""Collect the assessment's "Working Demo" deliverables from a finished job.

    python scripts/export_demo.py <job_id> [out_dir]

Creates <out_dir> (default: demo/<job_id>/) containing:
  original_chinese.mp4        - input video
  english_dubbed.mp4          - final dub (H.264 + AAC, soft English subtitles)
  english_dubbed_lipsync.mp4  - lip-synced dub, if lip sync was run
  chinese_transcript.json/.srt- diarized Chinese transcript with timestamps
  english_translation.json/.srt - translation (+ critic review, ASR fixes)
  speaker_mapping.json        - speaker -> gender, pitch, assigned voice
  english_audio_dialogue.wav  - generated English speech only
  english_audio_full_mix.wav  - English speech mixed over the original music/SFX
  report.json                 - per-stage timing, retries, API cost, sync stats
  README.md                   - what each file is, with headline numbers
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubber import artifacts as A  # noqa: E402
from dubber.config import get_settings  # noqa: E402
from dubber.pipeline import Job  # noqa: E402

FILES = [
    (A.INPUT_VIDEO, "original_chinese.mp4"),
    (A.OUTPUT_VIDEO, "english_dubbed.mp4"),
    (A.LIPSYNC_VIDEO, "english_dubbed_lipsync.mp4"),
    (A.TRANSCRIPT, "chinese_transcript.json"),
    (A.SUBTITLES_ZH, "chinese_transcript.srt"),
    (A.TRANSLATION, "english_translation.json"),
    (A.SUBTITLES_EN, "english_translation.srt"),
    (A.VOICES, "speaker_mapping.json"),
    (A.DUB_VOICE, "english_audio_dialogue.wav"),
    (A.DUB_MIX, "english_audio_full_mix.wav"),
    (A.REPORT, "report.json"),
]


def main(job_id: str, out: Path | None = None) -> Path:
    job = Job(get_settings().jobs_dir / job_id)
    out = out or ROOT / "demo" / job_id
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for src, dst in FILES:
        if job.path(src).exists():
            shutil.copy2(job.path(src), out / dst)
            copied.append(dst)
    if "english_dubbed.mp4" not in copied:
        raise SystemExit(f"job {job_id} has no output video yet - finish the dub first")

    m = job.manifest
    voices = json.loads(job.path(A.VOICES).read_text("utf-8"))["speakers"]
    translation = json.loads(job.path(A.TRANSLATION).read_text("utf-8"))
    review = translation.get("review") or {}
    lines = [
        f"# Dub demo: {m.input_name}",
        "",
        f"- Processing time: {m.total_duration_s:.1f} s, API cost: ${m.total_cost_usd:.4f}",
        f"- Lines: {len(translation['segments'])}, speakers: {len(voices)}",
    ]
    if review:
        lines.append(f"- Translation critic: accuracy {review['accuracy']}/10, naturalness "
                     f"{review['naturalness']}/10, tone {review['tone']}/10")
    lines += ["", "## Speaker mapping", "", "| Speaker | Gender | Median pitch | Voice |",
              "|---|---|---|---|"]
    lines += [f"| {s['speaker']} | {s['gender']} | {s.get('median_f0_hz') or '-'} Hz | "
              f"{s['voice']} |" for s in voices.values()]
    lines += ["", "## Files", ""] + [f"- `{name}`" for name in copied]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported {len(copied)} files to {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else None)
