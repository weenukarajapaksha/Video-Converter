"""Measure a finished dubbing job against ground truth (or without, for real clips).

    python scripts/evaluate.py job jobs/<id> --truth samples/scenarios/noisy_truth
    python scripts/evaluate.py all              # run every scenario end-to-end, then
                                                # write docs/EVALUATION.md + .json
    python scripts/evaluate.py all --reuse      # re-evaluate job ids recorded earlier

Metrics: ASR character error rate (overall + per line), speech coverage, speaker
count, gender accuracy, silence handling, sync offsets / stretch / overflow,
translation-critic scores, per-stage latency, realtime factor and API cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
DOCS = ROOT / "docs"
RESULTS_JSON = DOCS / "evaluation.json"
SCENARIOS = {
    "two_speakers_music": ROOT / "samples" / "two_speakers_music.mp4",
    **{p.stem: p for p in sorted((ROOT / "samples" / "scenarios").glob("*.mp4"))},
}
REAL_CLIP_JOB = JOBS / "debug-run"

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


# --- metrics (pure functions, unit-tested) ------------------------------------------

def normalize(text: str) -> str:
    """Drop whitespace and punctuation (CER is computed on characters only)."""
    return _PUNCT.sub("", text)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def coverage(truth: list[tuple[float, float]], hyp: list[tuple[float, float]]) -> float:
    """Fraction of ground-truth speech time covered by transcript segments."""
    total = sum(e - s for s, e in truth)
    if total <= 0:
        return 1.0
    hyp_u = _union(hyp)
    covered = sum(_overlap(s, e, hs, he) for s, e in truth for hs, he in hyp_u)
    return min(1.0, covered / total)


def silences(truth: list[tuple[float, float]], duration: float,
             min_len: float = 1.0) -> list[tuple[float, float]]:
    """Gaps between ground-truth lines (including lead-in / tail) of at least min_len."""
    gaps, t = [], 0.0
    for s, e in sorted(truth):
        if s - t >= min_len:
            gaps.append((t, s))
        t = max(t, e)
    if duration - t >= min_len:
        gaps.append((t, duration))
    return gaps


def silence_leaks(gaps: list[tuple[float, float]], dub_lines: list[tuple[float, float]],
                  allow_tail: float = 1.6, guard: float = 0.2) -> list[tuple[float, float]]:
    """Dub lines that start inside a silence, beyond the allowed spill-over.

    The pipeline may let a line run up to 1.5 s past the original speaker's end into
    the following pause (by design), so only audio *starting* in the protected part
    of a silence counts as unnecessary generated speech.
    """
    leaks = []
    for gs, ge in gaps:
        lo, hi = gs + allow_tail, ge - guard
        if hi <= lo:
            continue
        leaks += [(s, e) for s, e in dub_lines if lo < s < hi]
    return leaks


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


# --- job evaluation ------------------------------------------------------------------

def _load(path: Path):
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def evaluate_job(job: Path, truth_dir: Path | None = None) -> dict:
    manifest = _load(job / "manifest.json")
    meta = _load(job / "metadata.json") or {}
    transcript = _load(job / "transcript.json") or {"segments": []}
    translation = _load(job / "translation.json") or {"segments": []}
    track = _load(job / "dub_track.json") or {"lines": []}
    voices = _load(job / "voices.json") or {"speakers": {}}
    asr_stats = _load(job / "asr_stats.json")
    duration = meta.get("duration_s") or 0.0

    segs = transcript["segments"]
    budgets = {s["id"]: s["max_duration_s"] for s in translation["segments"]}
    lines = track["lines"]
    stages = {k: v.get("duration_s") for k, v in manifest["stages"].items()}
    total_s = sum(v or 0 for v in stages.values())
    review = translation.get("review")

    out: dict = {
        "job_id": manifest["job_id"],
        "duration_s": duration,
        "status": {k: v["status"] for k, v in manifest["stages"].items()},
        "lines": len(segs),
        "speakers_detected": len({s["speaker"] for s in segs}),
        "stage_s": stages,
        "total_s": round(total_s, 1),
        "rtf": round(total_s / duration, 2) if duration else None,
        "cost_usd": round(sum(c["usd"] for c in manifest.get("costs", [])), 4),
        "translation_model": translation.get("model"),
        "critic": ({k: review[k] for k in ("accuracy", "naturalness", "tone")}
                   | {"revisions": len(review.get("issues", []))}) if review else None,
        "sync": {
            "stretched_pct": round(100 * sum(ln["tempo"] > 1.0 for ln in lines) / len(lines), 1)
            if lines else 0.0,
            "max_tempo": max((ln["tempo"] for ln in lines), default=1.0),
            "rewritten": sum(1 for ln in lines if ln["rewrites"]),
            "overflow": sum(1 for ln in lines
                            if ln["final_duration_s"] > budgets.get(ln["id"], 1e9) + 0.05),
        },
        "asr_stats": asr_stats,
    }

    if truth_dir is None:
        return out

    truth = _load(truth_dir / "script.json")
    tlines = truth["lines"]
    t_spans = [(ln["start"], ln["end"]) for ln in tlines]
    ref_text = "".join(ln["text"] for ln in tlines)
    hyp_text = "".join(s["text"] for s in segs)

    # Per-line CER: words whose midpoint falls inside the truth line (+-0.3 s).
    words = [w for s in segs for w in s.get("words", [])]
    per_line = []
    for ln in tlines:
        hyp = "".join(w["text"] for w in words
                      if ln["start"] - 0.3 <= (w["start"] + w["end"]) / 2 <= ln["end"] + 0.3)
        per_line.append(round(cer(ln["text"], hyp), 3))

    # Speaker -> truth speaker by max overlap; gender accuracy per detected speaker.
    truth_speakers = {ln["speaker"] for ln in tlines}
    gender_of = {ln["speaker"]: ln.get("gender", ln["speaker"]) for ln in tlines}
    correct = judged = 0
    for spk, prof in voices["speakers"].items():
        spans = [(s["start"], s["end"]) for s in segs if s["speaker"] == spk]
        best = max(truth_speakers, key=lambda ts: sum(
            _overlap(a, b, ln["start"], ln["end"]) for a, b in spans
            for ln in tlines if ln["speaker"] == ts), default=None)
        if best is not None:
            judged += 1
            correct += prof["gender"] == gender_of[best]

    # Sync vs ground truth: first dub line overlapping each truth line.
    offsets = []
    for ln in tlines:
        hits = [d["start"] for d in lines
                if _overlap(d["start"], d["start"] + d["final_duration_s"], ln["start"], ln["end"])]
        if hits:
            offsets.append(abs(min(hits) - ln["start"]))

    gaps = silences(t_spans, truth["duration_s"])
    dub_spans = [(d["start"], d["start"] + d["final_duration_s"]) for d in lines]
    leaks = silence_leaks(gaps, dub_spans)
    silence_db = None
    if (job / "dub_voice.wav").exists() and gaps:
        audio, sr = sf.read(job / "dub_voice.wav", dtype="float32")
        protected = [audio[int((s + 1.6) * sr):int((e - 0.2) * sr)] for s, e in gaps
                     if e - 0.2 > s + 1.6]
        if protected:
            silence_db = round(rms_dbfs(np.concatenate(protected)), 1)

    out.update({
        "cer": round(cer(ref_text, hyp_text), 4),
        "cer_per_line": per_line,
        "coverage": round(coverage(t_spans, [(s["start"], s["end"]) for s in segs]), 3),
        "speakers_truth": len(truth_speakers),
        "gender_accuracy": round(correct / judged, 3) if judged else None,
        "sync_vs_truth": {
            "mean_start_offset_s": round(float(np.mean(offsets)), 3) if offsets else None,
            "max_start_offset_s": round(float(np.max(offsets)), 3) if offsets else None,
            "truth_lines_voiced": f"{len(offsets)}/{len(tlines)}",
        },
        "silences": {"count": len(gaps), "leaked_lines": len(leaks),
                     "dub_dbfs_in_silence": silence_db},
    })
    return out


# --- running scenarios ---------------------------------------------------------------

def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    exe = ROOT / ".venv" / "Scripts" / "dubber.exe"
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "dubber.cli"]
    return subprocess.run([*cmd, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def run_scenario(video: Path) -> tuple[str | None, str]:
    before = {p.name for p in JOBS.glob("*")} if JOBS.exists() else set()
    proc = _cli("run", str(video))
    new = sorted({p.name for p in JOBS.glob("*") if p.is_dir()} - before)
    job_id = new[-1] if new else None
    note = "ok"
    if proc.returncode != 0 and job_id:
        retry = _cli("resume", job_id)  # one recovery attempt, as a user would
        note = "ok after 1 resume" if retry.returncode == 0 else \
            f"FAILED: {(retry.stderr or retry.stdout).strip().splitlines()[-1][:200]}"
    elif proc.returncode != 0:
        note = f"FAILED: {(proc.stderr or proc.stdout).strip().splitlines()[-1][:200]}"
    return job_id, note


def _fmt(v, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


def write_report(results: dict, real: dict | None) -> None:
    DOCS.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps({"scenarios": results, "real_clip": real},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name, r in results.items():
        if "error" in r:
            rows.append(f"| {name} | {r['note']} |" + " — |" * 11)
            continue
        c = r["critic"] or {}
        rows.append(
            f"| {name} | {r['duration_s']:.0f}s | {100 * r['cer']:.1f}% | "
            f"{100 * r['coverage']:.0f}% | {r['speakers_detected']}/{r['speakers_truth']} | "
            f"{_fmt(r['gender_accuracy'] and round(100 * r['gender_accuracy']), '%')} | "
            f"{r['silences']['leaked_lines']} | "
            f"{_fmt(r['sync_vs_truth']['mean_start_offset_s'], 's')} | "
            f"{r['sync']['stretched_pct']}% / ×{r['sync']['max_tempo']} | "
            f"{r['sync']['overflow']} | "
            f"{c.get('accuracy', '—')}/{c.get('naturalness', '—')}/{c.get('tone', '—')} | "
            f"{r['total_s']}s ({r['rtf']}×) | ${r['cost_usd']} |")
    stage_names = ["ingest", "separate", "transcribe", "diarize", "voices", "translate",
                   "synthesize", "mix", "render"]
    lat_rows = [f"| {n} | " + " | ".join(_fmt(r.get("stage_s", {}).get(s)) for s in stage_names)
                + " |" for n, r in results.items() if "error" not in r]

    real_md = "_jobs/debug-run not found._"
    if real:
        a = real.get("asr_stats") or {}
        real_md = "\n".join([
            f"- Duration {real['duration_s']}s, {real['lines']} dubbed lines, "
            f"{real['speakers_detected']} speakers detected.",
            f"- ASR: {a.get('vad_speech_regions', '—')} VAD speech regions, "
            f"{a.get('recovered_regions', '—')} re-transcribed by the coverage check, "
            f"low-confidence segments: {a.get('low_confidence_segments', '—')}.",
            f"- Sync: {real['sync']['stretched_pct']}% of lines stretched, max ×"
            f"{real['sync']['max_tempo']}, {real['sync']['rewritten']} rewritten by the LLM, "
            f"{real['sync']['overflow']} overflowing their budget.",
            f"- Critic: {real['critic']}; translation model {real['translation_model']}.",
            f"- Latency {real['total_s']}s total ({real['rtf']}× realtime), "
            f"API cost ${real['cost_usd']}. Stages: "
            + ", ".join(f"{k} {v}s" for k, v in real["stage_s"].items()),
        ])

    md = f"""# Evaluation

Automated, reproducible measurements of the full pipeline on the assessment's test
scenarios. Every number below is produced by `python scripts/evaluate.py all`.

## Methodology

* **Test material** — `scripts/make_scenarios.py` / `scripts/make_test_video.py` render
  synthetic Mandarin dialogue with Microsoft neural voices (male `Yunxi`, `Yunjian`;
  female `Xiaoxiao`), optional synthetic music and white noise, and save **ground truth**
  next to each video: exact script, per-line timing, speaker identity and gender, and
  the clean speech / music stems.
* **ASR CER** — character error rate (Levenshtein distance / reference length) after
  removing punctuation and whitespace, over the whole transcript; per-line CER uses
  ASR words whose midpoint lies within the truth line (±0.3 s).
* **Coverage** — fraction of ground-truth speech time covered by transcript segments
  (catches silently skipped dialogue).
* **Speakers / gender** — detected speaker count vs truth; each detected speaker is
  mapped to the truth speaker it overlaps most, and its pitch-based gender is checked.
* **Silence leaks** — dub lines that *start* inside a ground-truth silence (≥1 s),
  beyond the 1.5 s spill-over the timing model deliberately allows. Should be 0.
  Also the dub-track level inside those protected silences (dBFS, −inf = digital silence).
* **Sync** — |dub start − truth line start| for the first dub line overlapping each
  truth line; share of lines time-stretched and the max tempo; lines whose final
  duration exceeds their budget (overflow).
* **Critic** — accuracy / naturalness / tone (1–10) from the LLM-as-judge pass.
* **Latency / cost** — per-stage wall time from the job manifest (RTX 3050 Laptop 4 GB,
  models already cached), realtime factor = processing time / video duration; API cost
  at paid-tier list prices (Edge TTS is free).

## Results

| Scenario | Len | CER | Coverage | Speakers (det/true) | Gender acc. | Silence leaks | Mean start offset | Stretched / max tempo | Overflow | Critic A/N/T | Time (RTF) | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

### Per-stage latency (seconds)

| Scenario | {" | ".join(stage_names)} |
|---|{"---|" * len(stage_names)}
{chr(10).join(lat_rows)}

Job ids, per-line CER and every raw metric: [`evaluation.json`](evaluation.json).

## Real clip (no ground truth)

`jobs/debug-run` — a 50 s Mandarin film excerpt supplied by the user.

{real_md}

## Observations

{OBSERVATIONS}
"""
    (DOCS / "EVALUATION.md").write_text(md, encoding="utf-8")


OBSERVATIONS = "_(filled in after reviewing the run)_"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("job_dir", type=Path)
    j.add_argument("--truth", type=Path)
    a = sub.add_parser("all")
    a.add_argument("--reuse", action="store_true", help="re-evaluate recorded job ids")
    a.add_argument("--only", nargs="*", help="scenario names")
    args = ap.parse_args()

    if args.cmd == "job":
        print(json.dumps(evaluate_job(args.job_dir, args.truth), ensure_ascii=False, indent=2))
        return

    previous = (_load(RESULTS_JSON) or {}).get("scenarios", {})
    results = {}
    for name, video in SCENARIOS.items():
        if args.only and name not in args.only:
            if name in previous:
                results[name] = previous[name]
            continue
        truth = video.with_name(video.stem + "_truth")
        if args.reuse and previous.get(name, {}).get("job_id"):
            job_id, note = previous[name]["job_id"], previous[name].get("note", "ok")
        else:
            print(f"running {name} ...", flush=True)
            job_id, note = run_scenario(video)
        if not job_id or not (JOBS / job_id / "dub_track.json").exists():
            results[name] = {"job_id": job_id, "note": note, "error": True}
        else:
            results[name] = evaluate_job(JOBS / job_id, truth) | {"note": note}
        r = results[name]
        print(f"  {name}: {note}" + ("" if "error" in r else
              f" CER={r['cer']:.3f} cov={r['coverage']} spk={r['speakers_detected']}/"
              f"{r['speakers_truth']} total={r['total_s']}s"), flush=True)
    real = evaluate_job(REAL_CLIP_JOB) if (REAL_CLIP_JOB / "manifest.json").exists() else None
    write_report(results, real)
    print(f"wrote {DOCS / 'EVALUATION.md'}")


if __name__ == "__main__":
    main()
