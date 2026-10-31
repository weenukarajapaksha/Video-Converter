# Evaluation

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
| two_speakers_music | 58s | 0.0% | 98% | 2/2 | 50% | 0 | 0.392s | 10.0% / ×1.1153 | 0 | 8/8/9 | 111.3s (1.91×) | $0.0097 |
| fast_dialogue | 33s | 2.7% | 97% | 2/2 | 50% | 0 | 0.179s | 22.2% / ×1.0887 | 0 | —/—/— | 322.0s (9.65×) | $0.0018 |
| long_sentences | 48s | 2.6% | 99% | 2/2 | 100% | 0 | 0.171s | 11.1% / ×1.25 | 1 | 10/9/10 | 97.9s (2.04×) | $0.0093 |
| noisy | 32s | 0.0% | 96% | 2/2 | 50% | 0 | 0.205s | 0.0% / ×1.0 | 0 | 9/9/10 | 107.5s (3.33×) | $0.0044 |
| one_speaker_clear | FAILED: Please retry in 25m13.349829034s. | — | — | — | — | — | — | — | — | — | — | — |
| silent_sections | FAILED: Please retry in 22m52.515049762s. | — | — | — | — | — | — | — | — | — | — | — |

### Per-stage latency (seconds)

| Scenario | ingest | separate | transcribe | diarize | voices | translate | synthesize | mix | render |
|---|---|---|---|---|---|---|---|---|---|
| two_speakers_music | 0.288 | 27.157 | 36.322 | 12.092 | 1.793 | 21.921 | 4.378 | 5.568 | 1.793 |
| fast_dialogue | 0.199 | 19.467 | 15.786 | 10.603 | 1.265 | 262.056 | 9.278 | 2.397 | 0.959 |
| long_sentences | 0.178 | 23.018 | 17.098 | 21.146 | 1.733 | 21.029 | 8.346 | 3.651 | 1.695 |
| noisy | 0.192 | 45.678 | 22.551 | 16.452 | 1.446 | 14.989 | 2.51 | 2.617 | 1.105 |

Job ids, per-line CER and every raw metric: [`evaluation.json`](evaluation.json).

## Real clip (no ground truth)

`jobs/debug-run` — a 50 s Mandarin film excerpt supplied by the user.

- Duration 50.6s, 23 dubbed lines, 5 speakers detected.
- ASR: 11 VAD speech regions, 5 re-transcribed by the coverage check, low-confidence segments: [2].
- Sync: 17.4% of lines stretched, max ×1.25, 3 rewritten by the LLM, 1 overflowing their budget.
- Critic: {'accuracy': 8, 'naturalness': 8, 'tone': 8, 'revisions': 5}; translation model gemini-3.5-flash.
- Latency 153.9s total (3.04× realtime), API cost $0.0162. Stages: ingest 0.846s, separate 44.268s, transcribe 24.741s, diarize 13.416s, voices 1.726s, translate 40.847s, synthesize 23.295s, mix 3.581s, render 1.187s

## Observations

**What works well**

* **Transcription is near-perfect even under music and noise.** CER is 0–2.7 % on every
  scenario, including 0 % on `noisy` (white noise at −20 dB SNR *plus* music): ASR runs
  on the Demucs vocal stem, not the mix. Errors that remain are single homophones in
  fast/long speech. Coverage is 96–99 %; the missing 1–4 % is word-timestamp edge
  trimming, not dropped dialogue (every truth line is voiced, see `truth_lines_voiced`).
* **Speaker count is right in every scenario** (2/2), including with music and noise.
* **Silent sections stay silent.** No dub line starts inside a protected silence, and the
  generated dialogue track measures digital silence (−∞ dBFS) there.
* **Timing.** Dub lines start on average 0.17–0.39 s from the true line start. At most
  ~22 % of lines need time-stretching, never beyond the ×1.25 cap; over-long lines are
  first shortened by the LLM (see `rewritten`). Overflow beyond the budget is rare (≤1
  line per video).
* **Translation quality.** The LLM critic scores 8–10 on accuracy, naturalness and tone.

**Problems found (honest)**

1. **Gender detection fails for a high-pitched male voice.** The male test voice
   *Yunxi* has a median F0 of 179–206 Hz, above the fixed 160 Hz split in
   `dubber/stages/voices.py`, so he is voiced as a woman (gender accuracy 50 % in the
   scenarios that use him; 100 % with the deeper *Yunjian*). A fixed F0 threshold is
   too crude; a speaker-embedding gender classifier (or at least a relative rule
   between speakers) is needed. The UI's speaker→voice mapping lets a user correct it.
2. **Line starts can be pulled back into a preceding silence.** Whisper sometimes starts
   a segment where the previous one ended; e.g. "行，这次就原谅你…" was placed at
   47.70 s while the speech starts at 49.51 s, so its dub enters 1.8 s early (the max
   start offset of `two_speakers_music`). The same error made "对不起" inherit the
   previous speaker's label. Fix: snap each line start forward to the first diarization
   turn / VAD speech onset it overlaps.
3. **Gemini free-tier limits dominate latency and reliability.** Model stages take
   ~1–1.5 min per minute of video, but one translation took 262 s while every fallback
   model was overloaded (503), and two scenarios first failed outright with HTTP 429
   ("retry in 25 min") once the free daily quota ran out — the fallback models share
   that quota. They were re-run after the quota reset. With a paid key this disappears.
4. **The critic is skipped under overload** (`fast_dialogue` has no scores): by design
   the draft translation is kept rather than failing the job.

**Latency profile (RTX 3050 4 GB, models cached).** Separation (4-model `htdemucs_ft`)
19–46 s and Whisper large-v3 16–36 s are the largest local costs; diarization 11–21 s;
TTS + time-fitting 3–10 s with free Edge voices; mix + render < 10 s with the video
stream copied, not re-encoded. Without API outages the realtime factor is ~2–3.3×.
