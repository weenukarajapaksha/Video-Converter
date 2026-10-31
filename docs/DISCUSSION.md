# Engineering Discussion Questions

Each answer separates **what this codebase does** from **what production would add**.

---

### 1. How do you keep sync when the English is longer than the Chinese?

The fix is layered, and starts before any audio exists.

1. **Budget at translation time.** Each line's time budget is its slot plus the following silence, minus a 0.15 s pause, and never more than 1.5 s past the speaker's end. That budget becomes `max_words` at 2.6 words per second, which the LLM must respect. The critic's fixes are rejected if they exceed it.
2. **Measure after TTS.** Real speech duration is measured after trimming silence.
3. **Rewrite if it's far over.** A line more than 12% over budget is sent back to the LLM to be shortened. All such lines go in one batched request. There are at most 2 rounds.
4. **Stretch if it's slightly over.** The remaining overrun is time-stretched with Rubber Band, preserving pitch and formants, **capped at 1.25×**. Above about 1.25× speech sounds unnatural, which is why rewriting comes first.
5. **Place each line at its original start time.** Overruns never push later lines, so there's no cumulative drift.

**Production would add** phoneme- and viseme-aware phrasing, and letting a line borrow time from a *preceding* pause, not only the following one.

### 2. How do you distinguish speakers and keep their voices consistent across scenes?

**Here:**
- pyannote 3.1 labels who speaks when. Each ASR *word* is assigned to the turn it overlaps most. One-word "blips" are smoothed away, speaker changes snap to nearby punctuation, and lines are re-cut per speaker.
- **Over-split speakers are merged.** pyannote returns a speaker embedding per cluster. Two clusters merge when their embeddings are similar (cosine ≥ 0.40) **and** their pitch is close (within ~3.5 semitones). Embeddings alone weren't enough: a real man/woman pair scored 0.44, the same as two halves of one speaker, but their pitch differed by ~8 semitones. Tiny clusters (<1.5 s) join the most similar same-gender speaker. Real clip: 5 → 3 speakers; synthetic two-person clip stays at 2. Every decision is logged in `diarization_meta.json`.
- Each speaker's gender is estimated from median pitch (pYIN; below 160 Hz is male). Each speaker then gets the unused same-gender voice **closest in pitch** (Edge voice pitches were measured), so a 112 Hz speaker gets a 117 Hz voice. The mapping lives in `voices.json` and stays fixed for the whole video.
- The user can fix the count at upload (`num_speakers`) or re-cast in the UI, previewing each voice before choosing.

**Across scenes or episodes in production:**
- Store pyannote speaker *embeddings* per character, match new clusters by cosine similarity, and keep a persistent registry from character to voice.
- Use consent-based voice cloning for actor timbre.

### 3. How do you keep music and sound effects while removing the Chinese speech?

**Here:**
- Demucs `htdemucs_ft` splits the soundtrack into vocals and accompaniment. The background is computed as mix − vocals, which is exactly complementary to the vocal stem.
- The English goes onto that background, level-matched to the original dialogue's loudness.
- The background is ducked by −4 dB with 80 ms ramps under English lines, which masks residual Chinese bleed.
- Measured against ground truth: vocal SDR 22.1 dB, music SDR 20.3 dB.

**Production would add:**
- Band-split RoFormer or MDX ensembles, which score higher SDR.
- A dialogue/effects/music three-stem split, so effects that sound like voices (screams, crowd noise) aren't removed.
- Using the studio's M&E stems (separate music-and-effects tracks) when they're available.

### 4. How would you reduce the cost of processing 1,000 videos a day?

See [COST_ANALYSIS.md](COST_ANALYSIS.md#scaling-to-1000-videos-per-day). The current estimate is about $76 a day.

1. Long-lived GPU workers with models kept loaded (no ~20 s load per job).
2. Batch or flex API pricing for translation, which isn't latency-critical in a queue.
3. Run the critic only when it's likely to help (flagged lines or low ASR confidence), and use lite models for easy lines.
4. `htdemucs` instead of `htdemucs_ft` on throughput tiers (3.5× faster).
5. Batched ASR, spot GPUs (safe, since stages checkpoint), and caching of TTS and translations (TTS is already content-hashed).

### 5. What would you change to process a 60-minute video reliably?

**Already in place:**
- Checkpointing per stage, with resume.
- Translation in 60-line windows with context on each side.
- Separation in 60 s chunks.
- The ASR coverage check, which matters more as videos get longer.
- Fast API failover.

**What I'd add:**
1. **Chunk the job** at long silences or scene cuts, about 5 minutes each with overlap, so each chunk checkpoints independently and a crash loses minutes, not hours.
2. **Run chunks in parallel** across GPU workers.
3. **Keep continuity across chunks:** a per-video glossary of names and terms, a summary carried between translation windows, and speaker re-identification across chunks by embedding.
4. **Store artifacts in object storage** and stream them, instead of keeping hour-long WAVs in RAM.
5. **Timeouts and heartbeats** per stage, with progress persisted inside long stages.

### 6. What's the difference between speech transcription, forced alignment and speaker diarization?

| | Input | Output | Question it answers | Here |
|---|---|---|---|---|
| **Transcription (ASR)** | audio | text (+ rough timestamps) | *What* was said? | Whisper large-v3 |
| **Forced alignment** | audio **and known text** | precise start and end of each word or phoneme | *Exactly when* was each word said? | Approximated with Whisper's cross-attention word timestamps. WhisperX's wav2vec2 aligner was considered and rejected (§3 of TECHNICAL.md). |
| **Speaker diarization** | audio | time spans labelled SPEAKER_n, no text | *Who* spoke when? | pyannote 3.1 |

The pipeline combines all three. Word-level timestamps let diarization turns be mapped onto the text, so each dubbing line has one speaker and accurate timing.

### 7. How would you evaluate translation quality, automatically and manually?

**Automatic, implemented here:**
- An LLM-as-judge critic pass scores **accuracy, naturalness and tone** from 1 to 10 and lists categorized issues with before/after fixes, stored per job. A fix is accepted only if it still fits the line's time budget. Real clip: 8/8/8 with revisions; synthetic clip: 10/10/10, with nothing to change.
- **ASR-error tracking:** the translator reports each homophone it repaired from context (`asr_fix`: 丹代→唐代, 扛够→唐构), so translation quality can be read separately from ASR quality.
- **Constraint metrics** are also recorded: lines over budget, rewrite counts, and stretch factors. The quality panel (`/quality`) turns them into warnings.
- **Against ground truth:** `scripts/evaluate.py` measures ASR CER and sync on synthetic scenarios with known scripts.

**Automatic, for production:**
- A reference-free metric that can be cited: MetricX-24 (QE mode) or COMET-Kiwi. Never use the same metric both to pick candidates and to report results.
- An MQM-style critic (error spans with category and severity) from a *different* model family than the translator.
- Back-translation similarity.
- Terminology consistency checks against a glossary.
- Speech-level checks: ASR the English dub and compare it to the script, to catch TTS mispronunciations.

**Manual:**
- Bilingual reviewers rate adequacy, fluency and lip-sync naturalness on a fixed scale over a sample.
- The script editor captures their edits, and the edit distance between the LLM draft and the final text tracks quality over time.

### 8. How would you handle overlapping dialogue?

**Here:**
- pyannote marks overlapping turns, and each line keeps its own speaker.
- `time_budgets` lets lines that overlap in the source keep their own duration, so they're placed overlapping in the dub, just as in the source.
- Overlap isn't created anywhere else, because every non-overlapping line is budgeted to end before the next one starts.

**Limitation:** the ASR transcript of overlapped speech is often only the dominant voice.

**Production would add:**
- Overlap-aware separation per speaker before ASR, using a speech-separation model.
- Lowering the secondary speaker's dub level, to match how mixes are normally balanced.

### 9. How would you build a retry mechanism for failed pipeline stages?

This is how the runner in [`dubber/pipeline/`](../dubber/pipeline) works:

1. **State on disk.**
   - Each stage writes its declared outputs into the job folder.
   - `manifest.json` records status, attempt count, timings, error and cost, and is written atomically.
   - A stage counts as complete only if it's marked `done` **and** its output files exist.
2. **Error classification.**
   - `TransientError` (HTTP 408/429/5xx, timeouts, network failures) is retried with exponential backoff, using a per-stage `max_attempts`: translate 4, synthesize 3.
   - Validation and configuration errors (bad input, gated model, missing key) fail immediately with an actionable message.
3. **Resume.**
   - `dubber resume <job>`, or the UI's Retry button, skips completed stages.
   - Once a stage re-runs, all downstream stages re-run too. `--force <stage>` invalidates from a chosen point.
4. **Retries inside a stage.**
   - LLM calls walk a fallback chain of models, with a **circuit breaker** that skips an overloaded model for 2 minutes.
   - TTS output is content-hash cached, so retrying synthesis re-pays only for lines that failed.
5. **Idempotency and cost.** Stage reruns overwrite their outputs, and the stage's cost entries are cleared at the start of each attempt, so cost isn't double-counted.

**Production would add:**
- A durable queue (Celery or Arq) with visibility timeouts and a dead-letter queue.
- Per-stage timeouts and heartbeats.
- Alerting on retry and fallback rates.
- Idempotency keys for external calls.
