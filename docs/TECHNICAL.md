# Technical Documentation

All timings below were measured on the development machine: an **RTX 3050 Laptop (4 GB VRAM)**, Windows 11, Python 3.12, PyTorch 2.7.1 + CUDA 12.6, FFmpeg 9.

Contents:
1. [Overall system architecture](#1-overall-system-architecture)
2. [AI models and APIs selected](#2-ai-models-and-apis-selected)
3. [Why these models](#3-why-these-models-alternatives-considered)
4. [Challenges encountered](#4-challenges-encountered)
5. [Translation and synchronization techniques](#5-translation-and-synchronization-techniques)
6. [Limitations and known issues](#6-limitations-and-known-issues)
7. [Production improvements](#7-potential-production-improvements)

Also: [Performance & accuracy optimizations](#performance--accuracy-optimizations-measured) · State-of-the-art comparison and upgrade path: [Dubbing pipeline technology review](../reports/Dubbing%20pipeline%20technology%20review.md)

---

## 1. Overall system architecture

### Stages

| # | Stage | Reads | Writes | Model / tool | Retries |
|---|---|---|---|---|---|
| 1 | `ingest` | `input.mp4` | `metadata.json`, `audio_original.wav` | ffprobe, FFmpeg | 1 |
| 2 | `separate` | `audio_original.wav` | `vocals.wav`, `background.wav` | Demucs `htdemucs_ft` (GPU) | 1 |
| 3 | `transcribe` | `vocals.wav` | `transcript_raw.json`, `asr_stats.json` | faster-whisper `large-v3` int8 + Silero VAD (GPU) | 1 |
| 4 | `diarize` | `vocals.wav`, `transcript_raw.json` | `diarization.json`, `diarization_meta.json`, `transcript.json` | pyannote `speaker-diarization-3.1` + embedding/pitch speaker merge (GPU) | 1 |
| 5 | `voices` | `transcript.json`, `vocals.wav` | `voices.json`, `voice_casting.json` | librosa pYIN pitch tracking, pitch-matched casting | 1 |
| 6 | `translate` | `transcript.json`, `metadata.json` | `translation.json` | Gemini (draft + critic) | 4 |
| 7 | `synthesize` | `translation.json`, `voices.json` | `tts/`, `dub_track.json`, `dub_voice.wav` | Edge TTS / Gemini TTS, Gemini (rewrites), Rubber Band | 3 |
| 8 | `mix` | `background.wav`, `dub_voice.wav`, `vocals.wav` | `dub_mix.wav` | NumPy DSP, FFmpeg `loudnorm` | 1 |
| 9 | `render` | `input.mp4`, `dub_mix.wav`, translation | `output.mp4`, SRTs, `report.json` | FFmpeg (H.264 copy, AAC, mov_text) | 1 |
| 10 | `lipsync` (opt.) | `output.mp4`, `dub_voice.wav`, `dub_track.json` | `output_lipsync.mp4` | Wav2Lip-GAN (separate environment) | 1 |

"Retries" is the stage's `max_attempts`. Only `TransientError` (HTTP 408/5xx, per-minute 429, network failures) is retried, with exponential backoff (2 s, 4 s, 8 s). Validation and configuration errors fail immediately with an actionable message, and so does a Gemini **daily** quota 429 (`QuotaExhausted`), once every fallback model is exhausted.

### Checkpointing runner ([`dubber/pipeline/`](../dubber/pipeline))
- **Job folder and manifest.** A job is a folder plus `manifest.json`, which records each stage's status, attempt count, start and finish times, duration and error, plus a cost ledger. Manifest writes are atomic: write a temp file, then `os.replace`.
- **When a stage counts as complete.** Its record says `done` **and** all its declared output files exist. Delete an artifact and the stage re-runs.
- **Resume.** Completed stages are skipped. Once any stage actually runs, every downstream stage re-runs too, because its inputs may have changed. `--force <stage>` invalidates from that point.
- **Review pause.** `stop_after="translate"` pauses the job for human review. Edits go in through the API; `/continue` then re-runs from `synthesize`, or from `translate` if the transcript was edited.
- **Retries don't double-count cost.** The stage's cost entries are cleared at the start of each attempt.
- **Progress.** Each stage has a weight, so progress can be reported as a single 0–1 value for the whole pipeline. Events stream to the UI over Server-Sent Events.

### GPU lock
`ml.gpu_scope()` takes a cross-process file lock (`jobs/.gpu.lock`) and frees the CUDA cache on exit. Each GPU stage loads its model, uses it, deletes it, and releases the lock. Peak VRAM therefore stays at one model's footprint. Whisper large-v3 int8 is about 2 GB.

### API and UI
**FastAPI ([`dubber/api/app.py`](../dubber/api/app.py)):**
- Upload with per-job options: target language, TTS provider, review pause, lip sync, number of speakers.
- SSE event stream.
- GET/PUT for transcript, translation and voices. Edits are validated, and translation edits can change wording and emotion but never timings.
- `continue` and `retry` endpoints.
- **Per-line re-voice**: `POST /lines/{id}/regenerate` re-synthesizes one line with the user's final wording (no LLM rewrite, stretch only), rebuilds the voice track, then re-runs only `mix` and `render`. Measured 8.1–8.4 s with Edge, about 3.5 s of which was the source-loudness measurement; that is now cached in `source_loudness.json`.
- **Voice preview**: `GET /api/voices/{provider}/{voice}/preview`, cached on disk (1.3 s, then ~0.03 s). The voice catalog can be filtered by target language.
- **Quality report**: `GET /quality` collects ASR confidence and recovered regions, speaker count, sync overflow and critic scores into warnings.
- A whitelisted file server: artifact names only, which blocks path traversal.
- A cost and time report.
- A single-worker queue, because the GPU fits one job at a time. Jobs run from the CLI show their state in the UI too, because it is inferred from the manifest.

**React UI ([`web/`](../web)):**
- Upload page with a live preview.
- The pipeline drawn as a signal chain with live progress.
- A two-track timeline (Chinese source and English dub) showing each line's time budget, tempo, LLM rewrites and silence.
- A/B monitor with karaoke subtitles.
- Script editor with fit gauges, low-ASR-confidence markers, `asr_fix` tags, one-click re-voice per line, and speaker cast cards with voice preview.
- Critic QA card, quality panel, cost report and downloads.

---

## 2. AI models and APIs selected

| Task | Choice | Runs |
|---|---|---|
| Source separation | Demucs v4 `htdemucs_ft` (bag of 4 fine-tuned Hybrid Transformer Demucs) | local GPU |
| Voice activity detection | Silero VAD (bundled with faster-whisper) | local |
| ASR | Whisper `large-v3` via faster-whisper (CTranslate2, int8_float16) | local GPU |
| Diarization | pyannote.audio 3.4, `speaker-diarization-3.1` (segmentation-3.0 + WeSpeaker ResNet34), plus our own speaker merge using its embeddings and pitch | local GPU |
| Gender / pitch / casting | librosa pYIN F0 tracking; voice chosen by closest pitch | local CPU |
| Translation, critic, rewrites | Gemini `gemini-3.8-flash`, falling back to `gemini-3.5-flash`, then `gemini-flash-latest`; structured JSON output | API |
| TTS | Microsoft Edge neural voices (`edge-tts`, default; native voices for es/fr/de/ja/ko/si/hi targets), or Gemini `gemini-3.8-flash-tts` (multilingual, style-prompted emotion) | API |
| Time-stretch | Rubber Band (FFmpeg `rubberband`, formant-preserving) | local |
| Mixing / loudness | NumPy, FFmpeg `loudnorm` (EBU R128) | local |
| Lip sync (bonus) | Wav2Lip-GAN | local GPU, isolated environment |

---

## 3. Why these models (alternatives considered)

### Separation: Demucs `htdemucs_ft`
Separation does double duty. ASR runs on the clean vocal stem, and the background stem keeps the music and effects. Benchmarked against ground-truth stems on the 58 s synthetic clip (`scripts/bench_separation.py`):

| Model | Time | Speed | Vocal SDR | Music SDR |
|---|---|---|---|---|
| `htdemucs` | 5.2 s | 11.3× realtime | 19.8 dB | 18.0 dB |
| **`htdemucs_ft`** | 18.4 s | 3.2× realtime | **22.1 dB** | **20.3 dB** |

+2.3 dB SDR means about 40% less residual error. It costs about 13 s per minute of video, and the gain feeds into both ASR accuracy and how clean the background stem is. `htdemucs` is one config switch away for throughput-bound deployments.

### ASR: faster-whisper large-v3 (not WhisperX, not large-v3-turbo)
- **large-v3-turbo** prunes the decoder to 4 layers. It's faster, but known to be weaker on Mandarin. ASR errors propagate into every later stage, so accuracy won.
- **WhisperX** adds wav2vec2 forced alignment. On Windows its pinned dependencies conflict with our torch and pyannote versions. We get word timestamps from Whisper's cross-attention instead, and the remaining boundary error is handled by punctuation snapping (§4).
- **CTranslate2 int8** makes large-v3 fit in about 2 GB of VRAM, with roughly 4× less memory and faster decoding than PyTorch fp16.

### Diarization: pyannote 3.1
The best open diarization pipeline that runs locally, and the assessment's suggested tool. We pass the waveform to it in memory, which avoids pyannote's file-decoding dependencies.

On real film audio it **over-splits** speakers, so we add a merge step ([`merge_speakers`](../dubber/stages/diarize.py)):
- pyannote is called with `return_embeddings=True`.
- Two clusters merge when their duration-weighted embedding centroids have cosine similarity ≥ 0.40 **and** their median pitch is within ~3.5 semitones (|ln f0a/f0b| ≤ 0.20). If a cluster has no reliable pitch, the similarity bar rises to 0.50.
- Clusters with under 1.5 s of speech are absorbed by the most similar cluster of the same gender.
- Every merge is logged to `diarization_meta.json`.
- `num_speakers` skips the merge; `min_speakers` / `max_speakers` are passed through to pyannote.
- Measured: the real clip went from **5 to 3 speakers**; the synthetic two-speaker clip stays at **2**.

The October 2026 review recommends **pyannote community-1** next. It improves DER on almost every benchmark (AliMeeting 24.5 → 20.3).

### Translation: Gemini
- The user's available key was Gemini.
- Native structured output (`response_schema`) gives typed JSON per line, so lines can't be merged or dropped.
- Its long context lets it see the whole conversation, and the low-thinking mode keeps latency down.
- DeepL and other MT systems were rejected: they can't follow per-line time budgets, tag emotion, or self-review.

### TTS: Edge (default) and Gemini TTS (optional)
- **ElevenLabs** and **OpenAI TTS** are paid, and no keys were available.
- **Gemini TTS** is emotion-steerable through natural-language style prompts, but its free tier allows only a handful of requests per day. A 60-line video wouldn't fit.
- **Edge neural voices** are free, unlimited and fast, with good male and female US voices. Non-English targets get native locale voices (verified against Edge's live voice list).
- Both sit behind one `TTSProvider` interface with content-hash caching, so switching is one option.
- **Casting by pitch:** the median F0 of all 10 English Edge voices was measured (100–197 Hz), and each speaker gets the unused same-gender voice nearest in log-pitch. Example on the real clip: 112 Hz → Brian (117 Hz), 138 Hz → Guy (143 Hz), 208 Hz → Ava (197 Hz). Gemini voice pitches are gender-typical estimates, because measuring them would spend free-tier quota.
- The review found **Gemini 3.8 Flash TTS ranked #2 on the Artificial Analysis speech arena** (about $0.0135/min). It is the right premium option once billing is enabled.

### Orchestration: a custom runner, not LangGraph
LangGraph was considered and deliberately not used:
1. **Most of the pipeline is deterministic, not agentic.** ASR, separation, mixing and muxing run the same steps every time, with nothing to decide. The LLM makes decisions in only two places, the translation critic and the sync fit/rewrite loop, and both are small loops in tested code.
2. **The hardware forces the GPU stages to run in sequence.** On 4 GB VRAM, Whisper, Demucs and pyannote can't be loaded together. ASR also *depends* on separation finishing first, since that's what makes it robust to music. Parallel graph branches wouldn't help.
3. **Every feature we'd use is already implemented and tested** in about 200 lines: checkpoint and resume, retry with backoff, a human-in-the-loop pause, downstream invalidation, and progress streaming (`tests/test_pipeline.py`).
4. **Fewer dependencies.** The LangChain ecosystem on top of a CUDA/pyannote stack means more version conflicts and no added capability.
5. **Easy to switch later.** Every stage is a plain `run(job, progress)` with declared inputs and outputs. If the system grows into genuinely agentic planning, the stages can become LangGraph nodes unchanged.

---

## 4. Challenges encountered

Every item below was hit on real runs. Each lists the root cause, the fix, and the evidence.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | ASR output was one 42 s segment reading "请使用简体中文并加上标点符号" ("please use simplified Chinese with punctuation"), confidence 0.23 | Whisper treats `initial_prompt` as *preceding transcript*, so it echoed an instruction-style prompt | Use a short natural *example* (`嗯，好的。那我们开始吧。`, "Mm, OK. Let's begin."). The synthetic clip then came out 9/9 lines word-for-word correct, with punctuation, confidence 0.95–1.00. Echoes of the prompt are also filtered. |
| 2 | **25 s of dialogue silently missing** on a real film clip (7–34 s), though the vocal stem was loud there | Known Whisper long-form failure: an early end timestamp makes the decoder skip most of a 30 s window | **Coverage self-check.** VAD speech regions less than 50% covered by the transcript are re-decoded on their own. On the real clip this recovered 2–5 regions, and the full dialogue came back. |
| 3 | Batched decoding produced 坦黛 instead of 唐代, and even a Japanese sentence | Batched mode decodes chunks without cross-chunk context | Default is now sequential decoding plus the coverage check. Batched mode is opt-in (`DUBBER_ASR_BATCHED`). Segments containing kana are discarded and re-decoded. |
| 4 | Lines split mid-phrase across speakers: "回去？回 \| 中国", "…空白。别 \| 找了" | pyannote's turn boundaries are often a syllable off from the word boundaries | **Punctuation snapping**: a speaker change within 2 words of punctuation moves to it. **Fragment merge**: stray ≤2-character fragments join the next line from the same speaker. Unit-tested on these exact cases. |
| 5 | `hf_hub_download() got an unexpected keyword argument 'use_auth_token'` | pyannote 3.x uses a keyword argument that huggingface_hub ≥ 1.0 removed | A small shim translates `use_auth_token` to `token`. We didn't downgrade the hub library, because faster-whisper and tokenizers need the new one. |
| 6 | `UnpicklingError: Weights only load failed` | PyTorch ≥ 2.6 loads checkpoints with `weights_only=True` | Allowlist exactly the 4 classes pyannote checkpoints contain (`TorchVersion`, `Specifications`, `Problem`, `Resolution`). The safety check stays on for everything else. |
| 7 | `'NoneType' object has no attribute 'eval'` | The gated `segmentation-3.0` returned 403. pyannote swallows that error and returns `None`. | A preflight check of every gated repo raises "accept the terms at <URL>" and is not retried. |
| 8 | `open() got an unexpected keyword argument 'metadata_errors'` | A PyAV API break inside faster-whisper's file decoder | Decode with libsndfile, resample with librosa, and pass NumPy arrays. One loader now serves ASR, diarization and pitch analysis. |
| 9 | `cublas64_12.dll not found` | CTranslate2 needs CUDA DLLs that only PyTorch's wheel ships | Import torch first, so its bundled CUDA runtime is loaded. No CUDA toolkit install is needed. |
| 10 | Translation took **222–372 s** | The Gemini SDK silently retried 503 "high demand" errors with long backoff before our fallback could act | Disable SDK retries (`attempts=1`), set a 120 s timeout, and walk a fast fallback chain. A **circuit breaker** skips a model for 2 minutes after a 503. The draft uses **low thinking**. Translation now takes **40.8 s**. |
| 11 | An ASR test took **1 hour** | Another process was using the 4 GB GPU at the same time, and VRAM spilled into shared system memory | **Cross-process GPU lock** in `ml.gpu_scope()` |
| 12 | Timeline, QA and report panels sometimes didn't render | Browser connection starvation: the SSE stream, video range requests and 7 per-file size probes exceeded 6 connections per host | File sizes now come with the job summary, so there are no per-file requests |
| 13 | Gemini client "closed" mid-request | A temporary `genai.Client` was garbage-collected | One cached client per process |
| 14 | 5 speakers detected for about 3 people, so one actor switched voices mid-scene | pyannote over-splits short film turns. Embedding similarity **alone** can't fix it: a real man/woman pair scored cosine 0.44, the same as over-split pairs of one person (0.41–0.44) | Merge only when the embeddings are similar **and** the pitch is within ~3.5 semitones (the man/woman pair differed by ~8). Real clip 5 → 3; synthetic clip stays at 2. |
| 15 | "没有测绘" became **"Blank."**, "别" became "Don't." | Sub-second lines got 0.4–0.6 s budgets, about one English word | `merge_short_lines` joins lines under 1 s (or ≤2 characters) with an adjacent line of the **same** speaker (gap ≤0.35 s). Every line also gets a minimum budget of min(0.8 s, gap). Result: *"Without architectural surveying, it's a total blank."* |
| 16 | ASR homophones (丹代 for 唐代 "Tang dynasty", 扛够 for 唐构) | Whisper's language model, weak on rare Chinese terms | The translator sees low-confidence words (`[uncertain: 丹代(0.41)]`), repairs them from context, and records each repair in `asr_fix`. Fixed 丹代→唐代 (twice) and 扛够→唐构. Missed 冰火→兵火, because Whisper was 0.94 confident about it. |
| 17 | Every Gemini model returned 429 mid-evaluation, and stages kept retrying | The free tier allows about 20 requests per model per day, and a daily 429 looked like a transient error | Daily-quota 429s are classified as `QuotaExhausted`: skip to the next model, never retry the same one, and if every fallback is exhausted, fail with "enable billing / use a -lite model / wait for reset". Per-minute 429s are still retried. |

---

## 5. Translation and synchronization techniques

### Time budgets ([`align.time_budgets`](../dubber/align.py))
- A line may use its own slot plus the silence before the next line, keeping a 0.15 s pause.
- It may run at most 1.5 s past where the original speaker stopped.
- Overlapping source lines keep their own duration.
- Every line gets at least min(0.8 s, time to the next line), giving up the 0.15 s pause if necessary.
- Before budgeting, sub-second lines are merged with an adjacent line of the same speaker ([`align.merge_short_lines`](../dubber/align.py)). Different-speaker neighbours are never merged, because that is real turn-taking.
- Budgets become word budgets at 2.6 words per second, roughly natural conversational English at 150–170 words per minute.

### Draft, critique, revise ([`dubber/translate.py`](../dubber/translate.py))
1. **Draft.**
   - The whole conversation is sent at once, in chunks of 60 lines with 8 lines of context on each side for long videos.
   - Each line comes with its speaker, timecode, seconds available and `max_words`.
   - The system prompt is written as a dubbing-script adapter: keep meaning, subtext and register; adapt idioms rather than translating them literally; resolve pronouns from context; use contractions; no stage directions; one output line per input id.
   - The output is schema-validated JSON (`id`, `text`, `emotion`, `asr_fix`). Ids the model skips are requested again.
   - **ASR-aware:** words Whisper was less than 60% confident about are listed per line, least confident first (`[uncertain: 丹代(0.41)]`). The prompt explains the source is ASR output that may contain homophone errors. The model corrects them from context and reports each correction (`丹代→唐代`), which is stored in `translation.json` and shown in the editor.
2. **Critique (LLM-as-judge).**
   - A second pass scores accuracy, naturalness and tone from 1 to 10.
   - It flags only real problems, in six categories: accuracy, idiom, tone, fluency, consistency and length.
   - **Budget guard:** a fix is accepted only if it fits `max_words` + 1. Fixes for nonexistent line ids are dropped.
   - Every before/after change is stored in `translation.json` and shown in the UI.
   - If the critic call fails, the job continues with the draft.
3. **Emotion tags** use 12 categories and drive Gemini TTS style prompts ("Say with a worried tone:").

**Example from the synthetic clip.** 解铃还须系铃人 (literally "whoever tied the bell must untie it") was rendered as *"I made this mess, I'll fix it."* The critic scored the script 10/10/10 and correctly proposed no changes. Draft plus critic cost $0.009.

### Fit loop ([`stages/synthesize.py`](../dubber/stages/synthesize.py), [`align.plan_fit`](../dubber/align.py))
1. All lines are synthesized in parallel: 6 workers for Edge, 3 for Gemini. Each clip is trimmed of leading and trailing silence (−42 dB) so its duration is real speaking time. Clips are cached by a hash of provider, voice, emotion, style and text.
2. Lines **more than 12% over budget** go back to the LLM **in a single batched request** to be shortened, then are re-synthesized. There are at most 2 rounds.
3. Remaining overruns are **time-stretched with Rubber Band**, preserving pitch and formants, **capped at 1.25×**.
4. Each line is placed at its **original start time** on a silent timeline. Silent sections stay silent, and pauses between lines are preserved.

### Mixing ([`stages/mix.py`](../dubber/stages/mix.py))
1. **Level match:** the dub is gained to the RMS of the original dialogue measured over speech regions, clamped to ±12 dB. The source's voice/music balance is preserved.
2. **Ducking:** the background is lowered by −4 dB under English lines, with 80 ms linear ramps. This masks residual Chinese bleed left by separation.
3. **Final loudness:** `loudnorm` sets the whole mix to the source's measured integrated loudness, with a −1.5 dBTP peak ceiling.

### Render ([`stages/render.py`](../dubber/stages/render.py))
- **Video:** H.264 yuv420p sources are **stream-copied**, so the picture is bit-exact with no generation loss. Anything else is re-encoded with libx264 CRF 18.
- **Audio:** AAC 192 kb/s, with a language tag.
- **Subtitles:** English soft subtitles (`mov_text`) follow the *dub* timing. Separate SRT files are written in English and Chinese.
- **Checks:** `+faststart`, and the output duration must match the input within 250 ms. Measured drift was 0 ms on the test clips.

---

## 6. Limitations and known issues

- **Speaker attribution on film audio is still imperfect.** The merge step reduced 5 detected speakers to 3 on the real clip, but even the best published systems reach about 37–45% DER on movie audio. `num_speakers` at upload, or recasting in the UI, is the manual fallback.
- **Homophones the ASR is *confident* about are not caught.** `asr_fix` repaired 丹代→唐代 and 扛够→唐构, but missed 冰火→兵火 (confidence 0.94), which became "fire and ice". The Chinese transcript itself keeps the ASR text; repairs live only in the translation.
- **Budgets assume English pace.** The 2.6 words/second budget is used for every target language. Syllable-based budgets would suit Japanese, Korean and the others better.
- **Edge TTS has no emotion control.** Emotion tags affect only Gemini TTS, whose free tier is too small for full videos.
- **Gemini free tier:** about 20 requests per model per day. A few full runs exhaust it, and a weaker `-lite` fallback can over-shorten lines in rewrites (seen: "Still right." for "But they're not necessarily wrong."). Billing removes both issues for about $0.01 per minute of video.
- **Overlapping speech** is mixed as overlapping dub lines, not separated per speaker.
- **Mouth movements** won't match the English unless lip sync is on.
- **Lip sync (Bonus A)** is implemented as an out-of-process Wav2Lip-GAN stage (`stages/lipsync.py`, `wav2lip_runner.py`, `scripts/setup_lipsync.ps1`):
  - Only frames inside dubbed segments are processed.
  - It needs the separate `.venv-lipsync` environment, and fails with setup instructions if that's missing.
  - Wav2Lip outputs 96×96 mouth crops, so faces look softer than the source.
  - It is off by default and adds significant runtime (measured about 2.2 s per video-second, 1.9 GB VRAM, on a 22 s talking-head sample).
  - It animates the largest face, not necessarily the active speaker.
  - The Wav2Lip weights are licensed for non-commercial use only.
  - The run on the real film clip still has to be started by the user (README → Lip sync), because executing the third-party code requires their approval.
- **Third-party pricing.** API prices are list prices from third-party sources, October 2026. See the cost analysis.
- **Input limits:** inputs are limited to MP4 files of 30–180 s by design; both are configurable.

---

## 7. Potential production improvements

The [technology review](../reports/Dubbing%20pipeline%20technology%20review.md) (October 2026, with sources) compares each stage with the current state of the art. Its highest-value upgrades that fit in 4 GB:
1. **Qwen3-ASR + Qwen3-ForcedAligner**, keeping faster-whisper as a fallback. Vendor-reported WenetSpeech CER: 9.86/19.11% → 4.97/5.88% (net/meeting). Word-timing error: 133 → 43 ms.
2. **Mel-Band RoFormer separation** via `audio-separator`, about +2.5 dB vocal SDR on MVSEP Multisong. Demucs has been archived since January 2025.
3. **pyannote community-1** diarization.
4. **Syllable budgets with several length variants**, chosen by measured duration, plus an MQM-style critic from a different model family.

None of these has been measured on 4 GB yet; each needs a smoke test before switching.

| Area | Improvement |
|---|---|
| Latency | Keep models resident in a GPU worker (no 10–20 s load per job); on ≥ 16 GB GPUs run diarization in parallel with ASR; batched ASR for long inputs; stream TTS while translating later chunks |
| Throughput | Job queue (Celery/Arq/Redis) with dedicated GPU workers; object storage for artifacts; idempotent stages already make workers stateless |
| Long videos | Chunk at scene cuts or long silences (~5 min) with overlap; translate with a rolling context and a per-video glossary; checkpoint per chunk |
| Speaker consistency | Cluster pyannote speaker embeddings across chunks/scenes; persistent speaker → voice registry per series |
| Voice quality | Consent-based voice cloning (XTTS, ElevenLabs) to keep each actor's timbre; prosody transfer of the original emotion |
| Translation QA | Automatic metrics (COMET/xCOMET, reference-free) alongside the LLM critic; back-translation checks; human QA sampling with edit tracking in the UI |
| Sync | Viseme-aware phrasing; phoneme-level forced alignment for tighter lip-sync windows |
| Lip sync | Higher-resolution models (MuseTalk/LatentSync) on larger GPUs; face-restoration pass |
| Reliability | Structured logging + tracing (OpenTelemetry), per-stage SLOs, alerting on fallback-rate spikes, dead-letter queue for permanent failures |
| Cost | Cache translations and TTS across re-runs (TTS cache exists); batch API calls; cheaper "lite" models for short or easy lines |

---

## Performance & accuracy optimizations (measured)

### Real-video run: 50.6 s film clip, 848×480, about 3 speakers

| Stage | Before | After | What changed |
|---|---|---|---|
| separate | 44.3 s | 44.3 s | (includes first model load) |
| transcribe | 111.5 s | **24.7 s** | No lost windows. Coverage check recovered the missing 25 s of dialogue. |
| diarize | — (403 failure) | 13.4 s | Preflight check, compatibility shims |
| voices | 8.8 s | 1.7 s | — |
| translate | 221.8 s / 372.4 s | **40.8 s** | No SDK retry storms, circuit breaker, low-thinking draft. The critic now completes. |
| synthesize | 165.6 s | **23.3 s** | The same failover applies to rewrite calls |
| mix | 3.4 s | 3.6 s | — |
| render | 1.7 s | 1.2 s | H.264 stream copy |
| **Total** | **575 s** | **154 s (3.7× faster)** | API cost $0.025 → $0.016 |

### Decision summary

| Decision | Latency effect | Accuracy effect | Evidence |
|---|---|---|---|
| ASR on Demucs vocal stem | +18 s per minute (separation) | Music no longer disturbs ASR; background is clean | 22.1 dB vocal SDR |
| `htdemucs_ft` over `htdemucs` | −13 s per minute | +2.3 dB vocal and music SDR | bench table, §3 |
| Example prompt (not instruction) | — | Removed the 42 s hallucination; punctuated sentences | 9/9 lines exact on synthetic clip |
| Coverage self-check | +1–3 s | Recovered 25 s of skipped dialogue | real clip |
| Sequential over batched ASR (default) | about +5 s on 1 min | Fixed 坦黛→唐代 and the Japanese drift | real clip |
| Punctuation snapping + fragment merge | ~0 | Lines no longer split mid-phrase across speakers | unit tests from real failures |
| int8 CTranslate2 Whisper | ~4× less VRAM, faster | Negligible WER impact | fits 4 GB |
| No SDK retries + fallback chain + circuit breaker | translate 222–372 s → 41 s | Critic now completes | real clip |
| Low thinking for draft and shorten, full for critic | Faster drafts | Quality gate stays strong | — |
| Batched LLM rewrites (one request for all long lines) | N calls → 1 | Model sees all fixes together | — |
| Parallel TTS + content-hash cache | Edits re-voice only changed lines | — | — |
| H.264 stream copy | Render ≈ 1 s | Bit-exact visuals | 0 ms duration drift |
| GPU file lock | Avoids ~100× slowdown | — | 1 h run without the lock |
| Embedding + pitch speaker merge | ~0 | 5 → 3 speakers on the real clip; synthetic stays at 2 | `diarization_meta.json`, unit tests with measured values |
| Pitch-matched casting | ~0 | Deep voices get deep voices (112 Hz → 117 Hz voice) | `voice_casting.json` |
| Short-line merge + minimum budget | ~0 | "Blank." → full sentence | real clip |
| ASR-aware translation (`asr_fix`) | ~0 (same request) | 3 homophone repairs on the real clip | real clip |
| Daily-quota classification | Avoids minutes of futile retries | — | unit test |
| Cached source loudness | ~3.5 s off each re-voice | — | profiled re-voice (8.1–8.4 s before) |
| Per-line re-voice (no full re-run) | One line in ~8 s instead of a ~2 min job | User wording is final | API test + manual run |
