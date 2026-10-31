# Cost Analysis

## Summary

The default configuration uses Edge TTS and runs on a cloud L4 GPU. Costs are in USD.

| Video duration | API fees | Infrastructure (GPU) | **Total** | Processing time: RTX 3050 laptop (measured basis) | Processing time: cloud L4 (estimate) |
|---|---|---|---|---|---|
| 1 minute | $0.020 | $0.014 | **$0.034** | ~3 min | ~1.1 min |
| 5 minutes | $0.10 | $0.054 | **$0.15** | ~13 min | ~4.5 min |
| 30 minutes | $0.60 | $0.30 | **$0.90** | ~76 min | ~24 min |
| 60 minutes | $1.20 | $0.60 | **$1.80** | ~2.5 h | ~48 min |

Switching TTS to **Gemini TTS** (emotion-steerable) adds about **$0.01 per minute**: +$0.01, +$0.05, +$0.30 and +$0.60 for the four rows. On the local laptop, infrastructure cost is only electricity, which is negligible (below).

## Measured basis

| Measurement | Value |
|---|---|
| Real 50.6 s film clip, end-to-end on RTX 3050 4 GB | **154 s** processing, **$0.0162** API cost (translation + critic + rewrites) |
| Same clip during a Gemini overload, before the failover fixes | 575 s, $0.025 |
| Synthetic 5-line dialogue, draft + critic | $0.009 |
| GPU-bound stages on that clip (separate + transcribe + diarize) | 44.3 + 24.7 + 13.4 = 82 s, about 1.6× realtime, including model loads |
| API-bound stages (translate + synthesize) | 40.8 + 23.3 = 64 s |

| Same clip after the A1–A4 accuracy changes, translation on `gemini-3.5-flash-lite` (main models' free quota exhausted) | 123 s, $0.0066 |
| Re-voicing one edited line (Edge TTS + mix + render) | 8.1–8.4 s, $0; faster now that source loudness is cached |
| Voice preview | 1.3 s first time, ~0.03 s cached, $0 with Edge |

**Free tier vs billing:** the Gemini free tier allows about 20 requests per model per day, which is only a few full jobs. The numbers in this document are paid-tier list prices. At roughly $0.01–0.02 per minute of video, enabling billing is the practical choice even for development.

## Assumptions

1. **API prices** come from `dubber/pricing.py`, where the higher of the third-party October 2026 listings is used:
   - `gemini-3.8-flash`: $0.75 per million input tokens, $3.75 per million output tokens, thinking tokens billed as output.
   - `gemini-3.8-flash-tts`: $0.50 per million input tokens, $9.00 per million audio output tokens, at 25 audio tokens per second.
   - Edge TTS: $0.

   **These rates should be checked against Google's official pricing page.** Costs are always computed at paid-tier rates, even on a free-tier key.
2. **Translation tokens per minute of video.**
   - Mandarin speech runs at about 4 characters per second, so about 240 characters (≈ 250 tokens) per minute.
   - The prompt adds per-line metadata and context (≈ 3× the source), the critic re-reads draft and source, and thinking plus JSON output are added on top.
   - Formula: `cost/min ≈ (in_tokens × 0.75 + out_tokens × 3.75) / 1e6`.
   - Calibrated against the measured $0.0162 for 50.6 s, this gives about **$0.02 per minute**, including about 10% for rewrite calls.
3. **Gemini TTS per minute.** With about 70% speech, that's 42 s × 25 tokens/s × $9 per million ≈ $0.0095, plus negligible input, so about **$0.01 per minute**.
4. **Processing time model.**
   - Laptop: `≈ 30 s fixed (model loads) + 2.5 × video duration`. This matches the measured 154 s for 50.6 s.
   - Cloud L4, estimated: `≈ 20 s + 0.8 × duration`. The L4 is about 4–5× the 3050 on the GPU stages, and API latency is unchanged.
5. **Cloud GPU rate.** NVIDIA L4 on-demand at about **$0.75/h** (typical for GCP g2-standard-4 or similar), billed for the job's wall-clock time.
6. **Laptop electricity.** About 90 W while processing at $0.15/kWh comes to about $0.0007 for a 3-minute job, so effectively $0.
7. **Long videos.** Videos over 3 minutes need `DUBBER_MAX_DURATION_S` raised. The architecture already handles long inputs: translation runs in 60-line windows with context, and separation in 60 s chunks. For 30–60 minute videos, production should also split the job at long silences or scene cuts, so a failure costs one chunk rather than the whole job (see TECHNICAL.md §7).

## Breakdown per minute of video

| Item | Cost per minute | Notes |
|---|---|---|
| Translation draft (low thinking) | ~$0.008 | whole conversation in context, structured JSON |
| Translation critic | ~$0.010 | full reasoning; the largest LLM item |
| Fit rewrites (batched) | ~$0.002 | only for lines >12% over their slot |
| TTS (Edge) | $0 | Gemini TTS: ~$0.01 |
| GPU on L4 | ~$0.010 | ~0.8 min of GPU per minute of video |
| **Total** | **~$0.03–0.04** | |

## Scaling to 1,000 videos per day

Assuming an average of 2 minutes per video, that's 2,000 minutes of video per day.

| Item | Per day | Notes |
|---|---|---|
| Gemini API | ~$40 | $0.02/min × 2,000 |
| GPU | ~$36 | 2,000 × 0.8 GPU-min ≈ 27 GPU-hours → **2 L4 workers** running continuously |
| **Total** | **~$76/day** (~$0.038/min) | plus storage and egress |

Ways to cut that, in order of impact:
1. **Keep models loaded** in long-lived GPU workers. This removes the ~20 s per-job load, which is a large share for short videos.
2. **Batch API / flex pricing.** Translation isn't latency-critical in a queued system, and batch endpoints are typically about 50% cheaper.
3. **Make the critic conditional.** Run it only on lines the draft flags, or on low-confidence ASR, or use a lite model for easy lines. That cuts about 50% of LLM cost.
4. **Use `htdemucs` instead of `htdemucs_ft`** for throughput-bound tiers: 3.5× faster separation, at −2.3 dB SDR.
5. **Batched ASR** (`DUBBER_ASR_BATCHED`) on long videos, plus spot/preemptible GPUs. This is safe because stages checkpoint and resume.
6. **Cache** TTS (already content-hashed) and translations of repeated content such as intros and recaps.
