// Typed client for the FastAPI backend (see dubber/api/app.py).

export type JobState = "queued" | "running" | "awaiting_review" | "done" | "failed" | "idle";

export interface StageRecord {
  status: "pending" | "running" | "done" | "failed";
  attempts: number;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  error: string | null;
}

export interface VideoMetadata {
  container: string;
  duration_s: number;
  width: number;
  height: number;
  fps: number;
  video_codec: string;
  audio_codec: string | null;
  size_bytes: number;
}

export interface JobSummary {
  job_id: string;
  input_name: string;
  created_at: string;
  state: JobState;
  stage: string | null;
  progress: number;
  message: string;
  error: string | null;
  options: Record<string, unknown>;
  artifacts: string[];
  stages: Record<string, StageRecord>;
  total_duration_s: number;
  total_cost_usd: number;
  metadata: VideoMetadata | null;
}

export interface PipelineEvent {
  seq: number;
  status: string;
  stage: string | null;
  progress: number;
  message: string;
  state: JobState;
}

export interface TranslatedSegment {
  id: number;
  start: number;
  end: number;
  speaker: string;
  source_text: string;
  text: string;
  emotion: string;
  max_duration_s: number;
}

export interface Translation {
  source_language: string;
  target_language: string;
  model: string;
  segments: TranslatedSegment[];
}

export interface SpeakerProfile {
  speaker: string;
  gender: "male" | "female" | "unknown";
  median_f0_hz: number | null;
  speaking_time_s: number;
  voice: string;
  style: string;
}

export interface VoiceMap {
  provider: string;
  speakers: Record<string, SpeakerProfile>;
}

export interface VoiceOption {
  id: string;
  gender: "male" | "female";
  description: string;
}

export interface Report {
  job_id: string;
  video_duration_s: number | null;
  stages: { stage: string; status: string; attempts: number; duration_s: number | null;
            cost_usd: number; error: string | null }[];
  costs: { stage: string; provider: string; item: string; quantity: number; unit: string;
           usd: number }[];
  cost_by_provider: Record<string, number>;
  total_cost_usd: number;
  total_duration_s: number;
  cost_per_minute_usd: number | null;
  realtime_factor: number | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  listJobs: () => request<JobSummary[]>("/api/jobs"),
  job: (id: string) => request<JobSummary>(`/api/jobs/${id}`),
  createJob: (form: FormData) => request<JobSummary>("/api/jobs", { method: "POST", body: form }),
  continueJob: (id: string) => request<JobSummary>(`/api/jobs/${id}/continue`, { method: "POST" }),
  retryJob: (id: string) => request<JobSummary>(`/api/jobs/${id}/retry`, { method: "POST" }),
  translation: (id: string) => request<Translation>(`/api/jobs/${id}/translation`),
  saveTranslation: (id: string, t: Translation) =>
    request<Translation>(`/api/jobs/${id}/translation`, json("PUT", t)),
  voices: (id: string) => request<VoiceMap>(`/api/jobs/${id}/voices`),
  saveVoices: (id: string, v: VoiceMap) =>
    request<VoiceMap>(`/api/jobs/${id}/voices`, json("PUT", v)),
  voiceCatalog: (provider: string) =>
    request<VoiceOption[]>(`/api/voices?provider=${encodeURIComponent(provider)}`),
  report: (id: string) => request<Report>(`/api/jobs/${id}/report`),
  fileUrl: (id: string, name: string, download = false) =>
    `/api/jobs/${id}/files/${name}${download ? "?download=true" : ""}`,
  events: (id: string) => new EventSource(`/api/jobs/${id}/events`),
};

export const STAGES: { name: string; label: string; detail: string }[] = [
  { name: "ingest", label: "Video processing", detail: "Probe metadata, extract audio" },
  { name: "separate", label: "Source separation", detail: "Demucs: dialogue vs. music & SFX" },
  { name: "transcribe", label: "Speech recognition", detail: "Whisper large-v3 + VAD, word timestamps" },
  { name: "diarize", label: "Speaker diarization", detail: "pyannote: who spoke when" },
  { name: "voices", label: "Voice casting", detail: "Pitch-based gender, voice per speaker" },
  { name: "translate", label: "Translation", detail: "Context-aware, timing-budgeted LLM" },
  { name: "synthesize", label: "Voice generation", detail: "Expressive TTS, duration fitting" },
  { name: "mix", label: "Audio mixing", detail: "Align dialogue over preserved background" },
  { name: "render", label: "Final video", detail: "H.264 + AAC MP4, subtitles" },
  { name: "lipsync", label: "Lip sync", detail: "Mouth movements matched to English" },
];

export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
};

export const fmtDuration = (s: number | null | undefined) =>
  s == null ? "—" : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;

export const fmtUsd = (v: number | null | undefined) =>
  v == null ? "—" : v === 0 ? "$0.00" : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
