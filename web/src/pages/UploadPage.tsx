import { AudioLines, Clapperboard, Languages, Loader2, Mic2, Sparkles, Wand2 } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { navigate } from "../App";
import { api, fmtBytes, fmtTC } from "../api";
import { HeroWave } from "../components/Brand";
import { Chip, ErrorNote, Panel, PatchToggle } from "../components/ui";

const LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Korean", "Sinhala", "Hindi"];
const MIN_S = 30;
const MAX_S = 180;

const PATH = [
  { icon: AudioLines, title: "Separate", text: "Demucs lifts the Chinese voices off the music & effects bed." },
  { icon: Mic2, title: "Listen", text: "Whisper + pyannote transcribe every line and tell speakers apart." },
  { icon: Languages, title: "Adapt", text: "Gemini writes a natural English script that fits each line's timing, then critiques it." },
  { icon: Wand2, title: "Perform", text: "Voices are cast per speaker, fitted to the picture and mixed back in." },
];

interface Probe { duration: number; width: number; height: number }

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [probe, setProbe] = useState<Probe | null>(null);
  const [drag, setDrag] = useState(false);
  const [language, setLanguage] = useState("English");
  const [provider, setProvider] = useState<"gemini" | "edge">("edge");
  const [review, setReview] = useState(true);
  const [lipsync, setLipsync] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const pick = (f: File | undefined) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".mp4")) {
      setError("Please choose an .mp4 file.");
      return;
    }
    setError(null);
    setProbe(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files[0]);
  };

  const durationOk = probe ? probe.duration >= MIN_S && probe.duration <= MAX_S : true;

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("target_language", language);
    form.append("tts_provider", provider);
    form.append("review", String(review));
    form.append("lipsync", String(lipsync));
    try {
      const job = await api.createJob(form);
      navigate(`/jobs/${job.job_id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="space-y-8">
      <section className="grid items-center gap-8 pt-2 lg:grid-cols-[1.15fr_1fr]">
        <div className="rise-in">
          <span className="eyebrow">Session 01 · Mandarin → {language}</span>
          <h1 className="mt-3 font-display text-4xl font-extrabold leading-[1.02] tracking-tight sm:text-6xl">
            Same scene.<br />
            Same music.<br />
            <span className="text-signal">New voice.</span>
          </h1>
          <p className="mt-5 max-w-lg text-[0.95rem] leading-relaxed text-muted">
            Drop in a Chinese video and the studio separates the dialogue, transcribes each speaker,
            writes a timing-aware {language} script, re-voices every line and mixes it back over
            the original soundtrack.
          </p>
        </div>
        <HeroWave />
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        {/* Film-slate drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className={`panel overflow-hidden transition ${drag ? "ring-2 ring-signal" : ""}`}
        >
          <div className="h-5"
            style={{ background: "repeating-linear-gradient(-45deg, var(--text) 0 18px, var(--panel) 18px 36px)" }} />
          <div className="grid grid-cols-2 border-b border-line font-mono text-[0.68rem] uppercase tracking-wider sm:grid-cols-4">
            {[["Prod", "Dub Studio"], ["Roll", `ZH → ${language.slice(0, 2).toUpperCase()}`],
              ["Scene", file ? file.name.replace(/\.mp4$/i, "") : "—"], ["Take", "01"]].map(([k, v]) => (
              <div key={k} className="min-w-0 border-r border-line px-4 py-2.5 last:border-r-0">
                <div className="text-faint">{k}</div>
                <div className="truncate text-text">{v}</div>
              </div>
            ))}
          </div>

          <input ref={inputRef} type="file" accept="video/mp4,.mp4" hidden
            onChange={(e) => pick(e.target.files?.[0])} />

          {preview && file ? (
            <div className="space-y-3 p-4">
              <video src={preview} controls playsInline
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  setProbe({ duration: v.duration, width: v.videoWidth, height: v.videoHeight });
                }}
                className="aspect-video w-full rounded-xl bg-black" />
              <div className="flex flex-wrap items-center gap-2">
                <Chip>{fmtBytes(file.size)}</Chip>
                {probe && <Chip>{probe.width}×{probe.height}</Chip>}
                {probe && <Chip tone={durationOk ? "ok" : "bad"}>{fmtTC(probe.duration)}</Chip>}
                {probe && <Chip tone={durationOk ? "ok" : "bad"}>
                  {durationOk ? "✓ within 30 s – 3 min" : "✕ must be 30 s – 3 min"}</Chip>}
                <button className="btn-ghost ml-auto" onClick={() => inputRef.current?.click()}>Replace</button>
              </div>
            </div>
          ) : (
            <button type="button" onClick={() => inputRef.current?.click()}
              className="group grid min-h-72 w-full cursor-pointer place-items-center p-8 text-center">
              <span>
                <span className="mx-auto mb-5 grid size-16 place-items-center rounded-2xl border border-line-strong bg-panel-2 transition group-hover:-rotate-6 group-hover:border-signal">
                  <Clapperboard className="size-7 text-signal" />
                </span>
                <span className="block font-display text-xl font-bold">Drop your take here</span>
                <span className="mt-1 block text-sm text-muted">MP4 · Mandarin dialogue · 30 seconds to 3 minutes</span>
                <span className="mt-4 inline-block font-mono text-[0.7rem] tracking-wider text-faint">
                  OR CLICK TO BROWSE
                </span>
              </span>
            </button>
          )}
        </div>

        {/* Patch bay */}
        <Panel eyebrow="Patch bay" title="Session settings" icon={<Sparkles className="size-4" />} className="h-fit">
          <div className="space-y-5">
            <div>
              <span className="eyebrow mb-2 block">Target language</span>
              <div className="flex flex-wrap gap-1.5">
                {LANGUAGES.map((l) => (
                  <button key={l} type="button" onClick={() => setLanguage(l)}
                    className={`rounded-md border px-2.5 py-1 text-xs transition ${l === language
                      ? "border-signal bg-signal text-signal-ink" : "border-line bg-panel-2 text-muted hover:text-text"}`}>
                    {l}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <span className="eyebrow mb-2 block">Voice engine</span>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ["edge", "Edge Neural", "Free · unlimited"],
                  ["gemini", "Gemini TTS", "Emotion-steerable"],
                ] as const).map(([id, name, hint]) => (
                  <button key={id} type="button" onClick={() => setProvider(id)} aria-pressed={provider === id}
                    className={`rounded-xl border p-3 text-left transition ${provider === id
                      ? "border-signal/60 bg-signal/8" : "border-line bg-panel-2 hover:border-line-strong"}`}>
                    <span className="block text-sm font-semibold">{name}</span>
                    <span className="block text-xs text-muted">{hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <PatchToggle code="REV" checked={review} onChange={setReview}
                label="Review script before voicing" hint="Pause to edit lines & recast voices." />
              <PatchToggle code="LIP" checked={lipsync} onChange={setLipsync}
                label="AI lip sync" hint="Re-render mouths to match English. Slower." />
            </div>

            {error && <ErrorNote>{error}</ErrorNote>}

            <button className="btn-signal w-full py-3 text-base" disabled={!file || busy || !durationOk} onClick={submit}>
              {busy ? <Loader2 className="size-4 animate-spin" /> : <span className="size-2 rounded-full bg-bad" />}
              {busy ? "Uploading…" : "Roll — start dubbing"}
            </button>
          </div>
        </Panel>
      </section>

      <section>
        <div className="eyebrow mb-3">Signal path</div>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {PATH.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="panel relative p-4">
              <span className="tc absolute right-4 top-4 text-[0.7rem] text-faint">0{i + 1}</span>
              <Icon className="mb-3 size-5 text-signal" />
              <p className="font-display font-bold">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-muted">{text}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
