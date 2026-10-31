import { AudioLines, FileVideo, Languages, Loader2, Sparkles, UploadCloud, Wand2 } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { navigate } from "../App";
import { api } from "../api";
import { Card, ErrorNote, Toggle } from "../components/ui";

const LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Korean", "Sinhala", "Hindi"];

const FEATURES = [
  { icon: AudioLines, title: "Keeps the soundtrack", text: "Demucs removes the Chinese voices while music and effects stay." },
  { icon: Languages, title: "Context-aware translation", text: "Gemini translates the whole dialogue, idioms and tone, within each line's time budget." },
  { icon: Wand2, title: "Timed to the picture", text: "Lines that run long are rewritten, then gently time-stretched to fit." },
];

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [language, setLanguage] = useState("English");
  const [provider, setProvider] = useState<"gemini" | "edge">("gemini");
  const [review, setReview] = useState(true);
  const [lipsync, setLipsync] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (f: File | undefined) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".mp4")) {
      setError("Please choose an .mp4 file.");
      return;
    }
    setError(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files[0]);
  };

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
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <div className="space-y-6">
        <div className="pt-4">
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Dub a Chinese video into{" "}
            <span className="bg-linear-to-r from-indigo-300 to-fuchsia-300 bg-clip-text text-transparent">natural {language}</span>
          </h1>
          <p className="mt-3 max-w-xl text-slate-400">
            Upload a 30 s – 3 min MP4. The pipeline separates the dialogue, transcribes and identifies each speaker,
            translates with full context, then re-voices every line timed to the picture.
          </p>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`card group grid min-h-72 cursor-pointer place-items-center border-2 border-dashed p-6 transition ${
            drag ? "border-indigo-400 bg-indigo-500/10" : "border-white/10 hover:border-indigo-400/50"}`}
        >
          <input ref={inputRef} type="file" accept="video/mp4,.mp4" hidden
            onChange={(e) => pick(e.target.files?.[0])} />
          {preview && file ? (
            <div className="w-full space-y-3" onClick={(e) => e.stopPropagation()}>
              <video src={preview} controls className="aspect-video w-full rounded-xl bg-black" />
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="flex min-w-0 items-center gap-2 text-slate-300">
                  <FileVideo className="size-4 shrink-0 text-indigo-300" />
                  <span className="truncate">{file.name}</span>
                  <span className="text-slate-500">{(file.size / 1e6).toFixed(1)} MB</span>
                </span>
                <button className="btn-ghost" onClick={() => inputRef.current?.click()}>Change</button>
              </div>
            </div>
          ) : (
            <div className="text-center">
              <span className="mx-auto mb-4 grid size-14 place-items-center rounded-2xl bg-indigo-500/10 ring-1 ring-indigo-400/30 transition group-hover:scale-105">
                <UploadCloud className="size-7 text-indigo-300" />
              </span>
              <p className="font-medium text-slate-100">Drop an MP4 here or click to browse</p>
              <p className="mt-1 text-sm text-slate-500">Mandarin dialogue · 30 seconds to 3 minutes</p>
            </div>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          {FEATURES.map(({ icon: Icon, title, text }) => (
            <div key={title} className="card p-4">
              <Icon className="mb-2 size-5 text-fuchsia-300" />
              <p className="text-sm font-medium text-slate-100">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">{text}</p>
            </div>
          ))}
        </div>
      </div>

      <Card title="Dubbing options" icon={<Sparkles className="size-4 text-fuchsia-300" />} className="h-fit lg:mt-4">
        <div className="space-y-4">
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">Target language</span>
            <select className="input" value={language} onChange={(e) => setLanguage(e.target.value)}>
              {LANGUAGES.map((l) => <option key={l}>{l}</option>)}
            </select>
          </label>

          <div>
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">Voice engine</span>
            <div className="grid grid-cols-2 gap-2">
              {([
                ["gemini", "Gemini TTS", "Expressive, emotion-steerable"],
                ["edge", "Edge Neural", "Free, no API key"],
              ] as const).map(([id, name, hint]) => (
                <button key={id} type="button" onClick={() => setProvider(id)}
                  className={`rounded-xl border p-3 text-left transition ${provider === id
                    ? "border-indigo-400/60 bg-indigo-500/10" : "border-white/10 bg-ink-800/60 hover:border-white/20"}`}>
                  <span className="block text-sm font-medium text-slate-100">{name}</span>
                  <span className="block text-xs text-slate-400">{hint}</span>
                </button>
              ))}
            </div>
          </div>

          <Toggle checked={review} onChange={setReview} label="Review translation before dubbing"
            hint="Pause to edit lines and choose a voice for each speaker." />
          <Toggle checked={lipsync} onChange={setLipsync} label="AI lip sync (beta)"
            hint="Re-render mouth movements to match the English audio. Slower." />

          {error && <ErrorNote>{error}</ErrorNote>}

          <button className="btn-primary w-full py-3" disabled={!file || busy} onClick={submit}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
            {busy ? "Uploading…" : "Start dubbing"}
          </button>
        </div>
      </Card>
    </div>
  );
}
