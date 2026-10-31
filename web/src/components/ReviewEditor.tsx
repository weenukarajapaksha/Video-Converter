import { Loader2, Mic2, PencilLine, Play, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, fmtTime, type Translation, type VoiceMap, type VoiceOption } from "../api";
import { Card, ErrorNote, SpeakerChip } from "./ui";

const EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "surprised", "fearful",
  "curious", "calm", "serious", "tender", "sarcastic", "urgent"];

// Average conversational English pace used for the live timing estimate.
const WORDS_PER_SECOND = 2.6;

const estimateSeconds = (text: string) =>
  text.trim() ? text.trim().split(/\s+/).length / WORDS_PER_SECOND : 0;

export default function ReviewEditor({ jobId, canGenerate, onGenerated }: {
  jobId: string; canGenerate: boolean; onGenerated: () => void;
}) {
  const [translation, setTranslation] = useState<Translation | null>(null);
  const [voices, setVoices] = useState<VoiceMap | null>(null);
  const [catalog, setCatalog] = useState<VoiceOption[]>([]);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.translation(jobId).then(setTranslation).catch((e) => setError(e.message));
    api.voices(jobId).then((v) => {
      setVoices(v);
      return api.voiceCatalog(v.provider).then(setCatalog);
    }).catch(() => { /* voices are optional until the voices stage exists */ });
  }, [jobId]);

  const overBudget = useMemo(() => translation?.segments.filter(
    (s) => estimateSeconds(s.text) > s.max_duration_s).length ?? 0, [translation]);

  const editSegment = (id: number, patch: { text?: string; emotion?: string }) => {
    setTranslation((t) => t && {
      ...t, segments: t.segments.map((s) => (s.id === id ? { ...s, ...patch } : s)),
    });
    setDirty(true);
  };

  const editVoice = (speaker: string, patch: { voice?: string; style?: string }) => {
    setVoices((v) => v && {
      ...v, speakers: { ...v.speakers, [speaker]: { ...v.speakers[speaker], ...patch } },
    });
    setDirty(true);
  };

  const save = async (generate: boolean) => {
    if (!translation) return;
    setBusy(true);
    setError(null);
    try {
      await api.saveTranslation(jobId, translation);
      if (voices) await api.saveVoices(jobId, voices);
      setDirty(false);
      if (generate) {
        await api.continueJob(jobId);
        onGenerated();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!translation) {
    return error ? <ErrorNote>{error}</ErrorNote> : null;
  }

  return (
    <div className="space-y-6">
      {voices && (
        <Card title="Speaker → voice mapping" icon={<Mic2 className="size-4 text-fuchsia-300" />}>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {Object.values(voices.speakers).map((p) => (
              <div key={p.speaker} className="space-y-3 rounded-xl border border-white/5 bg-ink-800/60 p-4">
                <div className="flex items-center justify-between">
                  <SpeakerChip speaker={p.speaker} />
                  <span className="text-xs text-slate-400">
                    {p.gender} · {p.median_f0_hz ? `${Math.round(p.median_f0_hz)} Hz` : "—"} · {p.speaking_time_s.toFixed(1)}s
                  </span>
                </div>
                <select className="input" value={p.voice}
                  onChange={(e) => editVoice(p.speaker, { voice: e.target.value })}>
                  {["male", "female"].map((g) => (
                    <optgroup key={g} label={g === "male" ? "Male voices" : "Female voices"}>
                      {catalog.filter((v) => v.gender === g).map((v) => (
                        <option key={v.id} value={v.id}>{v.id} — {v.description}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <input className="input" placeholder="Delivery style, e.g. warm, slightly tired"
                  value={p.style} onChange={(e) => editVoice(p.speaker, { style: e.target.value })} />
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card
        title={`Translation · ${translation.segments.length} lines`}
        icon={<PencilLine className="size-4 text-indigo-300" />}
        actions={
          <div className="flex items-center gap-2">
            {overBudget > 0 && (
              <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-xs text-amber-300 ring-1 ring-amber-400/30">
                {overBudget} line{overBudget > 1 ? "s" : ""} may run long
              </span>
            )}
            <button className="btn-ghost" disabled={!dirty || busy} onClick={() => save(false)}>
              <Save className="size-4" /> Save
            </button>
            {canGenerate && (
              <button className="btn-primary" disabled={busy} onClick={() => save(true)}>
                {busy ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                Save &amp; generate dub
              </button>
            )}
          </div>
        }
      >
        {error && <div className="mb-3"><ErrorNote>{error}</ErrorNote></div>}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[56rem] text-sm">
            <thead className="text-left text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="w-28 pb-3 font-medium">Time</th>
                <th className="w-16 pb-3 font-medium">Who</th>
                <th className="w-[28%] pb-3 font-medium">中文 source</th>
                <th className="pb-3 font-medium">{translation.target_language}</th>
                <th className="w-32 pb-3 font-medium">Emotion</th>
                <th className="w-28 pb-3 font-medium">Fit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 align-top">
              {translation.segments.map((s) => {
                const est = estimateSeconds(s.text);
                const ratio = s.max_duration_s > 0 ? est / s.max_duration_s : 0;
                const tone = ratio > 1.15 ? "bg-rose-500" : ratio > 1 ? "bg-amber-400" : "bg-emerald-400";
                return (
                  <tr key={s.id}>
                    <td className="py-3 pr-3 font-mono text-xs text-slate-400">
                      {fmtTime(s.start)}<br />→ {fmtTime(s.end)}
                    </td>
                    <td className="py-3 pr-3"><SpeakerChip speaker={s.speaker} /></td>
                    <td className="py-3 pr-4 leading-relaxed text-slate-300">{s.source_text}</td>
                    <td className="py-2 pr-3">
                      <textarea className="input min-h-16 resize-y leading-relaxed" value={s.text}
                        onChange={(e) => editSegment(s.id, { text: e.target.value })} />
                    </td>
                    <td className="py-2 pr-3">
                      <select className="input" value={s.emotion}
                        onChange={(e) => editSegment(s.id, { emotion: e.target.value })}>
                        {[...new Set([s.emotion, ...EMOTIONS])].map((em) => <option key={em}>{em}</option>)}
                      </select>
                    </td>
                    <td className="py-3">
                      <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                        <div className={`h-full ${tone}`} style={{ width: `${Math.min(100, ratio * 100)}%` }} />
                      </div>
                      <div className="mt-1 text-xs text-slate-400">
                        ~{est.toFixed(1)}s / {s.max_duration_s.toFixed(1)}s
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
