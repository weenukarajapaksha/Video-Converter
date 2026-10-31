import { FileText, Loader2, Mars, Play, Save, Users, Venus, VenusAndMars } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { LOW_CONFIDENCE, api, fmtTC, type Translation, type VoiceMap, type VoiceOption } from "../api";
import { ErrorNote, Panel, SpeakerTag, speakerHex } from "./ui";

const EMOTIONS = ["neutral", "happy", "excited", "sad", "angry", "surprised", "fearful",
  "calm", "serious", "tender", "sarcastic", "worried"];

// Average conversational English pace used for the live timing estimate.
const WORDS_PER_SECOND = 2.6;

const estimateSeconds = (text: string) =>
  text.trim() ? text.trim().split(/\s+/).length / WORDS_PER_SECOND : 0;

function FitMeter({ est, budget }: { est: number; budget: number }) {
  const ratio = budget > 0 ? est / budget : 0;
  const color = ratio > 1.15 ? "var(--bad)" : ratio > 1 ? "var(--warn)" : "var(--ok)";
  return (
    <div className="w-full" title="Estimated speaking time vs. the time this line may occupy">
      <div className="relative h-2 overflow-hidden rounded-full bg-panel-3">
        <div className="h-full rounded-full transition-[width] duration-300"
          style={{ width: `${Math.min(100, ratio * 80)}%`, background: color }} />
        {/* the budget mark sits at 80% so overruns stay visible */}
        <div className="absolute inset-y-0 w-px bg-text/50" style={{ left: "80%" }} />
      </div>
      <div className="tc mt-1 flex justify-between text-[0.65rem]">
        <span style={{ color }}>~{est.toFixed(1)}s</span>
        <span className="text-faint">slot {budget.toFixed(1)}s</span>
      </div>
    </div>
  );
}

export default function ScriptEditor({ jobId, onGenerated }: { jobId: string; onGenerated: () => void }) {
  const [translation, setTranslation] = useState<Translation | null>(null);
  const [voices, setVoices] = useState<VoiceMap | null>(null);
  const [catalog, setCatalog] = useState<VoiceOption[]>([]);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<Map<number, number>>(new Map());

  useEffect(() => {
    api.translation(jobId).then(setTranslation).catch((e) => setError(e.message));
    api.transcript(jobId).then((t) => setConfidence(new Map(
      t.segments.filter((s) => s.confidence !== null).map((s) => [s.id, s.confidence!]))))
      .catch(() => { /* confidence markers are optional */ });
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

  if (!translation) return error ? <ErrorNote>{error}</ErrorNote> : null;

  return (
    <div className="space-y-6">
      {voices && (
        <Panel eyebrow={`Voice engine · ${voices.provider}`} title="Cast" icon={<Users className="size-4" />}>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {Object.values(voices.speakers).map((p) => {
              const Gender = p.gender === "male" ? Mars : p.gender === "female" ? Venus : VenusAndMars;
              const color = speakerHex(p.speaker);
              return (
                <div key={p.speaker} className="relative overflow-hidden rounded-xl border border-line bg-panel-2 p-4">
                  <div className="absolute inset-x-0 top-0 h-1" style={{ background: color }} />
                  <div className="mb-3 flex items-center gap-3">
                    <SpeakerTag speaker={p.speaker} size="lg" />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 font-display font-bold">
                        <Gender className="size-4" style={{ color }} /> {p.gender}
                      </div>
                      <div className="tc text-xs text-muted">
                        {p.median_f0_hz ? `${Math.round(p.median_f0_hz)} Hz` : "— Hz"} · {p.speaking_time_s.toFixed(1)}s talk
                      </div>
                    </div>
                  </div>
                  <label className="eyebrow mb-1 block" htmlFor={`voice-${p.speaker}`}>Voice</label>
                  <select id={`voice-${p.speaker}`} className="field mb-2" value={p.voice}
                    onChange={(e) => editVoice(p.speaker, { voice: e.target.value })}>
                    {!catalog.some((v) => v.id === p.voice) && <option value={p.voice}>{p.voice}</option>}
                    {(["male", "female"] as const).map((g) => (
                      <optgroup key={g} label={g === "male" ? "Male voices" : "Female voices"}>
                        {catalog.filter((v) => v.gender === g).map((v) => (
                          <option key={v.id} value={v.id}>{v.id} — {v.description}</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                  <input className="field" placeholder="Delivery hint, e.g. warm, slightly tired"
                    aria-label={`Delivery style for ${p.speaker}`}
                    value={p.style} onChange={(e) => editVoice(p.speaker, { style: e.target.value })} />
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      <Panel
        eyebrow={`${translation.segments.length} lines · 中文 → ${translation.target_language}`}
        title="Script" icon={<FileText className="size-4" />}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {overBudget > 0 && (
              <span className="rounded-full border border-warn/40 bg-warn/10 px-2.5 py-1 font-mono text-[0.68rem] text-warn">
                {overBudget} line{overBudget > 1 ? "s" : ""} may run long
              </span>
            )}
            <button className="btn-ghost" disabled={!dirty || busy} onClick={() => save(false)}>
              <Save className="size-4" /> Save
            </button>
            <button className="btn-signal" disabled={busy} onClick={() => save(true)}>
              {busy ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              Save &amp; voice it
            </button>
          </div>
        }
      >
        {error && <div className="mb-4"><ErrorNote>{error}</ErrorNote></div>}
        <ol className="space-y-3">
          {translation.segments.map((s) => {
            const est = estimateSeconds(s.text);
            const conf = confidence.get(s.id);
            const low = conf !== undefined && conf < LOW_CONFIDENCE;
            return (
              <li key={s.id} className="grid gap-3 rounded-xl border border-line bg-panel-2 p-3.5 md:grid-cols-[8.5rem_1fr_9.5rem]"
                style={{ boxShadow: `inset 3px 0 0 ${speakerHex(s.speaker)}` }}>
                <div className="flex items-start gap-2.5 md:flex-col">
                  <SpeakerTag speaker={s.speaker} />
                  <div className="tc text-[0.7rem] leading-relaxed text-muted">
                    {fmtTC(s.start)}<br /><span className="text-faint">{fmtTC(s.end)}</span>
                  </div>
                </div>
                <div className="min-w-0 space-y-2">
                  <p className={`font-zh text-[0.95rem] leading-relaxed text-zh ${low ? "low-conf" : ""}`}
                    title={conf !== undefined ? `ASR confidence ${conf.toFixed(2)}` : undefined}>
                    {s.source_text}
                    {low && <span className="ml-2 align-middle font-mono text-[0.65rem] text-warn">
                      ASR {conf!.toFixed(2)} · check</span>}
                  </p>
                  <textarea className="field min-h-[3.25rem] resize-y text-[0.95rem] leading-relaxed" value={s.text}
                    aria-label={`English line ${s.id}`}
                    onChange={(e) => editSegment(s.id, { text: e.target.value })} />
                </div>
                <div className="space-y-3">
                  <select className="field py-1.5 text-xs" value={s.emotion} aria-label={`Emotion for line ${s.id}`}
                    onChange={(e) => editSegment(s.id, { emotion: e.target.value })}>
                    {[...new Set([s.emotion, ...EMOTIONS])].map((em) => <option key={em}>{em}</option>)}
                  </select>
                  <FitMeter est={est} budget={s.max_duration_s} />
                </div>
              </li>
            );
          })}
        </ol>
      </Panel>
    </div>
  );
}
