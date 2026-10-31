import { Columns2, MonitorPlay, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { api, fmtTC, type DubTrack, type JobSummary, type Transcript, type Translation } from "../api";
import Timeline, { type EnBlock, type ZhBlock } from "./Timeline";
import { Panel, SpeakerTag } from "./ui";

const WORDS_PER_SECOND = 2.6;

/** Current playback time of a <video>, sampled per animation frame while playing. */
function useVideoClock(ref: RefObject<HTMLVideoElement | null>, src: string) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let raf = 0;
    let last = -1;
    const loop = () => {
      if (Math.abs(v.currentTime - last) > 0.03) { last = v.currentTime; setT(last); }
      if (!v.paused) raf = requestAnimationFrame(loop);
    };
    const kick = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); };
    const events = ["play", "seeked", "timeupdate", "loadedmetadata"] as const;
    events.forEach((e) => v.addEventListener(e, kick));
    kick();
    return () => { cancelAnimationFrame(raf); events.forEach((e) => v.removeEventListener(e, kick)); };
  }, [ref, src]);
  return t;
}

export function buildTracks(transcript: Transcript | null, translation: Translation | null,
                            dub: DubTrack | null): { zh: ZhBlock[]; en: EnBlock[] } {
  const zh: ZhBlock[] = (transcript?.segments ?? []).map((s) => ({
    id: s.id, start: s.start, end: s.end, speaker: s.speaker, text: s.text }));
  const byId = new Map((translation?.segments ?? []).map((s) => [s.id, s]));
  const en: EnBlock[] = dub
    ? dub.lines.map((l) => {
        const t = byId.get(l.id);
        return { id: l.id, start: l.start, duration: l.final_duration_s, budget: t?.max_duration_s ?? l.final_duration_s,
                 speaker: t?.speaker ?? "SPEAKER_01", text: l.text, tempo: l.tempo, rewrites: l.rewrites,
                 emotion: t?.emotion ?? "neutral", planned: false };
      })
    : (translation?.segments ?? []).map((s) => ({
        id: s.id, start: s.start, budget: s.max_duration_s, speaker: s.speaker, text: s.text,
        duration: Math.max(0.3, s.text.trim().split(/\s+/).length / WORDS_PER_SECOND),
        tempo: 1, rewrites: 0, emotion: s.emotion, planned: true }));
  return { zh, en };
}

export default function Monitor({ job, transcript, translation, dub }: {
  job: JobSummary; transcript: Transcript | null; translation: Translation | null; dub: DubTrack | null;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const dubbedName = job.artifacts.includes("output_lipsync.mp4") ? "output_lipsync.mp4" : "output.mp4";
  const hasDub = job.artifacts.includes(dubbedName);
  const [source, setSource] = useState<"dub" | "orig">(hasDub ? "dub" : "orig");
  const [mode, setMode] = useState<"monitor" | "split">("monitor");
  const src = api.fileUrl(job.job_id, source === "dub" && hasDub ? dubbedName : "input.mp4");
  const resume = useRef<{ t: number; playing: boolean } | null>(null);
  const t = useVideoClock(video, src);

  const { zh, en } = useMemo(() => buildTracks(transcript, translation, dub), [transcript, translation, dub]);
  const duration = job.metadata?.duration_s ?? Math.max(1, ...zh.map((s) => s.end));
  const activeZh = zh.find((s) => t >= s.start && t < s.end) ?? null;
  const activeEn = en.find((l) => !l.planned && t >= l.start && t < l.start + l.duration) ?? null;

  // A/B: swap the source but keep position and play state.
  const switchSource = (next: "dub" | "orig") => {
    const v = video.current;
    if (v) resume.current = { t: v.currentTime, playing: !v.paused };
    setSource(next);
  };
  const onLoaded = () => {
    const v = video.current;
    if (!v || !resume.current) return;
    v.currentTime = resume.current.t;
    if (resume.current.playing) void v.play();
    resume.current = null;
  };

  const seek = (time: number) => {
    const v = video.current;
    if (!v) return;
    v.currentTime = time;
    void v.play();
  };

  return (
    <div className="space-y-4">
      <Panel
        eyebrow="Studio monitor" title={mode === "monitor" ? "Playback & subtitles" : "Original vs. dubbed"}
        icon={<MonitorPlay className="size-4" />} bodyClass="p-0"
        actions={
          <div className="flex items-center gap-2">
            {mode === "monitor" && hasDub && (
              <div className="flex rounded-lg border border-line bg-panel-2 p-0.5 font-mono text-xs" role="group" aria-label="Audio source">
                {(["orig", "dub"] as const).map((s) => (
                  <button key={s} onClick={() => switchSource(s)} aria-pressed={source === s}
                    className={`rounded-md px-2.5 py-1 transition ${source === s
                      ? s === "dub" ? "bg-en text-bg" : "bg-zh text-bg" : "text-muted hover:text-text"}`}>
                    {s === "orig" ? "A · 原声" : "B · DUB"}
                  </button>
                ))}
              </div>
            )}
            {hasDub && (
              <button className="btn-ghost px-2.5 py-1.5 text-xs" onClick={() => setMode(mode === "monitor" ? "split" : "monitor")}>
                {mode === "monitor" ? <Columns2 className="size-3.5" /> : <MonitorPlay className="size-3.5" />}
                {mode === "monitor" ? "Side by side" : "Monitor"}
              </button>
            )}
          </div>
        }>
        {mode === "monitor" ? (
          <div className="grid gap-0 lg:grid-cols-[1.6fr_1fr]">
            <div className="bg-black">
              <video key={src} ref={video} src={src} controls playsInline preload="metadata" onLoadedMetadata={onLoaded}
                className="aspect-video w-full bg-black" />
            </div>
            <SubtitleDeck t={t} zh={activeZh} en={activeEn} source={source} />
          </div>
        ) : (
          <SideBySide job={job} dubbedName={dubbedName} />
        )}
      </Panel>

      {mode === "monitor" && (zh.length > 0 || en.length > 0) && (
        <Timeline duration={duration} zh={zh} en={en} videoRef={video}
          activeZh={activeZh?.id ?? null} activeEn={activeEn?.id ?? null} onSeek={seek} />
      )}
    </div>
  );
}

function SubtitleDeck({ t, zh, en, source }: { t: number; zh: ZhBlock | null; en: EnBlock | null; source: string }) {
  const words = en ? en.text.split(/\s+/) : [];
  const frac = en ? Math.min(1, (t - en.start) / Math.max(0.01, en.duration)) : 0;
  const lit = Math.ceil(frac * words.length);
  const speaker = en?.speaker ?? zh?.speaker;

  return (
    <div className="flex min-h-56 flex-col justify-between gap-4 border-t border-line p-5 lg:border-l lg:border-t-0">
      <div className="flex items-center justify-between">
        <span className="tc text-2xl font-medium tracking-tight">{fmtTC(t)}</span>
        <span className="eyebrow">{source === "dub" ? "Monitoring DUB" : "Monitoring source"}</span>
      </div>

      <div className="space-y-4">
        {speaker ? (
          <div className="flex items-center gap-2">
            <SpeakerTag speaker={speaker} />
            {en && <span className="rounded border border-line px-1.5 font-mono text-[0.65rem] uppercase text-muted">{en.emotion}</span>}
            {en && en.tempo > 1.001 && <span className="font-mono text-[0.65rem] text-warn">×{en.tempo.toFixed(2)}</span>}
          </div>
        ) : (
          <div className="font-mono text-xs tracking-[0.3em] text-faint">— SILENCE —</div>
        )}
        <p className="min-h-[1.75rem] font-zh text-lg leading-snug text-zh" aria-live="polite">{zh?.text ?? ""}</p>
        <p className="min-h-[3.5rem] font-display text-2xl font-bold leading-tight" aria-live="polite">
          {words.map((w, i) => (
            <span key={i} className="transition-colors duration-150"
              style={{ color: i < lit ? "var(--en)" : "var(--faint)" }}>{w} </span>
          ))}
        </p>
      </div>

      <div className="h-1 overflow-hidden rounded-full bg-panel-3">
        <div className="h-full bg-en transition-[width] duration-100" style={{ width: `${frac * 100}%` }} />
      </div>
    </div>
  );
}

function SideBySide({ job, dubbedName }: { job: JobSummary; dubbedName: string }) {
  const left = useRef<HTMLVideoElement>(null);
  const right = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);

  const toggle = () => {
    const [a, b] = [left.current, right.current];
    if (!a || !b) return;
    if (playing) { a.pause(); b.pause(); }
    else { b.currentTime = a.currentTime; a.muted = true; void a.play(); void b.play(); }
    setPlaying(!playing);
  };

  return (
    <div className="space-y-4 p-4">
      <div className="grid gap-4 md:grid-cols-2">
        {[{ ref: left, name: "input.mp4", label: "A · Original 原声", color: "var(--zh)" },
          { ref: right, name: dubbedName, label: "B · English dub", color: "var(--en)" }].map((v) => (
          <figure key={v.name} className="space-y-2">
            <video ref={v.ref} src={api.fileUrl(job.job_id, v.name)} controls playsInline preload="metadata"
              onEnded={() => setPlaying(false)} className="aspect-video w-full rounded-xl bg-black" />
            <figcaption className="flex items-center gap-2 font-mono text-xs" style={{ color: v.color }}>
              <span className="size-1.5 rounded-full" style={{ background: v.color }} />{v.label}
            </figcaption>
          </figure>
        ))}
      </div>
      <button className="btn-signal" onClick={toggle}>
        {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        {playing ? "Pause both" : "Sync play (original muted)"}
      </button>
    </div>
  );
}
