import { Maximize2, Minus, Plus, RotateCw } from "lucide-react";
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type MouseEvent, type ReactNode, type RefObject } from "react";
import { fmtTC, fmtTime } from "../api";
import { speakerHex, speakerLabel } from "./ui";

export interface ZhBlock { id: number; start: number; end: number; speaker: string; text: string }
export interface EnBlock {
  id: number; start: number; duration: number; budget: number; speaker: string; text: string;
  tempo: number; rewrites: number; emotion: string; planned: boolean;
}

const LABEL_W = 92;
const ROWS = { ruler: 26, zh: 54, en: 66, vad: 22 };
const STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];

/** Multi-track timeline: source dialogue, dubbed dialogue (with fit diagnostics) and a
 *  speech/silence lane. The playhead is driven straight from the <video> element. */
export default function Timeline({ duration, zh, en, videoRef, activeZh, activeEn, onSeek }: {
  duration: number; zh: ZhBlock[]; en: EnBlock[]; videoRef: RefObject<HTMLVideoElement | null>;
  activeZh: number | null; activeEn: number | null; onSeek: (t: number) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const playhead = useRef<HTMLDivElement>(null);
  const [viewW, setViewW] = useState(800);
  const [zoom, setZoom] = useState<number | null>(null); // px/s; null = fit to width
  const [follow, setFollow] = useState(true);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewW(el.clientWidth));
    ro.observe(el);
    setViewW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const fit = Math.max(8, (viewW - 16) / Math.max(duration, 1));
  const pps = zoom ?? fit;
  const width = Math.ceil(duration * pps) + 16;

  // Playhead: rAF loop writes the transform directly (no React re-render per frame).
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      const el = scroller.current;
      if (v && playhead.current) {
        const x = v.currentTime * pps;
        playhead.current.style.transform = `translateX(${x}px)`;
        if (follow && el && !v.paused) {
          const left = el.scrollLeft;
          if (x < left || x > left + el.clientWidth - 48) el.scrollLeft = Math.max(0, x - el.clientWidth * 0.2);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pps, follow, videoRef]);

  const setZoomAround = (next: number) => {
    const el = scroller.current;
    const v = videoRef.current;
    const clamped = Math.min(240, Math.max(fit, next));
    setZoom(clamped <= fit + 0.01 ? null : clamped);
    if (el && v) requestAnimationFrame(() => {
      el.scrollLeft = Math.max(0, v.currentTime * clamped - el.clientWidth / 2);
    });
  };
  const zoomRef = useRef(setZoomAround);
  zoomRef.current = setZoomAround;
  const ppsRef = useRef(pps);
  ppsRef.current = pps;

  // Ctrl/⌘ + wheel zooms. Native non-passive listener so the page doesn't zoom too.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const onWheel = (e: globalThis.WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      zoomRef.current(ppsRef.current * (e.deltaY < 0 ? 1.2 : 1 / 1.2));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const seekFromRuler = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(duration, (e.clientX - rect.left) / pps)));
  };

  const totalH = ROWS.ruler + ROWS.zh + ROWS.en + ROWS.vad + 12;

  return (
    <div className="panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-3">
          <span className="eyebrow">Timeline</span>
          <Legend />
        </div>
        <div className="flex items-center gap-1">
          <label className="mr-2 flex cursor-pointer items-center gap-1.5 text-xs text-muted">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)}
              className="accent-[var(--signal)]" /> follow
          </label>
          <button className="btn-ghost px-2 py-1" aria-label="Zoom out" onClick={() => setZoomAround(pps / 1.5)}><Minus className="size-3.5" /></button>
          <button className="btn-ghost px-2 py-1 tc text-xs" aria-label="Fit to width" onClick={() => setZoom(null)}>
            {zoom ? `${Math.round(pps)}px/s` : <Maximize2 className="size-3.5" />}
          </button>
          <button className="btn-ghost px-2 py-1" aria-label="Zoom in" onClick={() => setZoomAround(pps * 1.5)}><Plus className="size-3.5" /></button>
        </div>
      </div>

      <div className="flex">
        {/* track labels */}
        <div className="shrink-0 border-r border-line bg-panel-2" style={{ width: LABEL_W }}>
          <div style={{ height: ROWS.ruler }} className="border-b border-line" />
          <TrackLabel h={ROWS.zh} name="原声 ZH" sub="source" color="var(--zh)" />
          <TrackLabel h={ROWS.en} name="DUB EN" sub="voiced" color="var(--en)" />
          <TrackLabel h={ROWS.vad} name="VAD" sub="" color="var(--faint)" />
        </div>

        <div ref={scroller} className="scroll-thin relative min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
          <div className="relative" style={{ width, height: totalH }}>
            <Ruler duration={duration} pps={pps} onSeek={seekFromRuler} />
            <Tracks zh={zh} en={en} pps={pps} activeZh={activeZh} activeEn={activeEn} onSeek={onSeek} duration={duration} />
            <div ref={playhead} className="pointer-events-none absolute left-2 top-0 z-20 h-full" style={{ width: 0 }}>
              <div className="absolute -left-[5px] top-0 size-0 border-x-[5px] border-t-[7px] border-x-transparent border-t-signal" />
              <div className="absolute left-0 top-0 h-full w-px bg-signal" style={{ boxShadow: "0 0 8px var(--signal)" }} />
            </div>
          </div>
        </div>
      </div>
      <div className="flex items-center justify-between border-t border-line px-4 py-1.5 font-mono text-[0.65rem] text-faint">
        <span>CTRL + SCROLL TO ZOOM · CLICK A LINE TO JUMP</span>
        <span className="flex items-center gap-1"><RotateCw className="size-3" /> = LLM rewrite · ×n = time-stretch</span>
      </div>
    </div>
  );
}

function Legend() {
  const item = (color: string, label: string, dashed = false) => (
    <span className="flex items-center gap-1.5 text-[0.7rem] text-muted">
      <span className="h-2.5 w-4 rounded-sm" style={dashed
        ? { border: `1.5px dashed ${color}` } : { background: color }} />
      {label}
    </span>
  );
  return (
    <span className="hidden flex-wrap items-center gap-3 md:flex">
      {item("var(--en)", "time budget", true)}
      {item("var(--bad)", "overflow")}
    </span>
  );
}

function TrackLabel({ h, name, sub, color }: { h: number; name: string; sub: string; color: string }) {
  return (
    <div className="flex flex-col justify-center border-b border-line px-3" style={{ height: h }}>
      <span className="flex items-center gap-1.5 font-mono text-[0.7rem] font-semibold" style={{ color }}>
        <span className="size-1.5 rounded-full" style={{ background: color }} />{name}
      </span>
      {sub && <span className="pl-3 font-mono text-[0.6rem] text-faint">{sub}</span>}
    </div>
  );
}

const Ruler = memo(function Ruler({ duration, pps, onSeek }: {
  duration: number; pps: number; onSeek: (e: MouseEvent<HTMLDivElement>) => void;
}) {
  const step = STEPS.find((s) => s * pps >= 64) ?? 120;
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let t = 0; t <= duration + 1e-6; t += step / 4) out.push(Math.round(t * 1000) / 1000);
    return out;
  }, [duration, step]);
  return (
    <div className="absolute left-2 right-0 top-0 cursor-pointer border-b border-line" style={{ height: ROWS.ruler }}
      onClick={onSeek} role="slider" aria-label="Seek" aria-valuemin={0} aria-valuemax={duration} tabIndex={-1}>
      {ticks.map((t, i) => {
        const major = i % 4 === 0;
        return (
          <div key={t} className="absolute bottom-0" style={{ left: t * pps }}>
            <div className="w-px bg-line-strong" style={{ height: major ? 9 : 4 }} />
            {major && <span className="tc absolute bottom-2.5 left-1 whitespace-nowrap text-[0.6rem] text-muted">{fmtTime(t)}</span>}
          </div>
        );
      })}
    </div>
  );
});

const Tracks = memo(function Tracks({ zh, en, pps, activeZh, activeEn, onSeek, duration }: {
  zh: ZhBlock[]; en: EnBlock[]; pps: number; activeZh: number | null; activeEn: number | null;
  onSeek: (t: number) => void; duration: number;
}) {
  const zhTop = ROWS.ruler;
  const enTop = zhTop + ROWS.zh;
  const vadTop = enTop + ROWS.en;

  const silences = useMemo(() => {
    const sorted = [...zh].sort((a, b) => a.start - b.start);
    const gaps: [number, number][] = [];
    let cursor = 0;
    for (const s of sorted) {
      if (s.start - cursor > 1.5) gaps.push([cursor, s.start]);
      cursor = Math.max(cursor, s.end);
    }
    if (duration - cursor > 1.5) gaps.push([cursor, duration]);
    return gaps;
  }, [zh, duration]);

  return (
    <div className="absolute inset-0 left-2">
      {/* row separators */}
      {[enTop, vadTop, vadTop + ROWS.vad].map((y) => (
        <div key={y} className="absolute left-0 right-0 h-px bg-line" style={{ top: y }} />
      ))}

      {zh.map((s) => {
        const c = speakerHex(s.speaker);
        const active = s.id === activeZh;
        return (
          <button key={`zh${s.id}`} onClick={() => onSeek(s.start)}
            title={`${speakerLabel(s.speaker)} ${fmtTC(s.start)} – ${fmtTC(s.end)}\n${s.text}`}
            className="absolute overflow-hidden rounded-md px-1.5 text-left transition-[filter] hover:brightness-125"
            style={{ left: s.start * pps, width: Math.max(4, (s.end - s.start) * pps), top: zhTop + 7,
                     height: ROWS.zh - 14, background: `${c}${active ? "55" : "26"}`,
                     boxShadow: `inset 3px 0 0 ${c}${active ? ", 0 0 0 1.5px " + c : ""}` }}>
            <span className="block truncate pt-1 font-mono text-[0.58rem] font-semibold" style={{ color: c }}>
              {speakerLabel(s.speaker)}
            </span>
            <span className="block truncate font-zh text-[0.72rem] text-text">{s.text}</span>
          </button>
        );
      })}

      {en.map((l) => {
        const c = speakerHex(l.speaker);
        const active = l.id === activeEn;
        const over = l.duration > l.budget + 0.05;
        const fitW = Math.min(l.duration, l.budget) * pps;
        return (
          <div key={`en${l.id}`}>
            {/* budget bracket */}
            <div className="pointer-events-none absolute rounded-md border-[1.5px] border-dashed"
              style={{ left: l.start * pps, width: l.budget * pps, top: enTop + 5, height: ROWS.en - 10,
                       borderColor: "color-mix(in srgb, var(--en) 45%, transparent)" }} />
            <button onClick={() => onSeek(l.start)}
              title={`${fmtTC(l.start)} · ${l.duration.toFixed(2)}s of ${l.budget.toFixed(2)}s budget${
                l.tempo > 1.001 ? ` · stretched ×${l.tempo.toFixed(2)}` : ""}${l.rewrites ? ` · rewritten ${l.rewrites}×` : ""}\n${l.text}`}
              className="absolute overflow-hidden rounded-md px-1.5 text-left transition-[filter] hover:brightness-125"
              style={{ left: l.start * pps, width: Math.max(4, l.duration * pps), top: enTop + 9,
                       height: ROWS.en - 18, opacity: l.planned ? 0.6 : 1,
                       background: `linear-gradient(90deg, color-mix(in srgb, var(--en) ${active ? 45 : 24}%, transparent) ${fitW}px, color-mix(in srgb, var(--bad) 55%, transparent) ${fitW}px)`,
                       boxShadow: `inset 3px 0 0 ${c}${active ? ", 0 0 0 1.5px var(--en)" : ""}${over ? ", inset -2px 0 0 var(--bad)" : ""}` }}>
              <span className="flex items-center gap-1 pt-1">
                {l.tempo > 1.001 && <Badge color="var(--warn)">×{l.tempo.toFixed(2)}</Badge>}
                {l.rewrites > 0 && <Badge color="var(--signal)"><RotateCw className="size-2.5" />{l.rewrites}</Badge>}
                {l.planned && <Badge color="var(--muted)">est.</Badge>}
              </span>
              <span className="block truncate text-[0.72rem] font-medium text-text">{l.text}</span>
            </button>
          </div>
        );
      })}

      {/* speech / silence lane */}
      {zh.map((s) => (
        <div key={`v${s.id}`} className="absolute rounded-full"
          style={{ left: s.start * pps, width: Math.max(2, (s.end - s.start) * pps), top: vadTop + 8, height: 6,
                   background: "color-mix(in srgb, var(--zh) 60%, transparent)" }} />
      ))}
      {silences.map(([a, b]) => (b - a) * pps > 54 && (
        <span key={`s${a}`} className="absolute truncate text-center font-mono text-[0.55rem] tracking-widest text-faint"
          style={{ left: a * pps, width: (b - a) * pps, top: vadTop + 5 }}>SILENCE</span>
      ))}
    </div>
  );
});

function Badge({ children, color }: { children: ReactNode; color: string }) {
  return (
    <span className="inline-flex items-center gap-0.5 rounded px-1 font-mono text-[0.55rem] font-semibold leading-4"
      style={{ color, background: `color-mix(in srgb, ${color} 16%, transparent)` }}>{children}</span>
  );
}
