import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";
import type { JobState } from "../api";

// Speaker palette: mid-saturation hues that read on both the dark and paper themes.
// Eight distinct hues: diarization on real clips can report 5+ speakers.
const SPEAKER_HEX = ["#5aa9ff", "#ff7a9c", "#c58bff", "#ffb547", "#3ddbc4", "#9ad85a",
  "#ff8a4c", "#8fa3ff"];

export function speakerHex(speaker: string) {
  const n = parseInt(speaker.replace(/\D/g, ""), 10);
  return SPEAKER_HEX[((Number.isNaN(n) ? 1 : n) - 1 + SPEAKER_HEX.length) % SPEAKER_HEX.length];
}

export const speakerLabel = (speaker: string) => speaker.replace("SPEAKER_", "S");

export function SpeakerTag({ speaker, size = "sm" }: { speaker: string; size?: "sm" | "lg" }) {
  const color = speakerHex(speaker);
  const dim = size === "lg" ? "size-10 text-sm" : "size-6 text-[0.65rem]";
  return (
    <span
      className={`inline-grid shrink-0 place-items-center rounded-full font-mono font-semibold ${dim}`}
      style={{ background: `${color}22`, color, boxShadow: `inset 0 0 0 1.5px ${color}88` }}
      title={speaker}
    >
      {speakerLabel(speaker)}
    </span>
  );
}

export function Panel({ title, eyebrow, icon, actions, children, className = "", bodyClass = "p-5" }: {
  title?: ReactNode; eyebrow?: string; icon?: ReactNode; actions?: ReactNode; children: ReactNode;
  className?: string; bodyClass?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3.5">
          <div className="flex min-w-0 items-center gap-2.5">
            {icon && <span className="text-signal">{icon}</span>}
            <div className="min-w-0">
              {eyebrow && <div className="eyebrow">{eyebrow}</div>}
              {title && <h2 className="truncate font-display text-base font-bold tracking-tight">{title}</h2>}
            </div>
          </div>
          {actions}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

const STATE_STYLE: Record<JobState, { label: string; color: string; live?: boolean }> = {
  queued: { label: "Queued", color: "var(--muted)" },
  running: { label: "On air", color: "var(--signal)", live: true },
  awaiting_review: { label: "Review", color: "var(--warn)" },
  done: { label: "Mastered", color: "var(--ok)" },
  failed: { label: "Failed", color: "var(--bad)" },
  idle: { label: "Idle", color: "var(--muted)" },
};

export function StatusPill({ state }: { state: JobState }) {
  const s = STATE_STYLE[state] ?? STATE_STYLE.idle;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[0.68rem] font-medium uppercase tracking-wider"
      style={{ color: s.color, borderColor: `color-mix(in srgb, ${s.color} 40%, transparent)`,
               background: `color-mix(in srgb, ${s.color} 10%, transparent)` }}>
      <span className={`size-1.5 rounded-full ${s.live ? "tally" : ""}`} style={{ background: s.color }} />
      {s.label}
    </span>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="flex gap-2 rounded-lg border border-bad/30 bg-bad/10 px-3 py-2 text-sm text-bad">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0 break-words">{children}</div>
    </div>
  );
}

/** A studio "patch" switch: a labelled toggle with an LED. */
export function PatchToggle({ checked, onChange, label, hint, code }: {
  checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string; code: string;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
      className={`group flex w-full items-center gap-3 rounded-xl border p-3 text-left transition ${
        checked ? "border-signal/50 bg-signal/8" : "border-line bg-panel-2 hover:border-line-strong"}`}>
      <span className="grid w-11 shrink-0 place-items-center gap-1">
        <span className="size-2 rounded-full transition"
          style={{ background: checked ? "var(--signal)" : "var(--faint)",
                   boxShadow: checked ? "0 0 10px var(--signal)" : "none" }} />
        <span className="font-mono text-[0.6rem] tracking-wider text-muted">{code}</span>
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="block text-xs text-muted">{hint}</span>}
      </span>
    </button>
  );
}

export function Stat({ label, value, sub, accent }: {
  label: string; value: ReactNode; sub?: ReactNode; accent?: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel-2 p-3.5">
      <div className="eyebrow">{label}</div>
      <div className="mt-1.5 font-display text-2xl font-bold tracking-tight tc"
        style={accent ? { color: accent } : undefined}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Chip({ children, tone }: { children: ReactNode; tone?: "ok" | "bad" | "signal" }) {
  const color = tone ? `var(--${tone})` : undefined;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-[0.7rem] text-muted"
      style={color ? { color, borderColor: `color-mix(in srgb, ${color} 35%, transparent)` } : undefined}>
      {children}
    </span>
  );
}
