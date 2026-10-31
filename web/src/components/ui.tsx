import type { ReactNode } from "react";
import type { JobState } from "../api";

const SPEAKER_COLORS = [
  "bg-sky-500/15 text-sky-300 ring-sky-400/30",
  "bg-fuchsia-500/15 text-fuchsia-300 ring-fuchsia-400/30",
  "bg-amber-500/15 text-amber-300 ring-amber-400/30",
  "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
  "bg-rose-500/15 text-rose-300 ring-rose-400/30",
  "bg-violet-500/15 text-violet-300 ring-violet-400/30",
];

export function speakerColor(speaker: string) {
  const n = parseInt(speaker.replace(/\D/g, ""), 10);
  return SPEAKER_COLORS[(Number.isNaN(n) ? 0 : n) % SPEAKER_COLORS.length];
}

export function SpeakerChip({ speaker }: { speaker: string }) {
  const label = speaker.replace("SPEAKER_", "S");
  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${speakerColor(speaker)}`}>
      {label}
    </span>
  );
}

const STATE_STYLES: Record<JobState, string> = {
  queued: "bg-slate-500/15 text-slate-300 ring-slate-400/30",
  running: "bg-indigo-500/15 text-indigo-300 ring-indigo-400/30",
  awaiting_review: "bg-amber-500/15 text-amber-300 ring-amber-400/30",
  done: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
  failed: "bg-rose-500/15 text-rose-300 ring-rose-400/30",
  idle: "bg-slate-500/15 text-slate-400 ring-slate-400/20",
};

export function StateBadge({ state }: { state: JobState }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${STATE_STYLES[state]}`}>
      {state === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
      {state.replace("_", " ")}
    </span>
  );
}

export function Card({ title, icon, actions, children, className = "" }: {
  title?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card p-5 ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-100">
            {icon}{title}
          </h2>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function ProgressBar({ value, tone = "indigo" }: { value: number; tone?: "indigo" | "emerald" | "rose" }) {
  const colors = {
    indigo: "from-indigo-500 to-fuchsia-500",
    emerald: "from-emerald-500 to-teal-400",
    rose: "from-rose-500 to-orange-400",
  }[tone];
  return (
    <div className="h-2 overflow-hidden rounded-full bg-white/5">
      <div className={`h-full rounded-full bg-linear-to-r ${colors} transition-[width] duration-500`}
        style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }} />
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }: {
  checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string;
}) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4 rounded-xl border border-white/5 bg-ink-800/60 p-3">
      <span>
        <span className="block text-sm font-medium text-slate-100">{label}</span>
        {hint && <span className="mt-0.5 block text-xs text-slate-400">{hint}</span>}
      </span>
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition ${checked ? "bg-indigo-500" : "bg-ink-600"}`}>
        <span className={`absolute top-0.5 size-5 rounded-full bg-white shadow transition-all ${checked ? "left-5.5" : "left-0.5"}`} />
      </button>
    </label>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
      {children}
    </div>
  );
}
