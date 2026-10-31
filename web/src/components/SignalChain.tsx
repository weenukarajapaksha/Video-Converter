import { Check, Loader2, X } from "lucide-react";
import { Fragment } from "react";
import { STAGES, fmtDuration, type JobSummary } from "../api";

type NodeState = "pending" | "running" | "done" | "failed";

function nodeState(job: JobSummary, name: string, liveStage: string | null): NodeState {
  const rec = job.stages[name];
  if (liveStage === name && (job.state === "running" || job.state === "queued")) return "running";
  if (!rec) return "pending";
  if (rec.status === "done") return "done";
  if (rec.status === "failed") return "failed";
  if (rec.status === "running") return job.state === "running" ? "running" : "failed";
  return "pending";
}

const COLOR: Record<NodeState, string> = {
  pending: "var(--faint)", running: "var(--signal)", done: "var(--ok)", failed: "var(--bad)",
};

/** Pipeline stages drawn as a patch chain: nodes joined by "audio cables". */
export default function SignalChain({ job, liveStage, liveMessage }: {
  job: JobSummary; liveStage: string | null; liveMessage: string;
}) {
  const stages = STAGES.filter((s) => s.name !== "lipsync" || job.options.lipsync || job.stages.lipsync);
  const states = stages.map((s) => nodeState(job, s.name, liveStage));

  return (
    <div className="scroll-thin overflow-x-auto pb-1">
      <ol className="flex min-w-max items-start px-1 pt-1">
        {stages.map((s, i) => {
          const st = states[i];
          const rec = job.stages[s.name];
          const next = states[i + 1];
          const cableLive = st === "done" && next === "running";
          return (
            <Fragment key={s.name}>
              <li className="flex w-[5.6rem] flex-col items-center text-center" title={`${s.label}: ${s.detail}`}>
                <span className={`relative grid size-11 place-items-center rounded-xl border-2 bg-panel-2 transition ${
                  st === "running" ? "node-live" : ""}`}
                  style={{ borderColor: COLOR[st], color: COLOR[st] }}>
                  {st === "done" ? <Check className="size-5" strokeWidth={3} />
                    : st === "failed" ? <X className="size-5" strokeWidth={3} />
                    : st === "running" ? <Loader2 className="size-5 animate-spin" />
                    : <span className="tc text-xs">{String(i + 1).padStart(2, "0")}</span>}
                  {rec && rec.attempts > 1 && (
                    <span className="absolute -right-2 -top-2 rounded-full border border-warn/50 bg-panel px-1.5 font-mono text-[0.6rem] text-warn"
                      title={`${rec.attempts} attempts`}>×{rec.attempts}</span>
                  )}
                </span>
                <span className={`mt-2 text-xs font-semibold ${st === "pending" ? "text-faint" : "text-text"}`}>{s.short}</span>
                <span className="tc text-[0.65rem] text-muted">
                  {st === "running" ? "live" : rec?.duration_s != null && st === "done" ? fmtDuration(rec.duration_s) : st === "failed" ? "error" : "—"}
                </span>
              </li>
              {i < stages.length - 1 && (
                <li aria-hidden="true" className="mt-[1.1rem] w-6 shrink-0 sm:w-8">
                  <svg viewBox="0 0 32 8" className="h-2 w-full overflow-visible" preserveAspectRatio="none">
                    <path d="M0 4 C 10 -1, 22 9, 32 4" fill="none" strokeWidth="2.5" strokeLinecap="round"
                      className={cableLive ? "cable-live" : ""}
                      stroke={st === "done" ? (cableLive ? "var(--signal)" : "var(--ok)") : "var(--line-strong)"} />
                  </svg>
                </li>
              )}
            </Fragment>
          );
        })}
      </ol>
      {liveStage && (job.state === "running" || job.state === "queued") && (
        <p className="mt-3 truncate font-mono text-xs text-muted">
          <span className="text-signal">▸ {liveStage}</span> {liveMessage}
        </p>
      )}
    </div>
  );
}

const SEGMENTS = 48;

/** Overall progress as a segmented, glowing VU meter. */
export function VuMeter({ value, state }: { value: number; state: string }) {
  const lit = Math.round(Math.max(0, Math.min(1, value)) * SEGMENTS);
  return (
    <div className="flex h-3.5 gap-[3px]" role="progressbar" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}>
      {Array.from({ length: SEGMENTS }, (_, i) => {
        const on = i < lit;
        const color = state === "failed" ? "var(--bad)"
          : i / SEGMENTS < 0.7 ? "var(--ok)" : i / SEGMENTS < 0.9 ? "var(--warn)" : "var(--signal)";
        return (
          <span key={i} className="flex-1 rounded-[2px] transition-colors duration-300"
            style={{ background: on ? color : "var(--panel-3)",
                     boxShadow: on ? `0 0 8px color-mix(in srgb, ${color} 55%, transparent)` : "none" }} />
        );
      })}
    </div>
  );
}
