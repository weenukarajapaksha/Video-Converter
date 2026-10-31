import { Check, CircleDashed, Loader2, RotateCcw, X } from "lucide-react";
import { STAGES, fmtDuration, type JobSummary } from "../api";

export default function StageStepper({ job, liveStage, liveMessage }: {
  job: JobSummary; liveStage: string | null; liveMessage: string;
}) {
  const lipsync = Boolean(job.options.lipsync);
  const stages = STAGES.filter((s) => s.name !== "lipsync" || lipsync);
  const active = job.state === "running" || job.state === "queued";

  return (
    <ol className="relative space-y-1">
      {stages.map((s, i) => {
        const rec = job.stages[s.name];
        const running = active && (rec?.status === "running" || liveStage === s.name)
          && rec?.status !== "done";
        const status = running ? "running" : rec?.status ?? "pending";
        const last = i === stages.length - 1;
        return (
          <li key={s.name} className="relative flex gap-3 pb-3">
            {!last && (
              <span className={`absolute top-8 left-[15px] h-[calc(100%-1.5rem)] w-px ${
                status === "done" ? "bg-emerald-400/40" : "bg-white/10"}`} />
            )}
            <span className={`z-10 grid size-8 shrink-0 place-items-center rounded-full ring-1 ${
              status === "done" ? "bg-emerald-500/15 text-emerald-300 ring-emerald-400/40"
              : status === "running" ? "bg-indigo-500/20 text-indigo-200 ring-indigo-400/60 shadow-lg shadow-indigo-500/30"
              : status === "failed" ? "bg-rose-500/15 text-rose-300 ring-rose-400/40"
              : "bg-ink-800 text-slate-500 ring-white/10"}`}>
              {status === "done" ? <Check className="size-4" />
                : status === "running" ? <Loader2 className="size-4 animate-spin" />
                : status === "failed" ? <X className="size-4" />
                : <CircleDashed className="size-4" />}
            </span>
            <div className="min-w-0 flex-1 pt-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className={`text-sm font-medium ${status === "pending" ? "text-slate-500" : "text-slate-100"}`}>
                  {s.label}
                </span>
                <span className="flex shrink-0 items-center gap-2 text-xs text-slate-500">
                  {rec && rec.attempts > 1 && (
                    <span className="flex items-center gap-1 text-amber-300"><RotateCcw className="size-3" />{rec.attempts}</span>
                  )}
                  {status === "done" && fmtDuration(rec?.duration_s)}
                </span>
              </div>
              <p className="truncate text-xs text-slate-500">
                {status === "running" && liveMessage ? liveMessage
                  : status === "failed" && rec?.error ? <span className="text-rose-300">{rec.error}</span>
                  : s.detail}
              </p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
