import { Clapperboard, Loader2, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { api, fmtDuration, fmtTC, type JobSummary } from "../api";
import { ErrorNote, StatusPill } from "../components/ui";

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () => api.listJobs().then(setJobs).catch((e) => setError(e.message));
    void load();
    const timer = window.setInterval(load, 4000);
    return () => window.clearInterval(timer);
  }, []);

  if (error) return <ErrorNote>{error}</ErrorNote>;
  if (!jobs) return <div className="grid place-items-center py-32 text-muted"><Loader2 className="size-6 animate-spin" /></div>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <span className="eyebrow">Archive</span>
          <h1 className="font-display text-3xl font-extrabold tracking-tight">Sessions</h1>
        </div>
        <a href="#/" className="btn-signal"><Plus className="size-4" /> New session</a>
      </div>

      {jobs.length === 0 ? (
        <div className="panel grid place-items-center gap-3 py-20 text-center">
          <Clapperboard className="size-8 text-signal" />
          <p className="font-display text-lg font-bold">No takes yet</p>
          <p className="text-sm text-muted">Upload a Chinese video to start your first dubbing session.</p>
        </div>
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {jobs.map((j) => {
            const running = j.state === "running" || j.state === "queued";
            return (
              <li key={j.job_id}>
                <a href={`#/jobs/${j.job_id}`}
                  className="panel group block overflow-hidden transition hover:-translate-y-0.5 hover:border-line-strong">
                  <div className="relative aspect-video bg-black">
                    <video src={`${api.fileUrl(j.job_id, "input.mp4")}#t=1`} preload="metadata" muted playsInline
                      className="size-full object-cover opacity-80 transition group-hover:opacity-100" />
                    <div className="absolute left-3 top-3 rounded-full bg-black/70 backdrop-blur-sm"><StatusPill state={j.state} /></div>
                    {j.metadata && (
                      <span className="tc absolute bottom-3 right-3 rounded bg-black/70 px-1.5 py-0.5 text-[0.7rem] text-white">
                        {fmtTC(j.metadata.duration_s).slice(3)}
                      </span>
                    )}
                    {running && (
                      <div className="absolute inset-x-0 bottom-0 h-1 bg-black/50">
                        <div className="h-full bg-signal transition-[width]" style={{ width: `${j.progress * 100}%` }} />
                      </div>
                    )}
                  </div>
                  <div className="space-y-1.5 p-4">
                    <p className="truncate font-display font-bold">{j.input_name.replace(/\.mp4$/i, "")}</p>
                    <div className="flex flex-wrap justify-between gap-2 font-mono text-[0.7rem] text-muted">
                      <span>{new Date(j.created_at).toLocaleString()}</span>
                      {j.total_duration_s > 0 && <span>{fmtDuration(j.total_duration_s)}</span>}
                    </div>
                    {running && <p className="truncate font-mono text-[0.7rem] text-signal">▸ {j.stage} {j.message}</p>}
                  </div>
                </a>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
