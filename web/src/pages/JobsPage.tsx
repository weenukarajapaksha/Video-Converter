import { FileVideo, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { api, fmtDuration, fmtUsd, type JobSummary } from "../api";
import { Card, ErrorNote, ProgressBar, StateBadge } from "../components/ui";

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.listJobs().then(setJobs).catch((e) => setError(e.message));

  useEffect(() => {
    load();
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, []);

  return (
    <Card title="Dubbing jobs" icon={<FileVideo className="size-4 text-indigo-300" />}
      actions={<button className="btn-ghost" onClick={load}><RefreshCw className="size-4" />Refresh</button>}>
      {error && <ErrorNote>{error}</ErrorNote>}
      {!jobs ? (
        <div className="grid place-items-center py-16 text-slate-500"><Loader2 className="size-6 animate-spin" /></div>
      ) : jobs.length === 0 ? (
        <p className="py-16 text-center text-slate-500">
          No jobs yet. <a href="#/" className="text-indigo-300 hover:underline">Upload a video</a> to start.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="pb-3 font-medium">Video</th>
                <th className="pb-3 font-medium">Status</th>
                <th className="pb-3 font-medium">Progress</th>
                <th className="pb-3 font-medium">Length</th>
                <th className="pb-3 font-medium">Processing</th>
                <th className="pb-3 font-medium">Cost</th>
                <th className="pb-3 font-medium">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {jobs.map((j) => (
                <tr key={j.job_id} className="cursor-pointer transition hover:bg-white/[0.03]"
                  onClick={() => { window.location.hash = `/jobs/${j.job_id}`; }}>
                  <td className="py-3 pr-4">
                    <div className="font-medium text-slate-100">{j.input_name}</div>
                    <div className="font-mono text-xs text-slate-500">{j.job_id}</div>
                  </td>
                  <td className="py-3 pr-4"><StateBadge state={j.state} /></td>
                  <td className="w-40 py-3 pr-4">
                    <ProgressBar value={j.progress}
                      tone={j.state === "failed" ? "rose" : j.state === "done" ? "emerald" : "indigo"} />
                  </td>
                  <td className="py-3 pr-4 text-slate-300">{fmtDuration(j.metadata?.duration_s)}</td>
                  <td className="py-3 pr-4 text-slate-300">{fmtDuration(j.total_duration_s)}</td>
                  <td className="py-3 pr-4 text-slate-300">{fmtUsd(j.total_cost_usd)}</td>
                  <td className="py-3 text-slate-400">{new Date(j.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
