import { Download, Film, Gauge, Pause, Play, Receipt } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, fmtDuration, fmtUsd, type JobSummary, type Report } from "../api";
import { Card } from "./ui";

const DOWNLOADS: { name: string; label: string }[] = [
  { name: "output.mp4", label: "Dubbed video (MP4)" },
  { name: "output_lipsync.mp4", label: "Lip-synced video (MP4)" },
  { name: "subtitles_en.srt", label: "English subtitles (SRT)" },
  { name: "subtitles_zh.srt", label: "Chinese subtitles (SRT)" },
  { name: "dub_mix.wav", label: "Final audio mix (WAV)" },
  { name: "dub_voice.wav", label: "English dialogue only (WAV)" },
  { name: "background.wav", label: "Music & effects stem (WAV)" },
  { name: "transcript.json", label: "Chinese transcript (JSON)" },
  { name: "translation.json", label: "Translation (JSON)" },
  { name: "voices.json", label: "Speaker mapping (JSON)" },
];

export function VideoCompare({ job }: { job: JobSummary }) {
  const left = useRef<HTMLVideoElement>(null);
  const right = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const dubbed = job.artifacts.includes("output_lipsync.mp4") ? "output_lipsync.mp4" : "output.mp4";

  const toggle = () => {
    const [a, b] = [left.current, right.current];
    if (!a || !b) return;
    if (playing) {
      a.pause(); b.pause();
    } else {
      b.currentTime = a.currentTime;
      a.muted = true;
      void a.play(); void b.play();
    }
    setPlaying(!playing);
  };

  return (
    <Card title="Original vs. dubbed" icon={<Film className="size-4 text-indigo-300" />}
      actions={
        <button className="btn-ghost" onClick={toggle}>
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
          {playing ? "Pause both" : "Play side by side"}
        </button>
      }>
      <div className="grid gap-4 md:grid-cols-2">
        {[
          { ref: left, name: "input.mp4", label: "Original · 中文" },
          { ref: right, name: dubbed, label: dubbed === "output.mp4" ? "Dubbed" : "Dubbed + lip sync" },
        ].map((v) => (
          <figure key={v.name} className="space-y-2">
            <video ref={v.ref} src={api.fileUrl(job.job_id, v.name)} controls preload="metadata"
              onEnded={() => setPlaying(false)} className="aspect-video w-full rounded-xl bg-black" />
            <figcaption className="text-xs text-slate-400">{v.label}</figcaption>
          </figure>
        ))}
      </div>
    </Card>
  );
}

export function Downloads({ job }: { job: JobSummary }) {
  const available = DOWNLOADS.filter((d) => job.artifacts.includes(d.name));
  return (
    <Card title="Downloads" icon={<Download className="size-4 text-emerald-300" />}>
      <div className="grid gap-2 sm:grid-cols-2">
        {available.map((d) => (
          <a key={d.name} href={api.fileUrl(job.job_id, d.name, true)}
            className="flex items-center justify-between rounded-lg border border-white/5 bg-ink-800/60 px-3 py-2 text-sm transition hover:border-indigo-400/40 hover:bg-indigo-500/5">
            <span className="text-slate-200">{d.label}</span>
            <Download className="size-4 text-slate-500" />
          </a>
        ))}
        {available.length === 0 && <p className="text-sm text-slate-500">No artifacts yet.</p>}
      </div>
    </Card>
  );
}

export function ReportCard({ job }: { job: JobSummary }) {
  const [report, setReport] = useState<Report | null>(null);
  useEffect(() => { api.report(job.job_id).then(setReport).catch(() => setReport(null)); },
    [job.job_id, job.state, job.total_duration_s]);
  if (!report) return null;

  const stats = [
    { label: "Processing time", value: fmtDuration(report.total_duration_s) },
    { label: "Realtime factor", value: report.realtime_factor != null ? `${report.realtime_factor.toFixed(2)}×` : "—" },
    { label: "API cost", value: fmtUsd(report.total_cost_usd) },
    { label: "Cost / minute", value: fmtUsd(report.cost_per_minute_usd) },
  ];
  const maxDuration = Math.max(...report.stages.map((s) => s.duration_s ?? 0), 1);

  return (
    <Card title="Cost & processing report" icon={<Receipt className="size-4 text-amber-300" />}>
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="rounded-xl border border-white/5 bg-ink-800/60 p-3">
            <div className="text-xs text-slate-400">{s.label}</div>
            <div className="mt-1 text-xl font-semibold text-white">{s.value}</div>
          </div>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th className="pb-2 font-medium">Stage</th>
              <th className="pb-2 font-medium">Time</th>
              <th className="w-2/5 pb-2 font-medium"><Gauge className="inline size-3.5" /></th>
              <th className="pb-2 font-medium">Attempts</th>
              <th className="pb-2 text-right font-medium">Cost</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {report.stages.map((s) => (
              <tr key={s.stage}>
                <td className="py-2 pr-3 text-slate-200">{s.stage}</td>
                <td className="py-2 pr-3 text-slate-400">{fmtDuration(s.duration_s)}</td>
                <td className="py-2 pr-3">
                  <div className="h-1.5 rounded-full bg-white/5">
                    <div className="h-full rounded-full bg-linear-to-r from-indigo-500 to-fuchsia-500"
                      style={{ width: `${((s.duration_s ?? 0) / maxDuration) * 100}%` }} />
                  </div>
                </td>
                <td className="py-2 pr-3 text-slate-400">{s.attempts}</td>
                <td className="py-2 text-right text-slate-300">{fmtUsd(s.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {report.costs.length > 0 && (
        <details className="mt-4 text-xs text-slate-400">
          <summary className="cursor-pointer text-slate-300">API usage breakdown</summary>
          <ul className="mt-2 space-y-1">
            {report.costs.map((c, i) => (
              <li key={i} className="flex justify-between gap-4">
                <span>{c.provider} · {c.item} · {c.quantity.toLocaleString()} {c.unit}</span>
                <span>{fmtUsd(c.usd)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Card>
  );
}
