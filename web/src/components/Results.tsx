import { Download, FileAudio, FileJson, FileText, FileVideo, Receipt } from "lucide-react";
import { useEffect, useState } from "react";
import { STAGES, api, fmtBytes, fmtDuration, fmtUsd, type JobSummary, type Report } from "../api";
import { Panel, Stat } from "./ui";

const TAPES: { name: string; label: string; kind: "video" | "audio" | "text" | "json" }[] = [
  { name: "output.mp4", label: "Dubbed video", kind: "video" },
  { name: "output_lipsync.mp4", label: "Lip-synced video", kind: "video" },
  { name: "subtitles_en.srt", label: "English subtitles", kind: "text" },
  { name: "subtitles_zh.srt", label: "Chinese subtitles", kind: "text" },
  { name: "dub_mix.wav", label: "Final mix", kind: "audio" },
  { name: "dub_voice.wav", label: "English dialogue stem", kind: "audio" },
  { name: "background.wav", label: "Music & FX stem", kind: "audio" },
  { name: "vocals.wav", label: "Original vocal stem", kind: "audio" },
  { name: "transcript.json", label: "Chinese transcript", kind: "json" },
  { name: "translation.json", label: "Translation + QA", kind: "json" },
  { name: "voices.json", label: "Speaker → voice map", kind: "json" },
  { name: "dub_track.json", label: "Sync report", kind: "json" },
];

const KIND = {
  video: { icon: FileVideo, color: "var(--signal)" },
  audio: { icon: FileAudio, color: "var(--en)" },
  text: { icon: FileText, color: "var(--zh)" },
  json: { icon: FileJson, color: "var(--muted)" },
};

export function Tapes({ job }: { job: JobSummary }) {
  const available = TAPES.filter((d) => job.artifacts.includes(d.name));
  const sizes: Record<string, number | null> = job.artifact_sizes ?? {};

  return (
    <Panel eyebrow="Deliverables" title="Tape library" icon={<Download className="size-4" />}>
      {available.length === 0 ? <p className="text-sm text-muted">Nothing recorded yet.</p> : (
        <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
          {available.map((d) => {
            const { icon: Icon, color } = KIND[d.kind];
            const ext = d.name.split(".").pop()!.toUpperCase();
            return (
              <a key={d.name} href={api.fileUrl(job.job_id, d.name, true)}
                className="group relative flex items-center gap-3 overflow-hidden rounded-xl border border-line bg-panel-2 py-2.5 pl-4 pr-3 transition hover:-translate-y-0.5 hover:border-line-strong">
                <span className="absolute inset-y-0 left-0 w-1" style={{ background: color }} />
                <Icon className="size-5 shrink-0" style={{ color }} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">{d.label}</span>
                  <span className="tc block text-[0.68rem] text-faint">{ext} · {fmtBytes(sizes[d.name]) || "…"}</span>
                </span>
                <Download className="size-4 text-faint transition group-hover:text-signal" />
              </a>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

export function ReportCard({ job }: { job: JobSummary }) {
  const [report, setReport] = useState<Report | null>(null);
  useEffect(() => { api.report(job.job_id).then(setReport).catch(() => setReport(null)); },
    [job.job_id, job.state, job.total_duration_s]);
  if (!report || report.stages.length === 0) return null;

  const labels = Object.fromEntries(STAGES.map((s) => [s.name, s.short]));
  const maxDuration = Math.max(...report.stages.map((s) => s.duration_s ?? 0), 1);
  const rtf = report.realtime_factor;

  return (
    <Panel eyebrow="Production report" title="Cost & time" icon={<Receipt className="size-4" />}>
      <div className="mb-5 grid grid-cols-2 gap-2.5">
        <Stat label="Processing" value={fmtDuration(report.total_duration_s)}
          sub={report.video_duration_s ? `for ${fmtDuration(report.video_duration_s)} of video` : undefined} />
        <Stat label="Realtime factor" value={rtf != null ? `${rtf.toFixed(2)}×` : "—"}
          sub={rtf != null ? (rtf <= 1 ? "faster than realtime" : "slower than realtime") : undefined}
          accent={rtf != null && rtf <= 1 ? "var(--ok)" : undefined} />
        <Stat label="API cost" value={fmtUsd(report.total_cost_usd)} accent="var(--signal)"
          sub="paid-tier list prices" />
        <Stat label="Per minute" value={fmtUsd(report.cost_per_minute_usd)} sub="of source video" />
      </div>

      <div className="eyebrow mb-2">Stage timing</div>
      <ul className="space-y-1.5">
        {report.stages.map((s) => {
          const w = ((s.duration_s ?? 0) / maxDuration) * 100;
          return (
            <li key={s.stage} className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-3 text-xs">
              <span className="truncate text-muted">{labels[s.stage] ?? s.stage}</span>
              <span className="relative h-4 overflow-hidden rounded bg-panel-3">
                <span className="absolute inset-y-0 left-0 rounded"
                  style={{ width: `${w}%`, background: s.status === "failed" ? "var(--bad)"
                    : "linear-gradient(90deg, color-mix(in srgb, var(--signal) 55%, transparent), var(--signal))" }} />
                {s.attempts > 1 && <span className="absolute right-1 top-0 font-mono text-[0.6rem] leading-4 text-warn">×{s.attempts}</span>}
              </span>
              <span className="tc text-right">
                {fmtDuration(s.duration_s)}
                {s.cost_usd > 0 && <span className="block text-[0.6rem] text-signal">{fmtUsd(s.cost_usd)}</span>}
              </span>
            </li>
          );
        })}
      </ul>

      {report.costs.length > 0 && (
        <details className="mt-4 rounded-lg border border-line bg-panel-2 px-3 py-2 text-xs">
          <summary className="cursor-pointer font-medium">API usage ledger</summary>
          <ul className="mt-2 space-y-1">
            {report.costs.map((c, i) => (
              <li key={i} className="tc flex justify-between gap-4 text-muted">
                <span className="truncate">{c.provider} · {c.item} · {c.quantity.toLocaleString()} {c.unit}</span>
                <span className="text-text">{fmtUsd(c.usd)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Panel>
  );
}
