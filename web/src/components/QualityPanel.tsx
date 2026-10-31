import { AlertTriangle, CircleAlert, Gauge, Info } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type JobSummary, type Quality, type QualityWarning } from "../api";
import { Panel } from "./ui";

const LEVEL: Record<QualityWarning["level"], { color: string; Icon: typeof Info }> = {
  info: { color: "var(--en)", Icon: Info },
  warn: { color: "var(--warn)", Icon: AlertTriangle },
  bad: { color: "var(--bad)", Icon: CircleAlert },
};

function Meter({ label, value, sub, tone }: {
  label: string; value: string | number; sub: string; tone?: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel-2 px-3 py-2.5">
      <div className="eyebrow text-[0.6rem]">{label}</div>
      <div className="mt-1 font-display text-xl font-bold tracking-tight tc"
        style={tone ? { color: tone } : undefined}>{value}</div>
      <div className="text-[0.7rem] text-muted">{sub}</div>
    </div>
  );
}

/** Self-reported limitations: ASR confidence + recovery, diarization, sync fit. */
export default function QualityPanel({ job }: { job: JobSummary }) {
  const [q, setQ] = useState<Quality | null>(null);
  const key = `${job.job_id}|${job.state}|${job.artifacts.join(",")}`;
  useEffect(() => { api.quality(job.job_id).then(setQ).catch(() => setQ(null)); },
    [key]); // refetch whenever new artifacts land

  if (!q || (!q.asr && !q.sync && q.warnings.length === 0)) return null;
  const { asr, sync, speakers } = q;
  const lowN = asr?.low_confidence.length ?? 0;
  const overflowN = sync?.overflow.length ?? 0;

  return (
    <Panel eyebrow="Self-check" title="Quality report" icon={<Gauge className="size-4" />}>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        {asr && (
          <Meter label="ASR confidence"
            value={asr.mean_confidence !== null ? asr.mean_confidence.toFixed(2) : "—"}
            sub={lowN ? `${lowN} low-confidence line${lowN > 1 ? "s" : ""}` : "all lines confident"}
            tone={lowN ? "var(--warn)" : "var(--ok)"} />
        )}
        {asr?.vad_speech_regions !== undefined && (
          <Meter label="Speech coverage" value={`${asr.vad_speech_regions}`}
            sub={asr.recovered_regions ? `regions · ${asr.recovered_regions} recovered`
                                       : "regions · none missed"}
            tone={asr.recovered_regions ? "var(--en)" : undefined} />
        )}
        {speakers && (
          <Meter label="Speakers" value={speakers.count}
            sub={`${speakers.genders.male}♂ · ${speakers.genders.female}♀`
                 + (speakers.genders.unknown ? ` · ${speakers.genders.unknown}?` : "")} />
        )}
        {sync && (
          <Meter label="Sync fit" value={overflowN ? `${overflowN} over` : "100%"}
            sub={`${sync.stretched} stretched (max ×${sync.max_tempo.toFixed(2)}) · ${sync.rewritten} rewritten`}
            tone={overflowN ? "var(--bad)" : "var(--ok)"} />
        )}
      </div>

      {q.warnings.length > 0 ? (
        <ul className="mt-4 space-y-2">
          {q.warnings.map((w) => {
            const { color, Icon } = LEVEL[w.level];
            return (
              <li key={w.message} className="flex items-start gap-2.5 rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm"
                style={{ boxShadow: `inset 3px 0 0 ${color}` }}>
                <Icon className="mt-0.5 size-4 shrink-0" style={{ color }} />
                <span>{w.message}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="mt-4 text-sm text-muted">No limitations detected for this video.</p>
      )}

      {lowN > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-muted">Low-confidence source lines</summary>
          <ul className="mt-2 space-y-1">
            {asr!.low_confidence.map((s) => (
              <li key={s.id} className="flex gap-3">
                <span className="tc w-10 text-warn">{s.confidence.toFixed(2)}</span>
                <span className="low-conf font-zh text-zh">{s.text}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Panel>
  );
}
