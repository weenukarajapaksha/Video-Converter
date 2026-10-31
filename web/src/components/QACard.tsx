import { ArrowRight, ShieldCheck } from "lucide-react";
import type { Translation } from "../api";
import { Panel } from "./ui";

function scoreColor(v: number) {
  return v >= 8 ? "var(--ok)" : v >= 6 ? "var(--warn)" : "var(--bad)";
}

function Gauge({ label, value }: { label: string; value: number }) {
  const r = 30;
  const arc = Math.PI * 1.5 * r; // 270° sweep
  const frac = Math.max(0, Math.min(10, value)) / 10;
  const color = scoreColor(value);
  return (
    <figure className="flex flex-col items-center">
      <svg viewBox="0 0 80 80" className="size-24" role="img" aria-label={`${label} ${value} out of 10`}>
        <g transform="rotate(135 40 40)">
          <circle cx="40" cy="40" r={r} fill="none" stroke="var(--panel-3)" strokeWidth="7"
            strokeDasharray={`${arc} 999`} strokeLinecap="round" />
          <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
            strokeDasharray={`${arc * frac} 999`}
            style={{ filter: `drop-shadow(0 0 4px color-mix(in srgb, ${color} 60%, transparent))`,
                     transition: "stroke-dasharray 0.8s ease-out" }} />
        </g>
        <text x="40" y="44" textAnchor="middle" className="font-display" fontSize="20" fontWeight="800"
          fill="var(--text)">{value}</text>
        <text x="40" y="58" textAnchor="middle" fontSize="7" fill="var(--muted)" className="font-mono">/ 10</text>
      </svg>
      <figcaption className="eyebrow -mt-1">{label}</figcaption>
    </figure>
  );
}

/** Output of the LLM critic pass: scores plus every revision it made (before → after). */
export default function QACard({ translation }: { translation: Translation }) {
  const review = translation.review;
  if (!review) return null;
  const sources = new Map(translation.segments.map((s) => [s.id, s]));

  return (
    <Panel eyebrow={`Critic · ${translation.model}`} title="Translation QA" icon={<ShieldCheck className="size-4" />}>
      <div className="grid gap-5 md:grid-cols-[auto_1fr] md:items-center">
        <div className="flex justify-center gap-1 sm:gap-3">
          <Gauge label="Accuracy" value={review.accuracy} />
          <Gauge label="Natural" value={review.naturalness} />
          <Gauge label="Tone" value={review.tone} />
        </div>
        <blockquote className="border-l-2 border-signal pl-4 text-sm leading-relaxed text-muted">
          {review.summary}
        </blockquote>
      </div>

      {review.issues.length > 0 ? (
        <ul className="mt-5 space-y-3">
          {review.issues.map((iss) => (
            <li key={`${iss.id}-${iss.category}`} className="rounded-xl border border-line bg-panel-2 p-3.5">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="tc text-xs text-faint">#{String(iss.id).padStart(2, "0")}</span>
                <span className="rounded bg-signal/15 px-1.5 py-0.5 font-mono text-[0.65rem] uppercase text-signal">{iss.category}</span>
                <span className="text-xs text-muted">{iss.comment}</span>
              </div>
              {sources.get(iss.id) && (
                <p className="mb-2 font-zh text-sm text-zh">{sources.get(iss.id)!.source_text}</p>
              )}
              <div className="grid items-center gap-2 text-sm md:grid-cols-[1fr_auto_1fr]">
                <del className="rounded-md bg-bad/10 px-2.5 py-1.5 text-muted decoration-bad/70">{iss.before}</del>
                <ArrowRight className="mx-auto size-4 rotate-90 text-faint md:rotate-0" />
                <ins className="rounded-md bg-ok/10 px-2.5 py-1.5 font-medium no-underline">{iss.after}</ins>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-4 text-sm text-muted">The critic found nothing to change.</p>
      )}
    </Panel>
  );
}
