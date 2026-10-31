import { AlertTriangle, CheckCircle2, Loader2, PencilLine, RotateCcw, Radio, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, fmtDuration, fmtTC, type DubTrack, type JobSummary, type PipelineEvent,
  type Transcript, type Translation } from "../api";
import Monitor from "../components/Monitor";
import QACard from "../components/QACard";
import { ReportCard, Tapes } from "../components/Results";
import ScriptEditor from "../components/ScriptEditor";
import SignalChain, { VuMeter } from "../components/SignalChain";
import { Chip, ErrorNote, Panel, StatusPill } from "../components/ui";

const REFRESH_ON = new Set(["done", "skipped", "failed", "retrying", "awaiting_review", "queued", "running"]);

interface LogLine { at: number; stage: string | null; status: string; message: string }

/** Transcript / translation / dub track, refetched whenever new artifacts appear. */
function useStudioData(job: JobSummary | null) {
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [translation, setTranslation] = useState<Translation | null>(null);
  const [dub, setDub] = useState<DubTrack | null>(null);
  const has = (n: string) => job?.artifacts.includes(n) ?? false;
  const key = job ? `${job.job_id}|${job.state}|${job.artifacts.join(",")}|${job.total_duration_s}` : "";

  useEffect(() => {
    if (!job) return;
    const id = job.job_id;
    if (has("transcript.json")) api.transcript(id).then(setTranscript).catch(() => setTranscript(null));
    if (has("translation.json")) api.translation(id).then(setTranslation).catch(() => setTranslation(null));
    if (has("dub_track.json")) api.dubTrack(id).then(setDub).catch(() => setDub(null));
    else setDub(null);
  }, [key]); // `key` captures every input that matters

  return { transcript, translation, dub };
}

export default function JobPage({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<{ stage: string | null; message: string; progress: number }>(
    { stage: null, message: "", progress: 0 });
  const [log, setLog] = useState<LogLine[]>([]);
  const [streamKey, setStreamKey] = useState(0);
  const [editing, setEditing] = useState(false);
  const { transcript, translation, dub } = useStudioData(job);

  const refresh = useCallback(
    () => api.job(jobId).then(setJob).catch((e) => setError(e.message)), [jobId]);

  // Live progress over Server-Sent Events; reconnects whenever a run is (re)started.
  useEffect(() => {
    const es = api.events(jobId);
    es.addEventListener("snapshot", (msg) => {
      const snap = JSON.parse((msg as MessageEvent).data) as JobSummary;
      setJob(snap);
      setLive({ stage: snap.stage, message: snap.message, progress: snap.progress });
    });
    es.onmessage = (msg) => {
      const e = JSON.parse(msg.data) as PipelineEvent;
      setLive((prev) => ({ stage: e.stage ?? prev.stage, message: e.message, progress: e.progress }));
      if (e.status !== "progress") {
        setLog((l) => [{ at: Date.now(), stage: e.stage, status: e.status, message: e.message }, ...l].slice(0, 40));
      }
      if (REFRESH_ON.has(e.status)) void refresh();
    };
    es.onerror = () => { /* EventSource retries automatically */ };
    return () => es.close();
  }, [jobId, refresh, streamKey]);

  const restart = async (action: "retry" | "continue") => {
    setError(null);
    try {
      setJob(await (action === "retry" ? api.retryJob(jobId) : api.continueJob(jobId)));
      setEditing(false);
      setStreamKey((k) => k + 1);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  if (!job) {
    return error ? <ErrorNote>{error}</ErrorNote>
      : <div className="grid place-items-center py-32 text-muted"><Loader2 className="size-6 animate-spin" /></div>;
  }

  const running = job.state === "running" || job.state === "queued";
  const progress = running ? live.progress : job.state === "done" ? 1 : job.progress;
  const hasTranslation = job.artifacts.includes("translation.json");
  const hasOutput = job.artifacts.includes("output.mp4");
  const showEditor = job.state === "awaiting_review" || (editing && hasTranslation && !running);
  const m = job.metadata;
  const opts = job.options as Record<string, string | boolean | undefined>;

  return (
    <div className="space-y-6">
      {/* Slate header */}
      <section className="panel overflow-hidden">
        <div className="flex flex-wrap items-start justify-between gap-4 p-5">
          <div className="min-w-0">
            <div className="eyebrow mb-1.5 flex flex-wrap items-center gap-x-3">
              <span>Session</span><span className="text-faint">{job.job_id}</span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="truncate font-display text-2xl font-extrabold tracking-tight sm:text-3xl">
                {job.input_name.replace(/\.mp4$/i, "")}
              </h1>
              <StatusPill state={job.state} />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {m ? (
                <>
                  <Chip>{fmtTC(m.duration_s)}</Chip>
                  <Chip>{m.width}×{m.height}</Chip>
                  <Chip>{m.fps} fps</Chip>
                  <Chip>{m.video_codec}/{m.audio_codec}</Chip>
                </>
              ) : <Chip>probing…</Chip>}
              <Chip tone="signal">中文 → {String(opts.target_language ?? "English")}</Chip>
              <Chip>{String(opts.tts_provider ?? "edge")} voices</Chip>
              {opts.lipsync ? <Chip>lip sync</Chip> : null}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {job.state === "failed" && (
              <button className="btn-signal" onClick={() => restart("retry")}>
                <RotateCcw className="size-4" /> Resume from failure
              </button>
            )}
            {job.state === "done" && hasTranslation && (
              <button className="btn-ghost" onClick={() => setEditing((v) => !v)}>
                {editing ? <X className="size-4" /> : <PencilLine className="size-4" />}
                {editing ? "Close script" : "Edit script & re-dub"}
              </button>
            )}
          </div>
        </div>

        {/* Signal chain + meter */}
        <div className="border-t border-line bg-panel-2/60 p-5">
          <SignalChain job={job} liveStage={running ? live.stage : null} liveMessage={live.message} />
          <div className="mt-4 flex items-center gap-4">
            <div className="flex-1"><VuMeter value={progress} state={job.state} /></div>
            <span className="tc w-12 text-right text-sm font-medium">{Math.round(progress * 100)}%</span>
          </div>
        </div>
      </section>

      {error && <ErrorNote>{error}</ErrorNote>}

      {job.state === "failed" && job.error && (
        <div className="panel flex gap-3 border-bad/40 p-4">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-bad" />
          <div className="min-w-0 text-sm">
            <p className="font-semibold text-bad">Signal lost</p>
            <p className="mt-1 break-words font-mono text-xs text-muted">{job.error}</p>
            <p className="mt-2 text-muted">Completed stages are checkpointed — resuming skips them.</p>
          </div>
        </div>
      )}

      {job.state === "awaiting_review" && (
        <div className="panel flex gap-3 border-warn/40 p-4">
          <PencilLine className="mt-0.5 size-5 shrink-0 text-warn" />
          <div className="text-sm">
            <p className="font-semibold">Script ready for review</p>
            <p className="mt-1 text-muted">
              Polish the English lines and recast voices, then voice it. The fit meter estimates whether
              each line fits its time slot.
            </p>
          </div>
        </div>
      )}

      {showEditor && (
        <div className="space-y-6">
          {translation?.review && <QACard translation={translation} />}
          <ScriptEditor jobId={jobId} onGenerated={() => {
            setEditing(false); setStreamKey((k) => k + 1); void refresh();
          }} />
        </div>
      )}

      {!showEditor && hasOutput && (
        <>
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle2 className="size-4 text-ok" />
            <span>Mastered in <span className="tc">{fmtDuration(job.total_duration_s)}</span></span>
          </div>
          <Monitor job={job} transcript={transcript} translation={translation} dub={dub} />
        </>
      )}

      {running && !hasOutput && <OnAir log={log} live={live} />}

      {!showEditor && (
        <div className="grid gap-6 xl:grid-cols-2">
          {translation?.review && hasOutput ? <QACard translation={translation} /> : null}
          <ReportCard job={job} />
          <div className="xl:col-span-2"><Tapes job={job} /></div>
        </div>
      )}
    </div>
  );
}

function OnAir({ log, live }: { log: LogLine[]; live: { stage: string | null; message: string } }) {
  return (
    <Panel eyebrow="On air" title="Processing" icon={<Radio className="size-4" />}>
      <div className="grid gap-6 md:grid-cols-[1fr_1.2fr]">
        <div className="grid place-items-center gap-4 py-6 text-center">
          <div className="flex h-16 items-center gap-1" aria-hidden="true">
            {Array.from({ length: 18 }, (_, i) => (
              <span key={i} className="wave-bar w-1.5 rounded-full bg-signal"
                style={{ height: `${30 + ((i * 37) % 70)}%`, animationDelay: `${i * -0.08}s` }} />
            ))}
          </div>
          <p className="font-display text-lg font-bold">{live.stage ?? "warming up"}</p>
          <p className="max-w-sm text-sm text-muted">{live.message || "Waiting for the worker…"}</p>
          <p className="text-xs text-faint">You can leave this page; the session keeps running on the server.</p>
        </div>
        <div className="scroll-thin max-h-72 overflow-y-auto rounded-xl border border-line bg-bg/60 p-3 font-mono text-xs">
          {log.length === 0 && <p className="text-faint">listening for events…</p>}
          {log.map((l, i) => (
            <p key={`${l.at}-${i}`} className="rise-in flex gap-2 py-0.5">
              <span className="text-faint">{new Date(l.at).toLocaleTimeString()}</span>
              <span style={{ color: l.status === "failed" ? "var(--bad)" : l.status === "done" ? "var(--ok)"
                : l.status === "retrying" ? "var(--warn)" : "var(--signal)" }}>{l.status}</span>
              <span className="text-muted">{l.stage}</span>
              <span className="truncate">{l.message}</span>
            </p>
          ))}
        </div>
      </div>
    </Panel>
  );
}
