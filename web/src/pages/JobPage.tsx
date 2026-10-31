import { Activity, AlertTriangle, CheckCircle2, Clapperboard, Loader2, PencilLine, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, fmtDuration, type JobSummary, type PipelineEvent } from "../api";
import ReviewEditor from "../components/ReviewEditor";
import { Downloads, ReportCard, VideoCompare } from "../components/Results";
import StageStepper from "../components/StageStepper";
import { Card, ErrorNote, ProgressBar, StateBadge } from "../components/ui";

const REFRESH_ON = new Set(["done", "skipped", "failed", "retrying", "awaiting_review", "queued"]);

export default function JobPage({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<{ stage: string | null; message: string; progress: number }>(
    { stage: null, message: "", progress: 0 });
  const [streamKey, setStreamKey] = useState(0);
  const [editing, setEditing] = useState(false);

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
      if (REFRESH_ON.has(e.status) || e.status === "running") void refresh();
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
      : <div className="grid place-items-center py-24 text-slate-500"><Loader2 className="size-6 animate-spin" /></div>;
  }

  const running = job.state === "running" || job.state === "queued";
  const progress = running ? live.progress : job.state === "done" ? 1 : job.progress;
  const hasTranslation = job.artifacts.includes("translation.json");
  const showEditor = job.state === "awaiting_review" || (editing && hasTranslation && !running);
  const m = job.metadata;

  return (
    <div className="space-y-6">
      <div className="card flex flex-wrap items-center justify-between gap-4 p-5">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate text-xl font-semibold text-white">{job.input_name}</h1>
            <StateBadge state={job.state} />
          </div>
          <p className="mt-1 text-sm text-slate-400">
            {m ? `${m.width}×${m.height} · ${m.fps} fps · ${m.video_codec}/${m.audio_codec} · ${fmtDuration(m.duration_s)}`
              : "Analysing video…"}
            {" · "}Chinese → {String(job.options.target_language ?? "English")}
            {" · "}{String(job.options.tts_provider ?? "gemini")} voices
          </p>
        </div>
        <div className="flex gap-2">
          {job.state === "failed" && (
            <button className="btn-primary" onClick={() => restart("retry")}>
              <RotateCcw className="size-4" /> Retry from failed stage
            </button>
          )}
          {job.state === "done" && hasTranslation && (
            <button className="btn-ghost" onClick={() => setEditing((v) => !v)}>
              <PencilLine className="size-4" /> {editing ? "Close editor" : "Edit & re-dub"}
            </button>
          )}
        </div>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}

      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <Card title="Pipeline" icon={<Activity className="size-4 text-indigo-300" />} className="h-fit">
          <div className="mb-5 space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{running ? live.message || "working…" : job.state.replace("_", " ")}</span>
              <span>{Math.round(progress * 100)}%</span>
            </div>
            <ProgressBar value={progress}
              tone={job.state === "failed" ? "rose" : job.state === "done" ? "emerald" : "indigo"} />
          </div>
          <StageStepper job={job} liveStage={running ? live.stage : null} liveMessage={live.message} />
        </Card>

        <div className="min-w-0 space-y-6">
          {job.state === "failed" && job.error && (
            <div className="card flex gap-3 border-rose-500/20 p-4">
              <AlertTriangle className="mt-0.5 size-5 shrink-0 text-rose-300" />
              <div className="text-sm">
                <p className="font-medium text-rose-200">The pipeline stopped</p>
                <p className="mt-1 break-words text-rose-200/80">{job.error}</p>
                <p className="mt-2 text-slate-400">Completed stages are checkpointed; retry resumes from the failed stage.</p>
              </div>
            </div>
          )}

          {job.state === "awaiting_review" && (
            <div className="card flex gap-3 border-amber-500/20 p-4">
              <PencilLine className="mt-0.5 size-5 shrink-0 text-amber-300" />
              <div className="text-sm">
                <p className="font-medium text-amber-200">Ready for review</p>
                <p className="mt-1 text-slate-400">
                  Polish the English lines and pick a voice for each speaker, then generate the dub.
                  The fit bar estimates whether a line fits its time slot.
                </p>
              </div>
            </div>
          )}

          {showEditor && (
            <ReviewEditor jobId={jobId} canGenerate onGenerated={() => {
              setEditing(false); setStreamKey((k) => k + 1); void refresh();
            }} />
          )}

          {job.artifacts.includes("output.mp4") && !showEditor && (
            <>
              <div className="card flex items-center gap-3 border-emerald-500/20 p-4 text-sm">
                <CheckCircle2 className="size-5 text-emerald-300" />
                <span className="text-emerald-100">Dub complete in {fmtDuration(job.total_duration_s)}.</span>
              </div>
              <VideoCompare job={job} />
            </>
          )}

          {running && !job.artifacts.includes("output.mp4") && (
            <Card>
              <div className="grid place-items-center gap-3 py-12 text-center">
                <span className="grid size-14 place-items-center rounded-2xl bg-indigo-500/10 ring-1 ring-indigo-400/30">
                  <Clapperboard className="size-7 animate-pulse text-indigo-300" />
                </span>
                <p className="font-medium text-slate-100">Dubbing in progress</p>
                <p className="max-w-md text-sm text-slate-400">
                  Follow each stage on the left. Leave this page and come back; the job keeps running on the server.
                </p>
              </div>
            </Card>
          )}

          <div className="grid gap-6 xl:grid-cols-2">
            <Downloads job={job} />
            <ReportCard job={job} />
          </div>
        </div>
      </div>
    </div>
  );
}
