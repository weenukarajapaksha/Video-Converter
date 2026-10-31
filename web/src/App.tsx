import { Clapperboard, ListVideo, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import JobPage from "./pages/JobPage";
import JobsPage from "./pages/JobsPage";
import UploadPage from "./pages/UploadPage";

// Minimal hash router: #/ (upload), #/jobs, #/jobs/<id>
function useRoute() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const onChange = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash.replace(/^#/, "");
}

export const navigate = (path: string) => {
  window.location.hash = path;
};

export default function App() {
  const route = useRoute();
  const jobMatch = route.match(/^\/jobs\/([\w-]+)$/);

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 sm:px-6">
      <header className="flex items-center justify-between py-5">
        <a href="#/" className="flex items-center gap-3">
          <span className="grid size-9 place-items-center rounded-xl bg-linear-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-indigo-500/30">
            <Clapperboard className="size-5 text-white" />
          </span>
          <span>
            <span className="block text-base font-semibold tracking-tight text-white">Dubber</span>
            <span className="block text-xs text-slate-400">AI video translation &amp; dubbing · 中文 → English</span>
          </span>
        </a>
        <nav className="flex gap-2">
          <a href="#/" className={`btn-ghost ${route === "/" ? "ring-1 ring-indigo-400/40" : ""}`}>
            <Upload className="size-4" /> New dub
          </a>
          <a href="#/jobs" className={`btn-ghost ${route.startsWith("/jobs") ? "ring-1 ring-indigo-400/40" : ""}`}>
            <ListVideo className="size-4" /> Jobs
          </a>
        </nav>
      </header>

      <main className="flex-1 pb-16">
        {jobMatch ? <JobPage key={jobMatch[1]} jobId={jobMatch[1]} />
          : route === "/jobs" ? <JobsPage />
          : <UploadPage />}
      </main>

      <footer className="border-t border-white/5 py-6 text-center text-xs text-slate-500">
        Demucs · faster-whisper · pyannote · Gemini · FFmpeg
      </footer>
    </div>
  );
}
