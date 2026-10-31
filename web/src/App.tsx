import { ListVideo, Moon, Plus, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { LogoMark } from "./components/Brand";
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

type Theme = "light" | "dark";

function useTheme(): [Theme, () => void] {
  const system = () => (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const [theme, setTheme] = useState<Theme>(
    () => (document.documentElement.dataset.theme as Theme | undefined) ?? system());
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("dub-theme", next); } catch { /* storage unavailable */ }
    setTheme(next);
  };
  return [theme, toggle];
}

export default function App() {
  const route = useRoute();
  const [theme, toggleTheme] = useTheme();
  const jobMatch = route.match(/^\/jobs\/([\w-]+)$/);

  const navLink = (href: string, active: boolean, children: ReactNode) => (
    <a href={href} className={`btn px-3 ${active ? "bg-panel-3 text-text" : "text-muted hover:text-text"}`}>
      {children}
    </a>
  );

  return (
    <div className="mx-auto flex min-h-screen max-w-[88rem] flex-col px-4 sm:px-6">
      <header className="sticky top-0 z-30 -mx-4 mb-6 border-b border-line bg-bg/80 px-4 backdrop-blur-md sm:-mx-6 sm:px-6"
        style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}>
        <div className="flex h-16 items-center justify-between gap-3">
          <a href="#/" className="flex items-center gap-3">
            <LogoMark />
            <span className="leading-tight">
              <span className="block font-display text-lg font-extrabold tracking-tight">
                DUB<span className="text-signal">·</span>STUDIO
              </span>
              <span className="hidden font-mono text-[0.65rem] tracking-wider text-muted sm:block">
                中文 → ENGLISH · AI LOCALIZATION SUITE
              </span>
            </span>
          </a>
          <nav className="flex items-center gap-1">
            {navLink("#/", route === "/", <><Plus className="size-4" /> <span className="hidden sm:inline">New session</span></>)}
            {navLink("#/jobs", route.startsWith("/jobs"), <><ListVideo className="size-4" /> <span className="hidden sm:inline">Sessions</span></>)}
            <button className="btn ml-1 border border-line px-2.5 text-muted hover:text-text" onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </nav>
        </div>
      </header>

      <main className="flex-1 pb-16">
        {jobMatch ? <JobPage key={jobMatch[1]} jobId={jobMatch[1]} />
          : route === "/jobs" ? <JobsPage />
          : <UploadPage />}
      </main>

      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-line py-5 font-mono text-[0.68rem] tracking-wider text-faint">
        <span>DEMUCS · FASTER-WHISPER · PYANNOTE · GEMINI · RUBBERBAND · FFMPEG</span>
        <span>DUB STUDIO v0.1</span>
      </footer>
    </div>
  );
}
