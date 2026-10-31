/** Logo mark: a Chinese-character stroke resolving into an English waveform. */
export function LogoMark({ className = "size-9" }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 40" className={className} aria-hidden="true">
      <rect x="0.5" y="0.5" width="39" height="39" rx="11" fill="var(--panel-2)" stroke="var(--line-strong)" />
      {/* 中 */}
      <path d="M8 14h10v10H8zM13 9v20" fill="none" stroke="var(--zh)" strokeWidth="2.2"
        strokeLinecap="round" strokeLinejoin="round" />
      {/* → waveform */}
      <path d="M20 19h2l1.6-5 2.6 11 2.4-14 2.4 16 2-8h3" fill="none" stroke="var(--en)"
        strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const BARS = 48;
const [HERO_ZH, HERO_EN] = ["解铃还须系铃人。", "I made this mess, I'll fix it."];

/** Hero motif: a live waveform with a subtitle that dissolves from 中文 into English. */
export function HeroWave() {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-line bg-panel-2 px-5 pb-5 pt-4" aria-hidden="true">
      <div className="mb-3 flex items-center justify-between">
        <span className="eyebrow flex items-center gap-2">
          <span className="size-1.5 rounded-full bg-bad tally" /> REC · CH 1–2
        </span>
        <span className="tc text-[0.7rem] text-faint">00:00:12.40</span>
      </div>
      <div className="flex h-20 items-center gap-[3px]">
        {Array.from({ length: BARS }, (_, i) => {
          const env = Math.sin((i / BARS) * Math.PI);
          const h = 18 + env * 70 * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7)));
          const zh = i < BARS / 2;
          return (
            <span key={i} className="wave-bar w-full rounded-full"
              style={{ height: `${h}%`, background: zh ? "var(--zh)" : "var(--en)",
                       opacity: 0.35 + env * 0.65, animationDelay: `${(i % 12) * -0.1}s` }} />
          );
        })}
      </div>
      <div className="relative mt-4 h-14">
        <div className="absolute inset-0 grid place-items-center text-center">
          <span className="swap-zh absolute font-zh text-xl text-zh">{HERO_ZH}</span>
          <span className="swap-en absolute font-display text-xl font-bold text-en">{HERO_EN}</span>
        </div>
      </div>
      <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-faint">
        <span>原声 ZH</span><span>→</span><span>DUB EN</span>
      </div>
    </div>
  );
}
