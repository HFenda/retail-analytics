"use client";
export const dynamic = "force-dynamic";

import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileVideo, CheckCircle2, X, Loader2, Download, Image as ImageIcon, AlertTriangle, ChartBar, Flame, Link as LinkIcon, TimerReset } from "lucide-react";

type HeatmapFiles = {
  overlay?: string;
  colored?: string;
  gray?: string;
  transparent_png?: string;
  preview?: string;
};

type Result = {
  job_id: string;
  counts_csv?: string;
  per_sec_csv?: string;
  by_min_csv?: string;
  peaks_csv?: string;
  heatmap?: HeatmapFiles;
  snapshot?: string;
  unique_total?: number;
  peak?: number;
};

type JobQueued = {
  job_id: string;
  status_url: string;   // npr. /api/v1/jobs/<id>
  results_url?: string; // npr. /files/results/<id>/
};

function cn(...cls: (string | false | null | undefined)[]) {
  return cls.filter(Boolean).join(" ");
}

const API =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (process.env.NODE_ENV !== "production" ? "http://localhost:8000" : "");

if (!API && process.env.NODE_ENV === "production") {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is not set");
}

const absUrl = (u?: string) => (!u ? "" : u.startsWith("/") ? `${API}${u}` : u);

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState<"idle" | "queued" | "processing" | "done" | "failed">("idle");
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [res, setRes] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [show, setShow] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "heatmap" | "files">("overview");

  // Upload handlers
  const onDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("video")) setFile(f);
  };
  const onChoose = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (f) setFile(f);
  };

  async function pollStatus(statusUrlAbs: string, maxSeconds = 480) {
    setPhase("queued");
    setStatusMsg("Queued…");
    const start = Date.now();
    while (Date.now() - start < maxSeconds * 1000) {
      try {
        const r = await fetch(statusUrlAbs, { cache: "no-store" });
        if (r.status === 404) throw new Error("Job je istekao ili ne postoji (404). Pošalji ponovo.");
        if (!r.ok) throw new Error(`Greška na statusu: ${r.status}`);
        const js = await r.json(); // očekujemo { status: "queued" | "processing" | "done" | "failed", error? }
        const s = (js.status || "").toLowerCase();

        if (s === "processing") {
          setPhase("processing");
          setStatusMsg("Processing…");
        } else if (s === "done") {
          setPhase("done");
          setStatusMsg("Done.");
          return true;
        } else if (s === "failed") {
          throw new Error(js?.error || "Job failed.");
        } else {
          setPhase("queued");
          setStatusMsg("Queued…");
        }
      } catch (e: any) {
        setError(e?.message || "Status error");
        setPhase("failed");
        return false;
      }
      await new Promise((res) => setTimeout(res, 2000));
    }
    setError("Polling timeout (predugo čeka).");
    setPhase("failed");
    return false;
  }

  async function fetchResultJson(resultsBase?: string) {
    if (!resultsBase) throw new Error("results_url nedostaje.");
    const url = absUrl(`${resultsBase.replace(/\/?$/, "/")}result.json`);
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`Ne mogu učitati result.json (${r.status}).`);
    const data: Result = await r.json();
    return data;
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) { setError("Select a video file."); return; }
    setError(null);
    setRes(null);
    setPhase("idle");
    setStatusMsg("");
    setLoading(true);
    setShow(false);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("vid_stride", "6");

      // 1) enqueue
      const r = await fetch(`${API}/api/v1/analyze`, { method: "POST", body: form });
      if (!r.ok) throw new Error(await r.text());
      const queued: JobQueued = await r.json();

      // 2) poll
      const ok = await pollStatus(absUrl(queued.status_url));
      if (!ok) throw new Error(error || "Status polling failed.");

      // 3) load result.json
      const data = await fetchResultJson(queued.results_url);
      // sanity fallbacks za preview
      if (data?.heatmap && !data.heatmap.preview) {
        data.heatmap.preview =
          data.heatmap.colored ||
          data.heatmap.gray ||
          data.heatmap.overlay ||
          data.heatmap.transparent_png ||
          "";
      }

      setRes({ ...data, job_id: queued.job_id });
      setShow(true);
      setActiveTab("overview");
    } catch (err: any) {
      setError(err?.message || "Server error.");
      setPhase("failed");
    } finally {
      setLoading(false);
    }
  }

  // Fake progress bar da UI ne bude prazan
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    if (!loading) { setProgress(0); return; }
    setProgress(8);
    const id = setInterval(() => setProgress(p => (p < 92 ? p + Math.random() * 7 : p)), 350);
    return () => clearInterval(id);
  }, [loading]);

  return (
    <main className="min-h-screen bg-gradient-to-b from-neutral-950 to-neutral-900 text-neutral-100">
      <div className="mx-auto w-full max-w-7xl px-4 py-10">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-2xl bg-white/10 backdrop-blur ring-1 ring-white/15 flex items-center justify-center">
              <ChartBar className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Retail Analytics</h1>
              <p className="text-sm text-neutral-400">Quick video analysis: people count, peak, heatmap</p>
            </div>
          </div>
          <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white">Demo</span>
        </header>

        {/* Grid */}
        <section className="mt-8 grid grid-cols-1 gap-8 sm:grid-cols-[1.2fr_1fr]">
          {/* Upload Card */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur">
            <h2 className="mb-4 flex items-center gap-2 text-white">
              <Upload className="h-5 w-5" /> Upload video
            </h2>
            <form onSubmit={onSubmit} className="space-y-4">
              <label
                onDrop={onDrop}
                onDragOver={(e) => e.preventDefault()}
                htmlFor="file"
                className={cn(
                  "group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed p-10 text-center",
                  "border-white/20 bg-gradient-to-b from-white/5 to-transparent hover:from-white/10"
                )}
              >
                <FileVideo className="mb-3 h-10 w-10 text-white/70 group-hover:scale-105 transition" />
                <div className="text-sm text-neutral-300">
                  Drag a video here or <span className="underline">choose</span>
                </div>
                <div className="mt-1 text-xs text-neutral-400">Supported formats: mp4, mov, webm…</div>
                <input id="file" type="file" accept="video/*" onChange={onChoose} className="absolute inset-0 opacity-0" />
              </label>

              {file && (
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 p-3 text-sm">
                  <div className="flex items-center gap-3 truncate">
                    <FileVideo className="h-5 w-5 text-white/60" />
                    <span className="truncate" title={file.name}>{file.name}</span>
                  </div>
                  <span className="rounded bg-white/10 px-2 py-0.5 text-xs">{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
                </div>
              )}

              <div className="flex items-center gap-3">
                <button type="submit" disabled={!file || loading} className={cn(
                  "inline-flex items-center rounded-xl bg-white/90 px-4 py-2 text-sm font-medium text-neutral-900",
                  "hover:bg-white disabled:opacity-50"
                )}>
                  <AnimatePresence>
                    {loading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Flame className="mr-2 h-4 w-4" />
                    )}
                  </AnimatePresence>
                  {loading ? "Analyzing…" : "Run analysis"}
                </button>

                {/* status badge */}
                {phase !== "idle" && (
                  <span className={cn(
                    "inline-flex items-center gap-1 rounded-xl border border-white/15 px-3 py-1 text-sm",
                    phase === "failed" ? "text-red-300" : "text-neutral-300"
                  )}>
                    <TimerReset className="h-4 w-4" />
                    {statusMsg}
                  </span>
                )}

                {error && (
                  <div className="flex items-center gap-2 text-red-300">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm">{error}</span>
                  </div>
                )}
              </div>

              {loading && (
                <div className="space-y-2">
                  <div className="h-2 w-full overflow-hidden rounded bg-white/10">
                    <div className="h-full bg-white" style={{ width: `${Math.min(100, Math.round(progress))}%` }} />
                  </div>
                  <p className="text-xs text-neutral-400">Processing frames, people, heatmap…</p>
                </div>
              )}
            </form>
          </div>

          {/* What you get */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur">
            <h2 className="mb-4 text-white">What you get</h2>
            <div className="grid gap-3 text-sm text-neutral-300">
              <FeatureItem title="Peak snapshot" icon={<ImageIcon className="h-4 w-4" />}>Automatic frame at the busiest moment.</FeatureItem>
              <FeatureItem title="Heatmap layers" icon={<ImageIcon className="h-4 w-4" />}>Overlay, transparent PNG, multiple preview styles.</FeatureItem>
              <FeatureItem title="CSV export" icon={<ImageIcon className="h-4 w-4" />}>Counts, occupancy per sec, by minute, peaks.</FeatureItem>
              <FeatureItem title="Fast demo API" icon={<LinkIcon className="h-4 w-4" />}>Works locally or via the configured API URL.</FeatureItem>
            </div>
          </div>
        </section>

        {/* Results Modal */}
        <AnimatePresence>
          {show && res && (
            <div className="fixed inset-0 z-50">
              <div className="absolute inset-0 bg-black/60" onClick={() => setShow(false)} />
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.18 }}
                className="absolute left-1/2 top-1/2 w-[95vw] max-w-4xl -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-white/10 bg-neutral-950 p-5 text-neutral-100 shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Analysis results</h3>
                    <p className="text-sm text-neutral-400">People counting, peak snapshot and heatmap.</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {res.job_id && <span className="rounded bg-white/10 px-2 py-1 text-xs">Job: {res.job_id}</span>}
                    <button onClick={() => setShow(false)} className="inline-flex items-center rounded-lg border border-white/15 px-3 py-1 text-sm hover:bg-white/10">
                      <X className="mr-1 h-4 w-4" /> Close
                    </button>
                  </div>
                </div>

                {/* KPIs */}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <KPI label="Unique total" value={res.unique_total ?? "-"} />
                  <KPI label="Peak" value={res.peak ?? "-"} />
                  <KPI label="CSV files" value={[res.counts_csv, res.per_sec_csv, res.by_min_csv, res.peaks_csv].filter(Boolean).length} />
                  <KPI label="Heatmap variants" value={Object.values(res.heatmap ?? {}).filter(Boolean).length} />
                </div>

                {/* Tabs */}
                <div className="mt-5">
                  <div className="inline-flex rounded-xl border border-white/15 bg-white/5 p-1 text-sm">
                    {(["overview","heatmap","files"] as const).map(t => (
                      <button key={t}
                        className={cn(
                          "rounded-lg px-3 py-1.5 capitalize",
                          activeTab === t ? "bg-white/15" : "text-neutral-300 hover:bg-white/10"
                        )}
                        onClick={() => setActiveTab(t)}
                      >{t === "overview" ? "Overview" : t === "heatmap" ? "Heatmap" : "Files"}</button>
                    ))}
                  </div>

                  {activeTab === "overview" && (
                    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                      <Panel title="Peak snapshot" href={absUrl(res.snapshot)}>
                        <ResponsiveImg src={absUrl(res.snapshot)} alt="snapshot" />
                      </Panel>
                      <Panel title="Heatmap preview" href={absUrl(res.heatmap?.preview)}>
                        <ResponsiveImg src={absUrl(res.heatmap?.preview)} alt="heatmap" />
                      </Panel>
                    </div>
                  )}

                  {activeTab === "heatmap" && (
                    <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
                      <Panel title="Overlay" href={absUrl(res.heatmap?.overlay)}>
                        <ResponsiveImg src={absUrl(res.heatmap?.overlay)} alt="overlay" />
                      </Panel>
                      <Panel title="Transparent PNG" href={absUrl(res.heatmap?.transparent_png)}>
                        <ResponsiveImg src={absUrl(res.heatmap?.transparent_png)} alt="transparent" />
                      </Panel>
                      <Panel title="Gray/Colored" href={absUrl(res.heatmap?.colored || res.heatmap?.gray)}>
                        <ResponsiveImg src={absUrl(res.heatmap?.colored || res.heatmap?.gray)} alt="variant" />
                      </Panel>
                    </div>
                  )}

                  {activeTab === "files" && (
                    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <FileLink label="counts.csv" href={absUrl(res.counts_csv)} />
                      <FileLink label="occupancy_per_sec.csv" href={absUrl(res.per_sec_csv)} />
                      <FileLink label="by_minute.csv" href={absUrl(res.by_min_csv)} />
                      <FileLink label="peaks.csv" href={absUrl(res.peaks_csv)} />
                      <FileLink label="heatmap_overlay.png" href={absUrl(res.heatmap?.overlay)} />
                      <FileLink label="heatmap_transparent.png" href={absUrl(res.heatmap?.transparent_png)} />
                    </div>
                  )}
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="mt-10 flex items-center justify-between text-xs text-neutral-500">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Built for a video-analysis demo. No private content is stored.
          </div>
          <span>API: {API}</span>
        </footer>
      </div>
    </main>
  );
}

function KPI({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="text-xs uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
    </div>
  );
}

function FeatureItem({ title, icon, children }: { title: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="mt-0.5 text-white/80">{icon}</div>
      <div>
        <div className="text-neutral-200">{title}</div>
        <div className="text-neutral-400">{children}</div>
      </div>
    </div>
  );
}

function ResponsiveImg({ src, alt }: { src?: string; alt: string }) {
  if (!src) return <div className="text-xs text-neutral-400">Not available</div>;
  return (
    <motion.img
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.25 }}
      src={src}
      alt={alt}
      className="w-full rounded-lg border border-white/10"
    />
  );
}

function Panel({ title, href, children }: { title: string; href?: string; children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-white/5">
      <div className="flex items-center justify-between p-3">
        <h3 className="text-sm font-medium">{title}</h3>
        {href && (
          <a href={href} target="_blank" rel="noreferrer" className="inline-flex items-center rounded-lg border border-white/15 px-3 py-1 text-sm hover:bg-white/10">
            <Download className="mr-2 h-4 w-4" /> Download
          </a>
        )}
      </div>
      <div className="border-t border-white/10 p-3">{children}</div>
    </div>
  );
}

function FileLink({ label, href }: { label: string; href?: string }) {
  if (!href) return null;
  return (
    <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="flex min-w-0 items-center gap-3">
        <Download className="h-4 w-4 shrink-0 text-white/60" />
        <span className="truncate" title={label}>{label}</span>
      </div>
      <a href={href} target="_blank" rel="noreferrer" className="inline-flex items-center rounded-lg border border-white/15 px-3 py-1 text-sm hover:bg-white/10">Download</a>
    </div>
  );
}
