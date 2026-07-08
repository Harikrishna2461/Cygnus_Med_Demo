import { useState, useRef, useCallback, useEffect } from "react";

const SAMPLE_OPTIONS = [
  { value: 3,  label: "High accuracy — every 3 frames" },
  { value: 5,  label: "Balanced — every 5 frames" },
  { value: 10, label: "Fast — every 10 frames" },
  { value: 20, label: "Very fast — every 20 frames" },
];

const CLS_META = {
  N1: { color: "#dc2626", label: "Deep Vein",        note: "below fascia" },
  N2: { color: "#06b6d4", label: "GSV",              note: "fascial compartment" },
  N3: { color: "#eab308", label: "Superficial Vein", note: "above fascia" },
};

export default function App() {
  const [file,            setFile]            = useState(null);
  const [isDragging,      setIsDragging]      = useState(false);
  const [sampleRate,      setSampleRate]      = useState(5);
  const [phase,           setPhase]           = useState("idle");
  const [progress,        setProgress]        = useState(0);
  const [statusMsg,       setStatusMsg]       = useState("");
  const [stats,           setStats]           = useState({ frames: 0, total: 0, veins: 0 });
  const [jobId,           setJobId]           = useState(null);
  const [streamUrl,       setStreamUrl]       = useState(null);
  const [downloadUrl,     setDownloadUrl]     = useState(null);
  const [previewUrl,      setPreviewUrl]      = useState(null);
  const [backendOk,       setBackendOk]       = useState(true);
  const [classifications, setClassifications] = useState([]);

  const fileInputRef = useRef(null);
  const esRef        = useRef(null);

  useEffect(() => () => esRef.current?.close(), []);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => setBackendOk(r.ok))
      .catch(() => setBackendOk(false));
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) setFile(f);
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) setFile(f);
  };

  const listenForProgress = (id) => {
    esRef.current?.close();
    const es = new EventSource(`/api/status/${id}`);
    esRef.current = es;

    es.onmessage = (e) => {
      const d = JSON.parse(e.data);
      setProgress(d.progress ?? 0);
      setStatusMsg(d.message ?? "");
      setStats({
        frames: d.frames_processed ?? 0,
        total:  d.total_frames     ?? 0,
        veins:  d.veins_found      ?? 0,
      });
      if (d.current_classifications) {
        setClassifications(d.current_classifications);
      }

      if (d.status === "done") {
        es.close();
        setPreviewUrl(`/api/preview/${id}`);
        setDownloadUrl(`/api/download/${id}`);
        setPhase("done");
      } else if (d.status === "error") {
        es.close();
        setPhase("error");
      }
    };

    es.onerror = () => {
      es.close();
      setPhase("error");
      setStatusMsg("Connection to server lost.");
    };
  };

  const startProcessing = async () => {
    if (!file) return;

    const fd = new FormData();
    fd.append("video", file);
    fd.append("sample_rate", sampleRate);

    setPhase("processing");
    setProgress(0);
    setStatusMsg("Uploading video...");
    setClassifications([]);

    try {
      const res = await fetch("/api/process", { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const { job_id } = await res.json();
      setJobId(job_id);
      setStreamUrl(`/api/stream/${job_id}`);
      listenForProgress(job_id);
    } catch (err) {
      setPhase("error");
      const msg = err.message === "Failed to fetch"
        ? "Cannot reach the backend. Make sure python3 app.py is running on port 5001, then reload this page."
        : err.message;
      setStatusMsg(msg);
      setBackendOk(false);
    }
  };

  const reset = () => {
    esRef.current?.close();
    setFile(null);
    setPhase("idle");
    setJobId(null);
    setProgress(0);
    setStatusMsg("");
    setStats({ frames: 0, total: 0, veins: 0 });
    setStreamUrl(null);
    setDownloadUrl(null);
    setPreviewUrl(null);
    setClassifications([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Vein VLM Classifier</h1>
        <p>Upload a simple-annotated ultrasound video — the VLM classifies each vein as N1, N2, or N3</p>
      </header>

      {!backendOk && (
        <div className="offline-banner">
          Backend not reachable — make sure <code>python3 app.py</code> is running,
          then&nbsp;
          <button className="reload-btn" onClick={() => window.location.reload()}>
            reload
          </button>
        </div>
      )}

      <div className="legend">
        <LegendBadge cls="N1" color="#dc2626" label="Deep Vein"        note="below fascia" />
        <LegendBadge cls="N2" color="#06b6d4" label="GSV"              note="within fascial compartment" />
        <LegendBadge cls="N3" color="#eab308" label="Superficial Vein" note="above fascia" />
        <div className="legend-fascia">
          <span className="fascia-swatch" />
          <span>Yellow lines = fascia layer boundary</span>
        </div>
      </div>

      <main className="card">

        {/* ── IDLE ── */}
        {phase === "idle" && (
          <>
            <div
              className={`upload-zone${isDragging ? " dragging" : ""}${file ? " has-file" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <input ref={fileInputRef} type="file" accept="video/*" hidden
                     onChange={handleFileChange} />
              <UploadIcon />
              {file ? (
                <>
                  <p className="upload-filename">{file.name}</p>
                  <p className="upload-sub">{(file.size / 1024 / 1024).toFixed(1)} MB — click to change</p>
                </>
              ) : (
                <>
                  <p className="upload-hint">Drag &amp; drop a video file here</p>
                  <p className="upload-sub">or <span className="link">click to browse</span></p>
                </>
              )}
            </div>

            <div className="settings-row">
              <label htmlFor="sample-select">Processing mode</label>
              <select id="sample-select" value={sampleRate}
                      onChange={(e) => setSampleRate(Number(e.target.value))}>
                {SAMPLE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            <p className="settings-hint">
              VLM is called once every {sampleRate} frame{sampleRate > 1 ? "s" : ""};
              classification is held between calls.
            </p>

            <button className="btn-primary" onClick={startProcessing} disabled={!file}>
              Classify Veins
            </button>
          </>
        )}

        {/* ── PROCESSING — live MJPEG stream + progress ── */}
        {phase === "processing" && (
          <div className="processing-panel">

            {streamUrl && (
              <div className="live-wrapper">
                <span className="live-badge">LIVE</span>
                <img
                  src={streamUrl}
                  alt="Live classification preview"
                  className="live-stream"
                />
              </div>
            )}

            {/* Live classification panel */}
            <ClassificationPanel classifications={classifications} />

            <div className="processing-footer">
              <div className="processing-header">
                <Spinner />
                <h2>Classifying veins...</h2>
              </div>

              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="status-msg">{statusMsg}</p>

              {stats.total > 0 && (
                <div className="stats-row">
                  <StatBox label="Frames"         value={`${stats.frames} / ${stats.total}`} />
                  <StatBox label="VLM detections" value={stats.veins} />
                  <StatBox label="Progress"        value={`${progress}%`} accent />
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── DONE — video player + download ── */}
        {phase === "done" && (
          <div className="result-panel">
            <div className="result-header">
              <CheckIcon />
              <h2>Classification complete</h2>
            </div>

            <video
              key={previewUrl}
              src={previewUrl}
              controls
              autoPlay
              className="result-video"
            />

            {/* Final frame classifications */}
            <ClassificationPanel classifications={classifications} />

            <div className="stats-row" style={{ marginTop: "1.25rem" }}>
              <StatBox label="Total frames"   value={stats.total} />
              <StatBox label="VLM detections" value={stats.veins} />
            </div>

            <div className="result-actions">
              <a href={downloadUrl} download="vein_classified.mp4" className="btn-primary">
                Download Annotated Video
              </a>
              <button className="btn-secondary" onClick={reset}>
                Process another video
              </button>
            </div>
          </div>
        )}

        {/* ── ERROR ── */}
        {phase === "error" && (
          <div className="error-panel">
            <ErrorIcon />
            <h2>Something went wrong</h2>
            <p className="error-msg">{statusMsg || "An unexpected error occurred."}</p>
            <button className="btn-secondary" onClick={reset}>Try again</button>
          </div>
        )}
      </main>

      <footer className="app-footer">
        Classification powered by Llama-4 Scout (Groq) &mdash; N1/N2/N3 per CHIVA protocol
      </footer>
    </div>
  );
}

/* ── Sub-components ── */

function ClassificationPanel({ classifications }) {
  if (!classifications || classifications.length === 0) return null;

  // Count per class
  const counts = classifications.reduce((acc, cls) => {
    acc[cls] = (acc[cls] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="cls-panel">
      <span className="cls-panel-label">Veins detected:</span>
      {Object.entries(counts).sort().map(([cls, count]) => {
        const meta = CLS_META[cls] || { color: "#aaa", label: cls, note: "" };
        return (
          <div key={cls} className="cls-pill" style={{ "--cls-color": meta.color }}>
            <span className="cls-pill-tag">{cls}</span>
            <span className="cls-pill-info">
              {meta.label}
              {count > 1 && <span className="cls-pill-count">&times;{count}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function LegendBadge({ cls, color, label, note }) {
  return (
    <div className="legend-badge">
      <span className="badge-dot" style={{ background: color }} />
      <span className="badge-cls" style={{ color }}>{cls}</span>
      <span className="badge-label">{label}</span>
      <span className="badge-note">({note})</span>
    </div>
  );
}

function StatBox({ label, value, accent }) {
  return (
    <div className={`stat-box${accent ? " stat-accent" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function Spinner() {
  return <div className="spinner" aria-label="Loading" />;
}

function CheckIcon() {
  return (
    <svg className="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
