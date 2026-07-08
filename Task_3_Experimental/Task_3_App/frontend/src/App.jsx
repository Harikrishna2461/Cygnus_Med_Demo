import { useState, useRef, useCallback, useEffect } from 'react'

const API = ''

const LABEL_META = {
  N1:      { color: '#ff5032', bg: '#ff503220', label: 'N1 — Deep Vein' },
  N2:      { color: '#32ff64', bg: '#32ff6420', label: 'N2 — GSV' },
  N3:      { color: '#3296ff', bg: '#3296ff20', label: 'N3 — Superficial' },
  unknown: { color: '#aaaaaa', bg: '#aaaaaa20', label: 'Unknown' },
}

const PIPELINES = {
  dl: {
    name: 'DL Pipeline',
    icon: '🧠',
    badge: 'DL',
    subtitle: 'UNet-ResNet50 · LSTM Tracking · VLM Classification',
    description: 'Task-specific deep learning models trained on annotated ultrasound data. Best accuracy on in-distribution scans.',
    features: ['UNet-ResNet50 fascia segmentation', 'UNet-ResNet50 vein segmentation', 'LSTM motion predictor', 'Geometric + VLM classification'],
    accent: '#5555cc',
    accentDim: '#5555cc30',
    endpoint: 'dl',
  },
  foundation: {
    name: 'Foundation Pipeline',
    icon: '🔬',
    badge: 'FM',
    subtitle: 'Depth Anything V2 · Grounding DINO · VLM Classification',
    description: 'Zero-shot foundation models. No task-specific training required — works on any scan type including unseen anatomies.',
    features: ['Depth Anything V2 fascia estimation', 'Grounding DINO vein detection', 'LSTM motion predictor', 'Geometric + VLM classification'],
    accent: '#b044cc',
    accentDim: '#b044cc30',
    endpoint: 'foundation',
  },
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function StatCard({ label, value, color, bg }) {
  return (
    <div style={{ background: bg, border: `1px solid ${color}`, borderRadius: 8,
                  padding: '10px 16px', display: 'flex', justifyContent: 'space-between',
                  alignItems: 'center', marginBottom: 8 }}>
      <span style={{ color: '#ccc', fontSize: 13 }}>{label}</span>
      <span style={{ color, fontSize: 22, fontWeight: 700 }}>{value}</span>
    </div>
  )
}

function FrameStatsTable({ records }) {
  if (!records.length) return null
  const cols = Object.keys(records[0])
  return (
    <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1e1e2e', position: 'sticky', top: 0 }}>
            {cols.map(c => (
              <th key={c} style={{ padding: '6px 10px', color: '#888',
                                   textAlign: 'left', borderBottom: '1px solid #333' }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.map((r, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#0d0d1a' : '#111122' }}>
              {cols.map(c => (
                <td key={c} style={{ padding: '5px 10px', borderBottom: '1px solid #222',
                                     color: c === 'label' ? LABEL_META[r[c]]?.color ?? '#fff' : '#ccc' }}>
                  {r[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Landing page ──────────────────────────────────────────────────────────────

function Landing({ onSelect }) {
  return (
    <div className="app">
      <header className="header">
        <span className="header-icon">🩺</span>
        <div>
          <h1>Ultrasound Vein Tracker</h1>
          <p>Select a pipeline to begin</p>
        </div>
      </header>

      <div className="landing">
        <div className="landing-title">
          <h2>Choose Your Pipeline</h2>
          <p>Both pipelines share the same LSTM tracker and VLM classifier. They differ in how fascia and veins are detected.</p>
        </div>

        <div style={{ textAlign: 'center', marginBottom: 16, display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => onSelect('fascia-test')}
                  style={{ background: '#00aa77', padding: '10px 28px', fontSize: 14 }}>
            📐 Fascia Detection Tester
          </button>
          <button className="btn" onClick={() => onSelect('vein-test')}
                  style={{ background: '#aa5500', padding: '10px 28px', fontSize: 14 }}>
            🔵 Vein Detection Tester
          </button>
          <button className="btn" onClick={() => onSelect('combined-test')}
                  style={{ background: '#7744cc', padding: '10px 28px', fontSize: 14 }}>
            🔬 Fascia + Vein Tester
          </button>
        </div>

        <div className="pipeline-grid">
          {Object.entries(PIPELINES).map(([key, p]) => (
            <div key={key} className="pipeline-card" onClick={() => onSelect(key)}
                 style={{ '--accent': p.accent, '--accent-dim': p.accentDim }}>
              <div className="pc-header">
                <span className="pc-icon">{p.icon}</span>
                <span className="pc-badge" style={{ background: p.accentDim, color: p.accent, border: `1px solid ${p.accent}` }}>
                  {p.badge}
                </span>
              </div>
              <h3 className="pc-name" style={{ color: p.accent }}>{p.name}</h3>
              <p className="pc-subtitle">{p.subtitle}</p>
              <p className="pc-desc">{p.description}</p>
              <ul className="pc-features">
                {p.features.map(f => (
                  <li key={f} style={{ color: '#999' }}>
                    <span style={{ color: p.accent }}>✓</span> {f}
                  </li>
                ))}
              </ul>
              <button className="btn pc-btn" style={{ background: p.accent }}>
                Launch {p.name} →
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Processing page (shared by both pipelines) ────────────────────────────────

function ProcessingPage({ pipelineKey, onBack }) {
  const p = PIPELINES[pipelineKey]

  const [phase, setPhase]             = useState('idle')
  const [uploadId, setUploadId]       = useState(null)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [frameIdx, setFrameIdx]       = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [counts, setCounts]           = useState({ N1: 0, N2: 0, N3: 0, unknown: 0 })
  const [detsThisFrame, setDetsThisFrame] = useState(0)
  const [allRecords, setAllRecords]   = useState([])
  const [error, setError]             = useState('')
  const [dragging, setDragging]       = useState(false)
  const [videoFile, setVideoFile]     = useState(null)
  const [speed, setSpeed]             = useState(0)

  const [groqKey, setGroqKey]   = useState('')
  const [fps, setFps]           = useState(5)
  const [minArea, setMinArea]   = useState(80)
  const [useVlm, setUseVlm]     = useState(true)
  const [vlmModel, setVlmModel] = useState('meta-llama/llama-4-scout-17b-16e-instruct')

  const esRef   = useRef(null)
  const t0Ref   = useRef(null)

  const resetState = () => {
    setPhase('idle'); setUploadId(null); setCurrentFrame(null)
    setFrameIdx(0); setTotalFrames(0); setCounts({ N1:0, N2:0, N3:0, unknown:0 })
    setDetsThisFrame(0); setAllRecords([]); setError(''); setSpeed(0)
    if (esRef.current) { esRef.current.close(); esRef.current = null }
  }

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov'].includes(ext)) {
      setError('Please upload an mp4, avi, or mov file.'); return
    }
    setVideoFile(file); setError('')
  }

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const startProcessing = async () => {
    if (!videoFile) { setError('Please select a video first.'); return }
    setError(''); setPhase('uploading')
    try {
      const form = new FormData()
      form.append('video', videoFile)
      const upRes = await fetch(`${API}/api/upload`, { method: 'POST', body: form })
      const { upload_id, error: upErr } = await upRes.json()
      if (upErr) throw new Error(upErr)
      setUploadId(upload_id)
      setPhase('processing')
      t0Ref.current = Date.now()

      const params = new URLSearchParams({ fps, min_area: minArea, use_vlm: useVlm, groq_key: groqKey, vlm_model: vlmModel })
      const es = new EventSource(`${API}/api/process/${p.endpoint}/${upload_id}?${params}`)
      esRef.current = es

      es.onmessage = async (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'meta') {
          setTotalFrames(msg.total)
        } else if (msg.type === 'frame') {
          setCurrentFrame(`data:image/jpeg;base64,${msg.image}`)
          setFrameIdx(msg.frame_idx + 1)
          setTotalFrames(msg.total)
          setCounts({ ...msg.counts })
          setDetsThisFrame(msg.dets)
          setSpeed(((msg.frame_idx + 1) / ((Date.now() - t0Ref.current) / 1000)).toFixed(1))
        } else if (msg.type === 'done') {
          es.close()
          setCounts({ ...msg.counts })
          try {
            const res = await fetch(`${API}/api/report/${upload_id}`)
            const text = await res.text()
            const lines = text.trim().split('\n')
            const headers = lines[0].split(',')
            const rows = lines.slice(1).map(line => {
              const vals = line.split(',')
              const obj = {}
              headers.forEach((h, i) => { obj[h.trim()] = vals[i]?.trim() ?? '' })
              return obj
            })
            setAllRecords(rows)
          } catch {}
          setPhase('done')
        }
      }
      es.onerror = () => {
        es.close()
        if (phase !== 'done') setError('Connection error during processing.')
        setPhase('done')
      }
    } catch (err) {
      setError(err.message); setPhase('idle')
    }
  }

  const downloadReport = () => {
    if (!uploadId) return
    const a = document.createElement('a')
    a.href = `${API}/api/report/${uploadId}`
    a.download = `vein_report_${uploadId.slice(0,8)}.csv`
    a.click()
  }

  const progress = totalFrames > 0 ? Math.round((frameIdx / totalFrames) * 100) : 0

  return (
    <div className="app">
      <header className="header">
        <span className="header-icon">{p.icon}</span>
        <div>
          <h1>{p.name}</h1>
          <p>{p.subtitle}</p>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {phase !== 'idle' && (
            <button className="btn btn-ghost" onClick={resetState}>↩ Reset</button>
          )}
          <button className="btn btn-ghost" onClick={() => { resetState(); onBack() }}>
            ← Pipelines
          </button>
        </div>
      </header>

      {/* Pipeline accent bar */}
      <div style={{ height: 3, background: `linear-gradient(90deg, ${p.accent}, transparent)` }} />

      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="pipeline-badge-sidebar" style={{ background: p.accentDim, border: `1px solid ${p.accent}`, borderRadius: 8, padding: '8px 12px', marginBottom: 12 }}>
            <span style={{ color: p.accent, fontWeight: 700, fontSize: 13 }}>{p.icon} {p.name}</span>
          </div>

          <h3>⚙️ Settings</h3>

          <label>Groq API Key</label>
          <input type="password" value={groqKey} onChange={e => setGroqKey(e.target.value)}
                 className="input" placeholder="gsk_..." />

          <label>VLM Model</label>
          <select value={vlmModel} onChange={e => setVlmModel(e.target.value)} className="input">
            <option value="meta-llama/llama-4-scout-17b-16e-instruct">Llama 4 Scout 17B (Groq)</option>
            <option value="llama-3.2-11b-vision-preview">Llama 3.2 11B Vision (Groq)</option>
          </select>

          <div className="toggle-row">
            <label>VLM Classification</label>
            <div className={`toggle ${useVlm ? 'on' : ''}`} style={{ '--toggle-on': p.accent }}
                 onClick={() => setUseVlm(!useVlm)}>
              <div className="thumb" />
            </div>
          </div>

          <label>Sample FPS: <strong>{fps}</strong></label>
          <input type="range" min={1} max={15} value={fps}
                 onChange={e => setFps(+e.target.value)} className="slider"
                 style={{ accentColor: p.accent }} />

          <label>Min Detection Area: <strong>{minArea}px</strong></label>
          <input type="range" min={10} max={200} value={minArea}
                 onChange={e => setMinArea(+e.target.value)} className="slider"
                 style={{ accentColor: p.accent }} />

          <div className="divider" />

          <h4>Label Guide</h4>
          {Object.entries(LABEL_META).map(([k, v]) => (
            <div key={k} className="legend-row">
              <span className="legend-dot" style={{ background: v.color }} />
              <span style={{ color: '#ccc', fontSize: 13 }}>{v.label}</span>
            </div>
          ))}
          <div className="legend-row">
            <span className="legend-dot" style={{ background: '#00dcdc' }} />
            <span style={{ color: '#ccc', fontSize: 13 }}>Fascia Layer</span>
          </div>
        </aside>

        {/* Main */}
        <main className="main">
          {phase === 'idle' && (
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''} ${videoFile ? 'has-file' : ''}`}
              style={videoFile ? {} : { '--dz-border': p.accent + '40' }}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input id="file-input" type="file" accept=".mp4,.avi,.mov,.MP4,.AVI"
                     style={{ display: 'none' }}
                     onChange={e => handleFile(e.target.files[0])} />
              {videoFile ? (
                <>
                  <div className="upload-icon">✅</div>
                  <p className="upload-title">{videoFile.name}</p>
                  <p className="upload-sub">{(videoFile.size / 1e6).toFixed(1)} MB — Click to change</p>
                </>
              ) : (
                <>
                  <div className="upload-icon">{p.icon}</div>
                  <p className="upload-title">Drop video here or click to browse</p>
                  <p className="upload-sub">MP4 · AVI · MOV</p>
                </>
              )}
            </div>
          )}

          {error && <div className="error-banner">⚠️ {error}</div>}

          {phase === 'idle' && videoFile && (
            <button className="btn btn-primary btn-start" onClick={startProcessing}
                    style={{ background: `linear-gradient(135deg, ${p.accent}, ${p.accent}99)` }}>
              ▶ Start {p.name}
            </button>
          )}

          {phase === 'uploading' && <div className="status-msg">⬆️ Uploading video...</div>}

          {(phase === 'processing' || phase === 'done') && currentFrame && (
            <div className="frame-section">
              <div className="frame-header">
                <span>
                  {phase === 'processing' ? '🔴 Live' : '✅ Complete'} —
                  Frame {frameIdx} / {totalFrames}
                  {phase === 'processing' && ` · ${speed} fps · ${detsThisFrame} detections`}
                </span>
                {phase === 'done' && (
                  <button className="btn btn-download" onClick={downloadReport}>⬇ Download CSV</button>
                )}
              </div>

              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%`, background: `linear-gradient(90deg, ${p.accent}, #32ff64)` }} />
              </div>

              <div className="frame-layout">
                <img src={currentFrame} alt="frame" className="frame-img" />
                <div className="stats-panel">
                  <h4 style={{ color: '#888', marginBottom: 12 }}>Cumulative Detections</h4>
                  {Object.entries(LABEL_META).map(([k, v]) => (
                    <StatCard key={k} label={v.label} value={counts[k] ?? 0} color={v.color} bg={v.bg} />
                  ))}
                  <div className="divider" />
                  <div style={{ color: '#555', fontSize: 12, textAlign: 'center' }}>
                    {phase === 'processing' ? `Processing at ${speed} frames/s` : 'Analysis complete'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {phase === 'done' && (
            <div className="report-section">
              <div className="report-header">
                <h3>📊 Frame-by-Frame Report</h3>
                <button className="btn btn-download" onClick={downloadReport}>⬇ Download CSV</button>
              </div>
              <p style={{ color: '#666', fontSize: 13, marginBottom: 12 }}>
                {allRecords.length} detection records across {frameIdx} processed frames.
              </p>
              {allRecords.length === 0 ? (
                <div style={{ color: '#555', textAlign: 'center', padding: 32 }}>
                  No records — click Download to get the full CSV from the server.
                </div>
              ) : (
                <FrameStatsTable records={allRecords} />
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

// ── Fascia Test Page ──────────────────────────────────────────────────────────

function FasciaTestPage({ onBack }) {
  const [phase, setPhase]         = useState('idle')
  const [videoFile, setVideoFile] = useState(null)
  const [uploadId, setUploadId]   = useState(null)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [frameIdx, setFrameIdx]   = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [fasciaY, setFasciaY]     = useState(null)
  const [fps, setFps]             = useState(5)
  const [error, setError]         = useState('')
  const [dragging, setDragging]   = useState(false)
  const esRef = useRef(null)

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov'].includes(ext)) { setError('mp4/avi/mov only'); return }
    setVideoFile(file); setError('')
  }

  const start = async () => {
    if (!videoFile) return
    setPhase('uploading'); setError('')
    const fd = new FormData(); fd.append('video', videoFile)
    const up = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
    if (!up.ok) { setError('Upload failed'); setPhase('idle'); return }
    const { upload_id } = await up.json()
    setUploadId(upload_id); setPhase('processing')

    const es = new EventSource(`${API}/api/test/fascia/${upload_id}?fps=${fps}`)
    esRef.current = es
    es.onmessage = (e) => {
      const d = JSON.parse(e.data)
      if (d.type === 'meta')  { setTotalFrames(d.total) }
      if (d.type === 'frame') { setCurrentFrame(d.image); setFrameIdx(d.frame_idx); setFasciaY(d.fascia_y) }
      if (d.type === 'done')  { setPhase('done'); es.close() }
    }
    es.onerror = () => { setError('Stream error'); setPhase('done'); es.close() }
  }

  const reset = () => {
    if (esRef.current) esRef.current.close()
    setPhase('idle'); setVideoFile(null); setUploadId(null)
    setCurrentFrame(null); setFrameIdx(0); setFasciaY(null); setError('')
  }

  const pct = totalFrames ? Math.round((frameIdx / totalFrames) * 100) : 0

  return (
    <div className="app">
      <header className="header">
        <button onClick={onBack} className="btn" style={{ marginRight: 16, background: '#333' }}>← Back</button>
        <span className="header-icon">📐</span>
        <div><h1>Fascia Detection Test</h1><p>Pipeline B — VLM-only, no trained DL model</p></div>
      </header>

      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 16px' }}>
        {phase === 'idle' && (
          <div>
            <div
              className={`drop-zone${dragging ? ' dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
              onClick={() => document.getElementById('ft-input').click()}
              style={{ cursor: 'pointer', border: '2px dashed #555', borderRadius: 12,
                       padding: 48, textAlign: 'center', color: '#888', marginBottom: 16 }}
            >
              {videoFile ? <span style={{ color: '#0af' }}>{videoFile.name}</span>
                         : <span>Drop a video here or click to browse</span>}
            </div>
            <input id="ft-input" type="file" accept=".mp4,.avi,.mov" style={{ display: 'none' }}
                   onChange={e => handleFile(e.target.files[0])} />
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
              <label style={{ color: '#888' }}>FPS:</label>
              <input type="number" value={fps} min={1} max={30}
                     onChange={e => setFps(Number(e.target.value))}
                     className="input" style={{ width: 80 }} />
            </div>
            {error && <p style={{ color: '#f55' }}>{error}</p>}
            <button className="btn" style={{ background: '#00aa77' }}
                    onClick={start} disabled={!videoFile}>
              Run Fascia Detection →
            </button>
          </div>
        )}

        {(phase === 'processing' || phase === 'done') && (
          <div>
            <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                {currentFrame && (
                  <img src={`data:image/jpeg;base64,${currentFrame}`}
                       style={{ width: '100%', borderRadius: 8, border: '1px solid #333' }} alt="fascia" />
                )}
                {phase === 'processing' && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ background: '#222', borderRadius: 4, height: 6 }}>
                      <div style={{ background: '#0af', width: `${pct}%`, height: '100%', borderRadius: 4, transition: 'width 0.3s' }} />
                    </div>
                    <p style={{ color: '#666', fontSize: 12, marginTop: 4 }}>Frame {frameIdx} / {totalFrames}</p>
                  </div>
                )}
              </div>
              <div style={{ width: 200, background: '#111', borderRadius: 8, padding: 16 }}>
                <h4 style={{ color: '#888', marginBottom: 12 }}>Fascia Position</h4>
                <div style={{ background: '#00aa7720', border: '1px solid #00aa77', borderRadius: 8,
                              padding: '12px 16px', textAlign: 'center' }}>
                  <div style={{ color: '#888', fontSize: 12 }}>fascia_y</div>
                  <div style={{ color: '#00aa77', fontSize: 32, fontWeight: 700 }}>
                    {fasciaY !== null ? fasciaY : '—'}
                  </div>
                  <div style={{ color: '#555', fontSize: 11 }}>px (0–255)</div>
                </div>
                {fasciaY !== null && (
                  <div style={{ marginTop: 12, background: '#222', borderRadius: 6, height: 200, position: 'relative' }}>
                    <div style={{ position: 'absolute', left: 0, right: 0,
                                  top: `${(fasciaY / 255) * 100}%`,
                                  borderTop: '2px solid #00ffff', transition: 'top 0.3s' }} />
                    <div style={{ position: 'absolute', bottom: 2, left: 4, color: '#333', fontSize: 10 }}>deep</div>
                    <div style={{ position: 'absolute', top: 2, left: 4, color: '#333', fontSize: 10 }}>surface</div>
                  </div>
                )}
                {phase === 'done' && (
                  <button className="btn" style={{ marginTop: 16, width: '100%', background: '#333' }}
                          onClick={reset}>Test Another</button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Vein Detection Test Page ──────────────────────────────────────────────────

function VeinTestPage({ onBack }) {
  const [phase, setPhase]             = useState('idle')
  const [videoFile, setVideoFile]     = useState(null)
  const [frameIdx, setFrameIdx]       = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [fps, setFps]                 = useState(5)
  const [error, setError]             = useState('')
  const [dragging, setDragging]       = useState(false)

  // All collected frames for playback
  const [allFrames, setAllFrames]     = useState([])   // { image, veins }
  const [playIdx, setPlayIdx]         = useState(0)
  const [isPlaying, setIsPlaying]     = useState(false)
  const [playSpeed, setPlaySpeed]     = useState(5)

  const esRef   = useRef(null)
  const timerRef = useRef(null)

  // Playback interval
  useEffect(() => {
    clearInterval(timerRef.current)
    if (isPlaying && allFrames.length > 0) {
      timerRef.current = setInterval(() => {
        setPlayIdx(i => {
          if (i >= allFrames.length - 1) { setIsPlaying(false); return i }
          return i + 1
        })
      }, 1000 / playSpeed)
    }
    return () => clearInterval(timerRef.current)
  }, [isPlaying, playSpeed, allFrames.length])

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov'].includes(ext)) { setError('mp4/avi/mov only'); return }
    setVideoFile(file); setError('')
  }

  const start = async () => {
    if (!videoFile) return
    setPhase('uploading'); setError(''); setAllFrames([]); setPlayIdx(0)
    const fd = new FormData(); fd.append('video', videoFile)
    const up = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
    if (!up.ok) { setError('Upload failed'); setPhase('idle'); return }
    const { upload_id } = await up.json()
    setPhase('processing')

    const es = new EventSource(`${API}/api/test/veins/${upload_id}?fps=${fps}`)
    esRef.current = es
    es.onmessage = (e) => {
      const d = JSON.parse(e.data)
      if (d.type === 'status') { setError(d.message) }
      if (d.type === 'meta')   { setTotalFrames(d.total); setError('') }
      if (d.type === 'frame')  {
        const f = { image: d.image, veins: d.veins || [] }
        setAllFrames(prev => { const n = [...prev, f]; setPlayIdx(n.length - 1); return n })
        setFrameIdx(d.frame_idx)
        setError('')
      }
      if (d.type === 'error') { setError(`Server error: ${d.message}`) }
      if (d.type === 'done')  {
        setPhase('done'); es.close()
        setPlayIdx(0)   // rewind to start for playback
      }
    }
    es.onerror = () => {
      setError('Stream error — Flask may not be running or needs restart')
      setPhase('done'); es.close()
    }
  }

  const reset = () => {
    clearInterval(timerRef.current)
    if (esRef.current) esRef.current.close()
    setPhase('idle'); setVideoFile(null); setAllFrames([])
    setFrameIdx(0); setTotalFrames(0); setPlayIdx(0)
    setIsPlaying(false); setError('')
  }

  const pct = totalFrames > 0 ? Math.round((frameIdx / totalFrames) * 100) : 0
  const cur  = allFrames[playIdx] || null

  return (
    <div className="app">
      <header className="header">
        <button onClick={onBack} className="btn" style={{ marginRight: 16, background: '#333' }}>← Back</button>
        <span className="header-icon">🔵</span>
        <div><h1>Vein Detection Test</h1><p>Pipeline B — Grounding DINO detection</p></div>
      </header>

      <div style={{ maxWidth: 1000, margin: '0 auto', padding: '24px 16px' }}>

        {/* ── Upload ── */}
        {phase === 'idle' && (
          <div>
            <div
              className={`drop-zone${dragging ? ' dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
              onClick={() => document.getElementById('vt-input').click()}
              style={{ cursor: 'pointer', border: '2px dashed #555', borderRadius: 12,
                       padding: 48, textAlign: 'center', color: '#888', marginBottom: 16 }}
            >
              {videoFile ? <span style={{ color: '#0af' }}>{videoFile.name}</span>
                         : <span>Drop a video here or click to browse</span>}
            </div>
            <input id="vt-input" type="file" accept=".mp4,.avi,.mov" style={{ display: 'none' }}
                   onChange={e => handleFile(e.target.files[0])} />
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
              <label style={{ color: '#888' }}>Sample FPS:</label>
              <input type="number" value={fps} min={1} max={30}
                     onChange={e => setFps(Number(e.target.value))}
                     className="input" style={{ width: 80 }} />
            </div>
            {error && <p style={{ color: '#f55' }}>{error}</p>}
            <button className="btn" style={{ background: '#aa5500' }}
                    onClick={start} disabled={!videoFile}>
              Run Vein Detection →
            </button>
          </div>
        )}

        {/* ── Processing / Playback ── */}
        {(phase === 'processing' || phase === 'done') && (
          <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>

            {/* Frame display */}
            <div style={{ flex: 1 }}>
              {error && <p style={{ color: '#f55', marginBottom: 8 }}>{error}</p>}
              {!cur && phase === 'done' && !error && (
                <p style={{ color: '#f88', marginBottom: 8 }}>No frames received — check Flask console</p>
              )}

              {cur && (
                <img src={`data:image/jpeg;base64,${cur.image}`}
                     style={{ width: '100%', borderRadius: 8, border: '1px solid #333' }} alt="veins" />
              )}

              {/* Live progress bar */}
              {phase === 'processing' && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ background: '#222', borderRadius: 4, height: 6 }}>
                    <div style={{ background: '#aa5500', width: `${pct}%`, height: '100%',
                                  borderRadius: 4, transition: 'width 0.3s' }} />
                  </div>
                  <p style={{ color: '#666', fontSize: 12, marginTop: 4 }}>
                    Processing — frame {frameIdx} / {totalFrames} ({allFrames.length} buffered)
                  </p>
                </div>
              )}

              {/* Playback controls (shown when done) */}
              {phase === 'done' && allFrames.length > 0 && (
                <div style={{ marginTop: 12, background: '#111', borderRadius: 8, padding: 12 }}>
                  {/* Scrubber */}
                  <input
                    type="range" min={0} max={allFrames.length - 1} value={playIdx}
                    onChange={e => { setIsPlaying(false); setPlayIdx(+e.target.value) }}
                    style={{ width: '100%', accentColor: '#00ffff', marginBottom: 10 }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {/* Rewind */}
                    <button onClick={() => { setIsPlaying(false); setPlayIdx(0) }}
                            style={{ background: '#222', border: '1px solid #444', borderRadius: 6,
                                     color: '#aaa', padding: '5px 10px', cursor: 'pointer', fontSize: 14 }}>
                      ⏮
                    </button>
                    {/* Play / Pause */}
                    <button onClick={() => setIsPlaying(p => !p)}
                            style={{ background: isPlaying ? '#aa5500' : '#00ffff22',
                                     border: `1px solid ${isPlaying ? '#aa5500' : '#00ffff'}`,
                                     borderRadius: 6, color: isPlaying ? '#fff' : '#00ffff',
                                     padding: '5px 16px', cursor: 'pointer', fontSize: 15, fontWeight: 700 }}>
                      {isPlaying ? '⏸ Pause' : '▶ Play'}
                    </button>
                    {/* Frame counter */}
                    <span style={{ color: '#666', fontSize: 12, marginLeft: 4 }}>
                      {playIdx + 1} / {allFrames.length}
                    </span>
                    {/* Speed selector */}
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ color: '#666', fontSize: 12 }}>Speed:</span>
                      {[1, 3, 5, 10].map(s => (
                        <button key={s} onClick={() => setPlaySpeed(s)}
                                style={{ background: playSpeed === s ? '#00ffff33' : '#1a1a2e',
                                         border: `1px solid ${playSpeed === s ? '#00ffff' : '#333'}`,
                                         borderRadius: 4, color: playSpeed === s ? '#00ffff' : '#555',
                                         padding: '3px 8px', cursor: 'pointer', fontSize: 11 }}>
                          {s}fps
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Side panel */}
            <div style={{ width: 180, background: '#111', borderRadius: 8, padding: 16, flexShrink: 0 }}>
              <h4 style={{ color: '#888', marginBottom: 12 }}>Detected Veins</h4>
              <div style={{ background: '#00ffff18', border: '1px solid #00ffff', borderRadius: 8,
                            padding: '10px 14px', textAlign: 'center', marginBottom: 12 }}>
                <div style={{ color: '#888', fontSize: 12 }}>this frame</div>
                <div style={{ color: '#00ffff', fontSize: 36, fontWeight: 700 }}>
                  {cur ? cur.veins.length : 0}
                </div>
                <div style={{ color: '#555', fontSize: 11 }}>veins</div>
              </div>

              {(cur?.veins || []).map((v, i) => (
                <div key={i} style={{ background: '#00ffff12', border: '1px solid #00ffff55',
                                      borderRadius: 4, padding: '5px 8px', marginBottom: 4,
                                      fontSize: 11, color: '#aaa' }}>
                  <span style={{ color: '#00ffff', fontWeight: 700 }}>V{i + 1}</span>
                  <span style={{ marginLeft: 6 }}>{v.x},{v.y} {v.w}×{v.h}</span>
                </div>
              ))}

              {phase === 'done' && (
                <button className="btn" style={{ marginTop: 16, width: '100%', background: '#333' }}
                        onClick={reset}>Test Another</button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Combined Fascia + Vein Test Page ─────────────────────────────────────────

function CombinedTestPage({ onBack }) {
  const [phase, setPhase]             = useState('idle')
  const [videoFile, setVideoFile]     = useState(null)
  const [frameIdx, setFrameIdx]       = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [fps, setFps]                 = useState(3)
  const [error, setError]             = useState('')
  const [dragging, setDragging]       = useState(false)

  const [allFrames, setAllFrames]     = useState([])   // { image, veins, fascia_y }
  const [playIdx, setPlayIdx]         = useState(0)
  const [isPlaying, setIsPlaying]     = useState(false)
  const [playSpeed, setPlaySpeed]     = useState(3)

  const esRef    = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    clearInterval(timerRef.current)
    if (isPlaying && allFrames.length > 0) {
      timerRef.current = setInterval(() => {
        setPlayIdx(i => {
          if (i >= allFrames.length - 1) { setIsPlaying(false); return i }
          return i + 1
        })
      }, 1000 / playSpeed)
    }
    return () => clearInterval(timerRef.current)
  }, [isPlaying, playSpeed, allFrames.length])

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov'].includes(ext)) { setError('mp4/avi/mov only'); return }
    setVideoFile(file); setError('')
  }

  const start = async () => {
    if (!videoFile) return
    setPhase('uploading'); setError(''); setAllFrames([]); setPlayIdx(0)
    const fd = new FormData(); fd.append('video', videoFile)
    const up = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
    if (!up.ok) { setError('Upload failed'); setPhase('idle'); return }
    const { upload_id } = await up.json()
    setPhase('processing')

    const es = new EventSource(`${API}/api/test/combined/${upload_id}?fps=${fps}`)
    esRef.current = es
    es.onmessage = (e) => {
      const d = JSON.parse(e.data)
      if (d.type === 'meta')  { setTotalFrames(d.total); setError('') }
      if (d.type === 'frame') {
        const f = { image: d.image, veins: d.veins || [], fascia_y: d.fascia_y ?? null }
        setAllFrames(prev => { const n = [...prev, f]; setPlayIdx(n.length - 1); return n })
        setFrameIdx(d.frame_idx)
        setError('')
      }
      if (d.type === 'error') { setError(`Server error: ${d.message}`) }
      if (d.type === 'done')  { setPhase('done'); es.close(); setPlayIdx(0) }
    }
    es.onerror = () => {
      setError('Stream error — Flask may not be running or needs restart')
      setPhase('done'); es.close()
    }
  }

  const reset = () => {
    clearInterval(timerRef.current)
    if (esRef.current) esRef.current.close()
    setPhase('idle'); setVideoFile(null); setAllFrames([])
    setFrameIdx(0); setTotalFrames(0); setPlayIdx(0)
    setIsPlaying(false); setError('')
  }

  const pct = totalFrames > 0 ? Math.round((frameIdx / totalFrames) * 100) : 0
  const cur  = allFrames[playIdx] || null
  const ACCENT = '#7744cc'

  return (
    <div className="app">
      <header className="header">
        <button onClick={onBack} className="btn" style={{ marginRight: 16, background: '#333' }}>← Back</button>
        <span className="header-icon">🔬</span>
        <div><h1>Fascia + Vein Detection</h1><p>Pipeline B — VLM fascia · Grounding DINO veins</p></div>
      </header>
      <div style={{ height: 3, background: `linear-gradient(90deg, ${ACCENT}, transparent)` }} />

      <div style={{ maxWidth: 1000, margin: '0 auto', padding: '24px 16px' }}>

        {phase === 'idle' && (
          <div>
            <div
              className={`drop-zone${dragging ? ' dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
              onClick={() => document.getElementById('cb-input').click()}
              style={{ cursor: 'pointer', border: `2px dashed ${ACCENT}66`, borderRadius: 12,
                       padding: 48, textAlign: 'center', color: '#888', marginBottom: 16 }}
            >
              {videoFile ? <span style={{ color: '#0af' }}>{videoFile.name}</span>
                         : <span>Drop a video here or click to browse</span>}
            </div>
            <input id="cb-input" type="file" accept=".mp4,.avi,.mov" style={{ display: 'none' }}
                   onChange={e => handleFile(e.target.files[0])} />
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
              <label style={{ color: '#888' }}>Sample FPS:</label>
              <input type="number" value={fps} min={1} max={10}
                     onChange={e => setFps(Number(e.target.value))}
                     className="input" style={{ width: 80 }} />
              <span style={{ color: '#555', fontSize: 12 }}>
                (lower = more accurate fascia, 2-3 recommended)
              </span>
            </div>
            {error && <p style={{ color: '#f55' }}>{error}</p>}
            <button className="btn" style={{ background: ACCENT }}
                    onClick={start} disabled={!videoFile}>
              Run Combined Detection →
            </button>
          </div>
        )}

        {(phase === 'processing' || phase === 'done') && (
          <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>

            <div style={{ flex: 1 }}>
              {error && <p style={{ color: '#f55', marginBottom: 8 }}>{error}</p>}
              {!cur && phase === 'done' && !error && (
                <p style={{ color: '#f88', marginBottom: 8 }}>No frames received — check Flask console</p>
              )}
              {cur && (
                <img src={`data:image/jpeg;base64,${cur.image}`}
                     style={{ width: '100%', borderRadius: 8, border: '1px solid #333' }} alt="combined" />
              )}

              {phase === 'processing' && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ background: '#222', borderRadius: 4, height: 6 }}>
                    <div style={{ background: ACCENT, width: `${pct}%`, height: '100%',
                                  borderRadius: 4, transition: 'width 0.3s' }} />
                  </div>
                  <p style={{ color: '#666', fontSize: 12, marginTop: 4 }}>
                    Processing — frame {frameIdx} / {totalFrames} ({allFrames.length} buffered)
                  </p>
                </div>
              )}

              {phase === 'done' && allFrames.length > 0 && (
                <div style={{ marginTop: 12, background: '#111', borderRadius: 8, padding: 12 }}>
                  <input
                    type="range" min={0} max={allFrames.length - 1} value={playIdx}
                    onChange={e => { setIsPlaying(false); setPlayIdx(+e.target.value) }}
                    style={{ width: '100%', accentColor: ACCENT, marginBottom: 10 }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <button onClick={() => { setIsPlaying(false); setPlayIdx(0) }}
                            style={{ background: '#222', border: '1px solid #444', borderRadius: 6,
                                     color: '#aaa', padding: '5px 10px', cursor: 'pointer', fontSize: 14 }}>
                      ⏮
                    </button>
                    <button onClick={() => setIsPlaying(p => !p)}
                            style={{ background: isPlaying ? ACCENT : ACCENT + '22',
                                     border: `1px solid ${ACCENT}`,
                                     borderRadius: 6, color: isPlaying ? '#fff' : ACCENT,
                                     padding: '5px 16px', cursor: 'pointer', fontSize: 15, fontWeight: 700 }}>
                      {isPlaying ? '⏸ Pause' : '▶ Play'}
                    </button>
                    <span style={{ color: '#666', fontSize: 12 }}>
                      {playIdx + 1} / {allFrames.length}
                    </span>
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ color: '#666', fontSize: 12 }}>Speed:</span>
                      {[1, 3, 5, 10].map(s => (
                        <button key={s} onClick={() => setPlaySpeed(s)}
                                style={{ background: playSpeed === s ? ACCENT + '33' : '#1a1a2e',
                                         border: `1px solid ${playSpeed === s ? ACCENT : '#333'}`,
                                         borderRadius: 4, color: playSpeed === s ? ACCENT : '#555',
                                         padding: '3px 8px', cursor: 'pointer', fontSize: 11 }}>
                          {s}fps
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Side panel */}
            <div style={{ width: 190, background: '#111', borderRadius: 8, padding: 16, flexShrink: 0 }}>

              {/* Fascia */}
              <h4 style={{ color: '#888', marginBottom: 8, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Fascia</h4>
              <div style={{ background: '#00ffff18', border: '1px solid #00ffff', borderRadius: 8,
                            padding: '10px 14px', textAlign: 'center', marginBottom: 16 }}>
                <div style={{ color: '#888', fontSize: 11 }}>depth (px)</div>
                <div style={{ color: '#00ffff', fontSize: 28, fontWeight: 700 }}>
                  {cur?.fascia_y !== null && cur?.fascia_y !== undefined ? cur.fascia_y : '—'}
                </div>
              </div>

              {/* Veins */}
              <h4 style={{ color: '#888', marginBottom: 8, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Veins</h4>
              <div style={{ background: '#ffff0018', border: '1px solid #ffff00', borderRadius: 8,
                            padding: '10px 14px', textAlign: 'center', marginBottom: 12 }}>
                <div style={{ color: '#888', fontSize: 11 }}>this frame</div>
                <div style={{ color: '#ffff00', fontSize: 28, fontWeight: 700 }}>
                  {cur ? cur.veins.length : 0}
                </div>
              </div>

              {(cur?.veins || []).map((v, i) => (
                <div key={i} style={{ background: '#ffff0012', border: '1px solid #ffff0055',
                                      borderRadius: 4, padding: '4px 8px', marginBottom: 4,
                                      fontSize: 11, color: '#aaa' }}>
                  <span style={{ color: '#ffff00', fontWeight: 700 }}>V{i + 1}</span>
                  <span style={{ marginLeft: 6 }}>{v.x},{v.y} {v.w}×{v.h}</span>
                </div>
              ))}

              {phase === 'done' && (
                <button className="btn" style={{ marginTop: 16, width: '100%', background: '#333' }}
                        onClick={reset}>Test Another</button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState(null)

  if (page === 'fascia-test')   return <FasciaTestPage    onBack={() => setPage(null)} />
  if (page === 'vein-test')     return <VeinTestPage      onBack={() => setPage(null)} />
  if (page === 'combined-test') return <CombinedTestPage  onBack={() => setPage(null)} />
  if (page)                     return <ProcessingPage pipelineKey={page} onBack={() => setPage(null)} />
  return <Landing onSelect={setPage} />
}
