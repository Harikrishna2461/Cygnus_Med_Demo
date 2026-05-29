import { useState, useRef, useCallback } from 'react'

const API = ''

const LABEL_META = {
  N1:      { color: '#ff5032', bg: '#ff503220', label: 'N1 — Deep Vein' },
  N2:      { color: '#32ff64', bg: '#32ff6420', label: 'N2 — GSV' },
  N3:      { color: '#3296ff', bg: '#3296ff20', label: 'N3 — Superficial' },
  unknown: { color: '#aaaaaa', bg: '#aaaaaa20', label: 'Unknown' },
}

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
                                   textAlign: 'left', borderBottom: '1px solid #333' }}>
                {c}
              </th>
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

export default function App() {
  const [phase, setPhase]           = useState('idle')   // idle | uploading | processing | done
  const [uploadId, setUploadId]     = useState(null)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [frameIdx, setFrameIdx]     = useState(0)
  const [totalFrames, setTotalFrames]   = useState(0)
  const [counts, setCounts]         = useState({ N1: 0, N2: 0, N3: 0, unknown: 0 })
  const [detsThisFrame, setDetsThisFrame] = useState(0)
  const [allRecords, setAllRecords] = useState([])
  const [error, setError]           = useState('')
  const [dragging, setDragging]     = useState(false)
  const [videoFile, setVideoFile]   = useState(null)
  const [speed, setSpeed]           = useState(0)

  // Settings
  const [groqKey, setGroqKey]   = useState('')
  const [fps, setFps]           = useState(5)
  const [minArea, setMinArea]   = useState(80)
  const [useVlm, setUseVlm]     = useState(true)
  const [vlmModel, setVlmModel] = useState('meta-llama/llama-4-scout-17b-16e-instruct')

  const esRef    = useRef(null)
  const t0Ref    = useRef(null)
  const recsRef  = useRef([])

  const resetState = () => {
    setPhase('idle'); setUploadId(null); setCurrentFrame(null)
    setFrameIdx(0); setTotalFrames(0); setCounts({ N1:0, N2:0, N3:0, unknown:0 })
    setDetsThisFrame(0); setAllRecords([]); setError(''); setSpeed(0)
    recsRef.current = []
    if (esRef.current) { esRef.current.close(); esRef.current = null }
  }

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov'].includes(ext)) {
      setError('Please upload an mp4, avi, or mov file.'); return
    }
    setVideoFile(file)
    setError('')
  }

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const startProcessing = async () => {
    if (!videoFile) { setError('Please select a video first.'); return }
    setError('')
    setPhase('uploading')

    try {
      const form = new FormData()
      form.append('video', videoFile)
      const upRes = await fetch(`${API}/api/upload`, { method: 'POST', body: form })
      const { upload_id, error: upErr } = await upRes.json()
      if (upErr) throw new Error(upErr)
      setUploadId(upload_id)
      setPhase('processing')
      t0Ref.current = Date.now()

      const params = new URLSearchParams({
        fps, min_area: minArea, use_vlm: useVlm,
        groq_key: groqKey, vlm_model: vlmModel,
      })
      const es = new EventSource(`${API}/api/process/${upload_id}?${params}`)
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
          const elapsed = (Date.now() - t0Ref.current) / 1000
          setSpeed(((msg.frame_idx + 1) / elapsed).toFixed(1))
        } else if (msg.type === 'done') {
          es.close()
          setCounts({ ...msg.counts })
          // Fetch the CSV records and parse for table display
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
          } catch (err) {
            console.error('Failed to fetch report:', err)
          }
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

  const downloadReport = async () => {
    if (!uploadId) return
    const a = document.createElement('a')
    a.href = `${API}/api/report/${uploadId}`
    a.download = `vein_report_${uploadId.slice(0,8)}.csv`
    a.click()
  }

  const progress = totalFrames > 0 ? Math.round((frameIdx / totalFrames) * 100) : 0

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <span className="header-icon">🩺</span>
        <div>
          <h1>Ultrasound Vein Tracker</h1>
          <p>Fascia detection · Vein segmentation · LSTM tracking · VLM classification</p>
        </div>
        {phase !== 'idle' && (
          <button className="btn btn-ghost" onClick={resetState}>↩ Reset</button>
        )}
      </header>

      <div className="layout">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <h3>⚙️ Settings</h3>

          <label>Groq API Key</label>
          <input type="password" value={groqKey} onChange={e => setGroqKey(e.target.value)}
                 className="input" placeholder="gsk_..." />

          <label>VLM Model</label>
          <select value={vlmModel} onChange={e => setVlmModel(e.target.value)} className="input">
            <option value="meta-llama/llama-4-scout-17b-16e-instruct">Llama 4 Scout 17B</option>
            <option value="llama-3.2-11b-vision-preview">Llama 3.2 11B</option>
            <option value="llama-3.2-90b-vision-preview">Llama 3.2 90B</option>
          </select>

          <div className="toggle-row">
            <label>VLM Classification</label>
            <div className={`toggle ${useVlm ? 'on' : ''}`} onClick={() => setUseVlm(!useVlm)}>
              <div className="thumb" />
            </div>
          </div>

          <label>Sample FPS: <strong>{fps}</strong></label>
          <input type="range" min={1} max={15} value={fps}
                 onChange={e => setFps(+e.target.value)} className="slider" />

          <label>Min Detection Area: <strong>{minArea}px</strong></label>
          <input type="range" min={10} max={200} value={minArea}
                 onChange={e => setMinArea(+e.target.value)} className="slider" />

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

        {/* ── Main ── */}
        <main className="main">
          {/* Upload zone */}
          {phase === 'idle' && (
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''} ${videoFile ? 'has-file' : ''}`}
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
                  <div className="upload-icon">📹</div>
                  <p className="upload-title">Drop video here or click to browse</p>
                  <p className="upload-sub">MP4 · AVI · MOV</p>
                </>
              )}
            </div>
          )}

          {error && <div className="error-banner">⚠️ {error}</div>}

          {phase === 'idle' && videoFile && (
            <button className="btn btn-primary btn-start" onClick={startProcessing}>
              ▶ Start Processing
            </button>
          )}

          {phase === 'uploading' && (
            <div className="status-msg">⬆️ Uploading video...</div>
          )}

          {/* Live frame viewer */}
          {(phase === 'processing' || phase === 'done') && currentFrame && (
            <div className="frame-section">
              <div className="frame-header">
                <span>
                  {phase === 'processing' ? '🔴 Live' : '✅ Complete'} —
                  Frame {frameIdx} / {totalFrames}
                  {phase === 'processing' && ` · ${speed} fps · ${detsThisFrame} detections`}
                </span>
                {phase === 'done' && (
                  <button className="btn btn-download" onClick={downloadReport}>
                    ⬇ Download Report CSV
                  </button>
                )}
              </div>

              {/* Progress bar */}
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>

              {/* Frame + stats side by side */}
              <div className="frame-layout">
                <img src={currentFrame} alt="frame" className="frame-img" />
                <div className="stats-panel">
                  <h4 style={{ color: '#888', marginBottom: 12 }}>Cumulative Detections</h4>
                  {Object.entries(LABEL_META).map(([k, v]) => (
                    <StatCard key={k} label={v.label} value={counts[k] ?? 0}
                              color={v.color} bg={v.bg} />
                  ))}
                  <div className="divider" />
                  <div style={{ color: '#555', fontSize: 12, textAlign: 'center' }}>
                    {phase === 'processing' ? `Processing at ${speed} frames/s` : 'Analysis complete'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Report table */}
          {phase === 'done' && (
            <div className="report-section">
              <div className="report-header">
                <h3>📊 Frame-by-Frame Report</h3>
                <button className="btn btn-download" onClick={downloadReport}>
                  ⬇ Download CSV
                </button>
              </div>
              <p style={{ color: '#666', fontSize: 13, marginBottom: 12 }}>
                Showing all {allRecords.length} detection records across {frameIdx} processed frames.
                Each row = one tracked vein in one frame.
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
