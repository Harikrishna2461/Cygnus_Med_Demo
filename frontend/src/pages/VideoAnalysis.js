import React, { useState, useRef, useEffect, useCallback } from 'react';
import './VisionAnalyzer.css';

const VideoAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [frameSkip, setFrameSkip] = useState(3);
  const [maxFrames, setMaxFrames] = useState(300);

  // Playback state
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const playIntervalRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);
      setError(null);
      setCurrentFrameIdx(0);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
  };
  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('drag-over');
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
      setResults(null);
      setError(null);
      setCurrentFrameIdx(0);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setCurrentFrameIdx(0);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('frame_skip', frameSkip);
      formData.append('max_frames', maxFrames);

      const response = await fetch('http://localhost:5002/api/vision/analyze-video-realtime', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || response.statusText);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Playback controls
  const stopPlayback = useCallback(() => {
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const startPlayback = useCallback(() => {
    if (!results || !results.annotated_frames || results.annotated_frames.length === 0) return;
    stopPlayback();
    setIsPlaying(true);

    const fps = results.video_info?.fps || 30;
    const effectiveFps = fps / (results.processing?.frame_skip || 1);
    const interval = Math.max(30, 1000 / (effectiveFps * playbackSpeed));

    playIntervalRef.current = setInterval(() => {
      setCurrentFrameIdx(prev => {
        const next = prev + 1;
        if (next >= results.annotated_frames.length) {
          stopPlayback();
          return 0;
        }
        return next;
      });
    }, interval);
  }, [results, playbackSpeed, stopPlayback]);

  useEffect(() => {
    return () => stopPlayback();
  }, [stopPlayback]);

  const togglePlayback = () => {
    if (isPlaying) stopPlayback();
    else startPlayback();
  };

  const currentFrame = results?.frames?.[currentFrameIdx];
  const totalFrames = results?.annotated_frames?.length || 0;

  const getClassColor = (cls) => ({
    N1: '#00ff00', N2: '#ff00ff', N3: '#ffa500',
  }[cls] || '#ccc');

  const getClassLabel = (cls) => ({
    N1: 'Deep Vein (N1)', N2: 'GSV (N2)', N3: 'Superficial (N3)',
  }[cls] || cls);

  return (
    <div className="vision-analyzer">
      <div className="container">
        <div className="header">
          <h1>🎬 Real-Time Video Vein Analysis</h1>
          <p>Upload an ultrasound video for frame-by-frame fascia detection and vein classification</p>
        </div>

        <div className="main-content">
          {/* Upload */}
          <div className="upload-section">
            <div className="drop-zone" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
              <input type="file" id="video-input" accept="video/*" onChange={handleFileSelect} style={{ display: 'none' }} />
              <label htmlFor="video-input">
                <div className="drop-zone-content">
                  <div className="icon">🎬</div>
                  <h2>Upload Ultrasound Video</h2>
                  <p>Drag and drop or click to select</p>
                  <p className="file-types">MP4, AVI, MOV</p>
                </div>
              </label>
            </div>

            {selectedFile && (
              <div style={{ marginTop: 12, padding: '10px 15px', background: '#1a1a2e', borderRadius: 8, color: '#ccc' }}>
                Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)
              </div>
            )}

            {/* Settings */}
            <div className="options" style={{ marginTop: 15 }}>
              <h3>Settings</h3>
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                <div className="option-group">
                  <label>Process every Nth frame:</label>
                  <input type="number" min="1" max="30" value={frameSkip}
                    onChange={e => setFrameSkip(parseInt(e.target.value) || 1)}
                    style={{ width: 60, padding: '4px 8px', borderRadius: 4, border: '1px solid #444', background: '#1a1a2e', color: '#eee' }} />
                </div>
                <div className="option-group">
                  <label>Max frames:</label>
                  <input type="number" min="10" max="1000" value={maxFrames}
                    onChange={e => setMaxFrames(parseInt(e.target.value) || 100)}
                    style={{ width: 80, padding: '4px 8px', borderRadius: 4, border: '1px solid #444', background: '#1a1a2e', color: '#eee' }} />
                </div>
              </div>
            </div>

            <button className="analyze-button" onClick={handleAnalyze} disabled={!selectedFile || loading}>
              {loading ? (<><span className="spinner"></span> Processing Video...</>) : '🔍 Analyze Video'}
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>

          {/* Results */}
          {results && (
            <div className="results-section">
              {/* Video Info */}
              <div className="summary-stats">
                <div className="stat">
                  <div className="stat-number">{results.video_info?.duration_s}s</div>
                  <div className="stat-label">Duration</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{results.processing?.frames_processed}</div>
                  <div className="stat-label">Frames Analyzed</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{results.processing?.avg_ms_per_frame}ms</div>
                  <div className="stat-label">Per Frame</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{results.processing?.processing_time_ms}ms</div>
                  <div className="stat-label">Total Time</div>
                </div>
              </div>

              {/* Detection Summary */}
              <div className="summary-stats" style={{ marginTop: 15 }}>
                <div className="stat">
                  <div className="stat-number" style={{ color: '#ffa500' }}>{results.summary?.N3_superficial || 0}</div>
                  <div className="stat-label">N3 Superficial</div>
                </div>
                <div className="stat">
                  <div className="stat-number" style={{ color: '#ff00ff' }}>{results.summary?.N2_gsv || 0}</div>
                  <div className="stat-label">N2 GSV</div>
                </div>
                <div className="stat">
                  <div className="stat-number" style={{ color: '#00ff00' }}>{results.summary?.N1_deep || 0}</div>
                  <div className="stat-label">N1 Deep</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{(results.summary?.avg_confidence * 100).toFixed(0)}%</div>
                  <div className="stat-label">Avg Confidence</div>
                </div>
              </div>

              {/* Video Player */}
              {totalFrames > 0 && (
                <div style={{ marginTop: 25 }}>
                  <h3>Annotated Video Playback</h3>

                  {/* Frame display */}
                  <div style={{ position: 'relative', background: '#000', borderRadius: 8, overflow: 'hidden', marginTop: 10 }}>
                    <img
                      src={`data:image/jpeg;base64,${results.annotated_frames[currentFrameIdx]}`}
                      alt={`Frame ${currentFrameIdx}`}
                      style={{ width: '100%', display: 'block' }}
                    />
                    {/* Frame info overlay */}
                    <div style={{
                      position: 'absolute', top: 10, right: 10, background: 'rgba(0,0,0,0.7)',
                      color: '#fff', padding: '4px 10px', borderRadius: 4, fontSize: 13,
                    }}>
                      Frame {currentFrameIdx + 1}/{totalFrames}
                      {currentFrame && ` | ${currentFrame.timestamp_ms}ms`}
                      {currentFrame && ` | ${currentFrame.num_veins} veins`}
                    </div>
                  </div>

                  {/* Controls */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
                    <button onClick={togglePlayback} style={{
                      padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
                      background: isPlaying ? '#e74c3c' : '#667eea', color: '#fff', fontWeight: 'bold', fontSize: 14,
                    }}>
                      {isPlaying ? '⏸ Pause' : '▶ Play'}
                    </button>

                    <button onClick={() => setCurrentFrameIdx(0)} style={{
                      padding: '8px 14px', borderRadius: 6, border: '1px solid #444',
                      background: '#1a1a2e', color: '#ccc', cursor: 'pointer',
                    }}>⏮</button>

                    <button onClick={() => setCurrentFrameIdx(Math.max(0, currentFrameIdx - 1))} style={{
                      padding: '8px 14px', borderRadius: 6, border: '1px solid #444',
                      background: '#1a1a2e', color: '#ccc', cursor: 'pointer',
                    }}>◀</button>

                    <button onClick={() => setCurrentFrameIdx(Math.min(totalFrames - 1, currentFrameIdx + 1))} style={{
                      padding: '8px 14px', borderRadius: 6, border: '1px solid #444',
                      background: '#1a1a2e', color: '#ccc', cursor: 'pointer',
                    }}>▶</button>

                    <button onClick={() => setCurrentFrameIdx(totalFrames - 1)} style={{
                      padding: '8px 14px', borderRadius: 6, border: '1px solid #444',
                      background: '#1a1a2e', color: '#ccc', cursor: 'pointer',
                    }}>⏭</button>

                    {/* Scrubber */}
                    <input type="range" min={0} max={totalFrames - 1} value={currentFrameIdx}
                      onChange={e => { stopPlayback(); setCurrentFrameIdx(parseInt(e.target.value)); }}
                      style={{ flex: 1, minWidth: 120 }} />

                    {/* Speed */}
                    <select value={playbackSpeed} onChange={e => setPlaybackSpeed(parseFloat(e.target.value))}
                      style={{ padding: '6px 8px', borderRadius: 4, border: '1px solid #444', background: '#1a1a2e', color: '#ccc' }}>
                      <option value={0.25}>0.25x</option>
                      <option value={0.5}>0.5x</option>
                      <option value={1}>1x</option>
                      <option value={2}>2x</option>
                      <option value={4}>4x</option>
                    </select>
                  </div>

                  {/* Current frame detections */}
                  {currentFrame && currentFrame.detections.length > 0 && (
                    <div style={{ marginTop: 15 }}>
                      <h4>Detections in Frame {currentFrameIdx + 1}</h4>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 8 }}>
                        {currentFrame.detections.map((d, i) => (
                          <div key={i} style={{
                            padding: '8px 14px', borderRadius: 6,
                            borderLeft: `4px solid ${getClassColor(d.classification)}`,
                            background: '#1a1a2e', color: '#eee', fontSize: 13,
                          }}>
                            <strong>{getClassLabel(d.classification)}</strong>
                            <span style={{ marginLeft: 8, opacity: 0.7 }}>
                              {(d.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Fascia info */}
                  {currentFrame?.fascia_bounds && (
                    <div style={{ marginTop: 10, fontSize: 13, color: '#aaa' }}>
                      Fascia detected: y={currentFrame.fascia_bounds[0]} to y={currentFrame.fascia_bounds[1]}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoAnalysis;
