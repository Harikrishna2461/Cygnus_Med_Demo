import React, { useState } from 'react';
import './VisionAnalyzer.css';

const VisionAnalyzer = () => {
  // Image analysis state
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [enableLLM, setEnableLLM] = useState(true);  // ENABLED by default - uses free local Ollama
  
  // Video blob detection state
  const [analysisMode, setAnalysisMode] = useState('image');  // 'image' or 'video'
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoResults, setVideoResults] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const [maxFrames, setMaxFrames] = useState(300);
  const [skipFrames, setSkipFrames] = useState(1);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    }
  };

  const handleVideoFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      
      // Create preview (video URL preview)
      const videoUrl = URL.createObjectURL(file);
      setVideoPreview(videoUrl);
      setVideoError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('enable_llm', enableLLM.toString());
      formData.append('return_visualizations', 'true');

      const response = await fetch('http://localhost:5002/api/vision/analyze-frame', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Error analyzing image: ${err.message}`);
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeVideoBlobs = async () => {
    if (!videoFile) {
      setVideoError('Please select a video first');
      return;
    }

    setVideoLoading(true);
    setVideoError(null);
    setVideoResults(null);

    try {
      const formData = new FormData();
      formData.append('file', videoFile);
      formData.append('max_frames', maxFrames.toString());
      formData.append('skip_frames', skipFrames.toString());

      const response = await fetch('http://localhost:5002/api/vision/analyze-video-blobs', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setVideoResults(data);
    } catch (err) {
      setVideoError(`Error analyzing video: ${err.message}`);
      console.error('Video analysis error:', err);
    } finally {
      setVideoLoading(false);
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
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      setSelectedFile(file);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="vision-analyzer">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>🩺 Ultrasound Vein Analysis</h1>
          <p>Upload ultrasound images or videos to detect and track veins</p>
          
          {/* Mode Toggle */}
          <div className="mode-toggle">
            <button
              className={`mode-button ${analysisMode === 'image' ? 'active' : ''}`}
              onClick={() => {
                setAnalysisMode('image');
                setVideoError(null);
                setVideoResults(null);
              }}
            >
              📸 Image Analysis
            </button>
            <button
              className={`mode-button ${analysisMode === 'video' ? 'active' : ''}`}
              onClick={() => {
                setAnalysisMode('video');
                setError(null);
                setResults(null);
              }}
            >
              🎬 Video Blob Detection
            </button>
          </div>
        </div>

        <div className="main-content">
          {/* ===== IMAGE ANALYSIS MODE ===== */}
          {analysisMode === 'image' && (
            <>
          {/* Upload Section */}
          <div className="upload-section">
            <div
              className="drop-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="file-input"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input">
                <div className="drop-zone-content">
                  <div className="icon">📸</div>
                  <h2>Upload Ultrasound Image</h2>
                  <p>Drag and drop or click to select</p>
                  <p className="file-types">PNG, JPG, GIF - Max 50MB</p>
                </div>
              </label>
            </div>

            {/* Preview */}
            {preview && (
              <div className="preview-section">
                <h3>Original Image</h3>
                <img src={preview} alt="Preview" className="preview-image" />
              </div>
            )}

            {/* Options */}
            <div className="options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={enableLLM}
                  onChange={(e) => setEnableLLM(e.target.checked)}
                />
                <span>Enable AI Analysis (GPT-4 Vision)</span>
              </label>
              <p className="option-description">
                Uses advanced Vision LLM for GSV identification and ambiguous cases
              </p>
            </div>

            {/* Analyze Button */}
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Analyzing...
                </>
              ) : (
                '🔍 Analyze Image'
              )}
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>

          {/* Results Section */}
          {results && (
            <div className="results-section">
              <h2>Analysis Results</h2>

              {/* Visualization */}
              {results.visualization && results.visualization.data && (
                <div className="visualization-area">
                  <h3>Marked Up Image</h3>
                  <img
                    src={`data:image/png;base64,${results.visualization.data}`}
                    alt="Segmentation Result"
                    className="result-image"
                  />
                </div>
              )}

              {/* Vein Detection Results */}
              <div className="detection-results">
                <h3>🔍 Detected Veins ({results.veins?.length || 0})</h3>
                
                {results.veins && results.veins.length > 0 ? (
                  <div className="veins-list">
                    {results.veins.map((vein, idx) => (
                      <div key={idx} className="vein-card">
                        <div className="vein-header">
                          <span className={`vein-type ${vein.primary_classification}`}>
                            {vein.primary_classification?.replace('_', ' ').toUpperCase()}
                          </span>
                          <span className={`n-level level-${vein.n_level}`}>
                            {vein.n_level}
                          </span>
                        </div>
                        
                        <div className="vein-details">
                          <div className="detail-row">
                            <span className="label">Classification:</span>
                            <span className="value">{vein.primary_classification}</span>
                          </div>
                          
                          <div className="detail-row">
                            <span className="label">Depth Level:</span>
                            <span className="value">{vein.n_level}</span>
                          </div>
                          
                          <div className="detail-row">
                            <span className="label">Confidence:</span>
                            <span className="value">
                              {(vein.confidence * 100).toFixed(1)}%
                            </span>
                          </div>

                          {vein.distance_to_fascia_mm && (
                            <div className="detail-row">
                              <span className="label">Distance to Fascia:</span>
                              <span className="value">
                                {vein.distance_to_fascia_mm.toFixed(1)} mm
                              </span>
                            </div>
                          )}

                          {vein.relative_position && (
                            <div className="detail-row">
                              <span className="label">Position:</span>
                              <span className="value">{vein.relative_position}</span>
                            </div>
                          )}

                          {vein.reasoning && (
                            <div className="detail-row full-width">
                              <span className="label">Reasoning:</span>
                              <span className="value">{vein.reasoning}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-results">No veins detected in this image</p>
                )}
              </div>

              {/* Fascia Detection */}
              {results.fascia_detected && (
                <div className="fascia-info">
                  <h3>✓ Fascia Detected</h3>
                  <p>Fascial layer was successfully segmented in the image</p>
                </div>
              )}

              {/* Summary Stats */}
              <div className="summary-stats">
                <div className="stat">
                  <div className="stat-number">{results.veins?.length || 0}</div>
                  <div className="stat-label">Veins Detected</div>
                </div>
                <div className="stat">
                  <div className="stat-number">
                    {results.veins?.filter(v => v.primary_classification === 'deep_vein').length || 0}
                  </div>
                  <div className="stat-label">Deep Veins</div>
                </div>
                <div className="stat">
                  <div className="stat-number">
                    {results.veins?.filter(v => v.primary_classification === 'superficial_vein').length || 0}
                  </div>
                  <div className="stat-label">Superficial</div>
                </div>
                <div className="stat">
                  <div className="stat-number">
                    {results.veins?.filter(v => v.primary_classification === 'perforator_vein').length || 0}
                  </div>
                  <div className="stat-label">Perforators</div>
                </div>
              </div>

              {/* Comparison Grid (if available) */}
              {results.visualizations?.comparison_grid && (
                <div className="comparison-section">
                  <h3>Comparison Grid</h3>
                  <img
                    src={`data:image/png;base64,${results.visualizations.comparison_grid}`}
                    alt="Comparison Grid"
                    className="comparison-image"
                  />
                </div>
              )}

              {/* Detailed Analysis (if available) */}
              {results.visualizations?.detailed_analysis && (
                <div className="detailed-section">
                  <h3>Detailed Analysis</h3>
                  <img
                    src={`data:image/png;base64,${results.visualizations.detailed_analysis}`}
                    alt="Detailed Analysis"
                    className="detailed-image"
                  />
                </div>
              )}

              {/* Export Button */}
              <div className="export-section">
                <button
                  className="export-button"
                  onClick={() => {
                    const element = document.createElement('a');
                    element.setAttribute(
                      'href',
                      `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(results, null, 2))}`
                    );
                    element.setAttribute('download', 'analysis_results.json');
                    element.style.display = 'none';
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                  }}
                >
                  📥 Export Results (JSON)
                </button>
              </div>

              {/* Reset Button */}
              <button
                className="reset-button"
                onClick={() => {
                  setSelectedFile(null);
                  setPreview(null);
                  setResults(null);
                  setError(null);
                }}
              >
                🔄 Analyze Another Image
              </button>
            </div>
          )}
            </>
          )}

          {/* ===== VIDEO BLOB DETECTION MODE ===== */}
          {analysisMode === 'video' && (
            <>
          {/* Video Upload Section */}
          <div className="upload-section">
            <div
              className="drop-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                  const file = files[0];
                  setVideoFile(file);
                  const videoUrl = URL.createObjectURL(file);
                  setVideoPreview(videoUrl);
                }
              }}
            >
              <input
                type="file"
                id="video-input"
                accept="video/*"
                onChange={handleVideoFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="video-input">
                <div className="drop-zone-content">
                  <div className="icon">🎬</div>
                  <h2>Upload Ultrasound Video</h2>
                  <p>Drag and drop or click to select</p>
                  <p className="file-types">MP4, AVI, MOV - Max 500MB</p>
                </div>
              </label>
            </div>

            {/* Video Preview */}
            {videoPreview && (
              <div className="preview-section">
                <h3>Video Preview</h3>
                <video
                  src={videoPreview}
                  className="preview-video"
                  controls
                  style={{ maxWidth: '100%', maxHeight: '400px' }}
                />
              </div>
            )}

            {/* Blob Detection Options */}
            <div className="options">
              <div className="option-group">
                <label>Max Frames to Process:</label>
                <div className="option-input">
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(parseInt(e.target.value) || 300)}
                  />
                  <span className="help-text">Process up to this many frames</span>
                </div>
              </div>

              <div className="option-group">
                <label>Skip Frames:</label>
                <div className="option-input">
                  <input
                    type="number"
                    min="1"
                    max="30"
                    value={skipFrames}
                    onChange={(e) => setSkipFrames(parseInt(e.target.value) || 1)}
                  />
                  <span className="help-text">Process every Nth frame (e.g., 3 = every 3rd frame)</span>
                </div>
              </div>

              <p className="option-description">
                🤖 Uses KLT optical flow tracking to detect and follow two blobs (veins) across frames
              </p>
            </div>

            {/* Analyze Video Button */}
            <button
              className="analyze-button"
              onClick={handleAnalyzeVideoBlobs}
              disabled={!videoFile || videoLoading}
            >
              {videoLoading ? (
                <>
                  <span className="spinner"></span> Processing Video...
                </>
              ) : (
                '🔍 Analyze Video Blobs'
              )}
            </button>

            {videoError && <div className="error-message">{videoError}</div>}
          </div>

          {/* Video Results Section */}
          {videoResults && (
            <div className="results-section">
              <h2>Video Blob Detection Results</h2>

              {/* Output Video */}
              {videoResults.summary?.output_video?.data && (
                <div className="visualization-area">
                  <h3>Processed Video (with overlays)</h3>
                  <video
                    src={`data:video/mp4;base64,${videoResults.summary.output_video.data}`}
                    className="result-video"
                    controls
                    style={{ maxWidth: '100%', maxHeight: '500px' }}
                  />
                </div>
              )}

              {/* Summary Stats */}
              <div className="summary-stats">
                <div className="stat">
                  <div className="stat-number">{videoResults.frames_processed}</div>
                  <div className="stat-label">Frames Processed</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{videoResults.summary?.successful_detections}</div>
                  <div className="stat-label">Successful Detections</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{videoResults.summary?.detection_rate}</div>
                  <div className="stat-label">Detection Rate</div>
                </div>
                <div className="stat">
                  <div className="stat-number">{videoResults.summary?.average_confidence?.toFixed(1)}</div>
                  <div className="stat-label">Avg Confidence</div>
                </div>
              </div>

              {/* Detection Timeline */}
              <div className="detection-timeline">
                <h3>📊 Frame-by-Frame Detections</h3>
                <div className="timeline-viewer" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                  {videoResults.detections?.map((det, idx) => (
                    <div key={idx} className={`timeline-frame ${det.successful ? 'success' : 'failed'}`}>
                      <div className="timeline-header">
                        <span className="frame-num">Frame {det.frame_idx}</span>
                        <span className={`status-badge ${det.successful ? 'detected' : 'missed'}`}>
                          {det.successful ? '✓ Detected' : '✗ No Detection'}
                        </span>
                        <span className="confidence">{det.confidence?.toFixed(0)}%</span>
                      </div>
                      {det.targets && det.targets.length > 0 && (
                        <div className="targets-list">
                          {det.targets.map((target, tidx) => (
                            <div key={tidx} className="target-item">
                              <span className="target-id">{target.id}</span>
                              <span className="target-info">
                                Center: ({target.center?.[0]}, {target.center?.[1]})
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Export Button */}
              <div className="export-section">
                <button
                  className="export-button"
                  onClick={() => {
                    const element = document.createElement('a');
                    element.setAttribute(
                      'href',
                      `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(videoResults, null, 2))}`
                    );
                    element.setAttribute('download', 'video_analysis_results.json');
                    element.style.display = 'none';
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                  }}
                >
                  📥 Export Results (JSON)
                </button>
              </div>

              {/* Reset Button */}
              <button
                className="reset-button"
                onClick={() => {
                  setVideoFile(null);
                  setVideoPreview(null);
                  setVideoResults(null);
                  setVideoError(null);
                }}
              >
                🔄 Analyze Another Video
              </button>
            </div>
          )}
            </>
          )}
        </div>

        {/* Info Section */}
        <div className="info-section">
          <h3>📚 How It Works</h3>
          <ul>
            <li><strong>Upload:</strong> Select an ultrasound image (PNG, JPG)</li>
            <li><strong>Analyze:</strong> AI detects fascia and vein structures</li>
            <li><strong>Classify:</strong> Veins classified by type (GSV, Deep, Perforator, etc.)</li>
            <li><strong>Results:</strong> Get marked image with measurements and insights</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VisionAnalyzer;
