import React, { useState } from 'react';
import './VisionAnalyzer.css'; // Reuse existing styles

const VeinClassification = () => {
  // Upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Fascia parameters
  const [fasciaY, setFasciaY] = useState('');
  const [autoDetectFascia, setAutoDetectFascia] = useState(true);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
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
      
      // Add fascia parameters if provided
      if (fasciaY && !autoDetectFascia) {
        formData.append('fascia_center_y', fasciaY);
      }

      const response = await fetch('http://localhost:5002/api/vision/classify-veins-with-fascia', {
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
      setError(`Error classifying veins: ${err.message}`);
      console.error('Classification error:', err);
    } finally {
      setLoading(false);
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

  const getVeinColorDot = (veinType) => {
    const colors = {
      'N1_deep': '#00FF00',      // Green
      'N2_gsv': '#FF00FF',       // Magenta
      'N3_superficial': '#00A5FF' // Orange
    };
    return colors[veinType] || '#CCCCCC';
  };

  const getVeinTypeDisplay = (veinType) => {
    const displays = {
      'N1_deep': 'Deep Vein (N1)',
      'N2_gsv': 'GSV/N2',
      'N3_superficial': 'Superficial Vein (N3)',
      'TRB': 'Tributary'
    };
    return displays[veinType] || veinType;
  };

  return (
    <div className="vision-analyzer">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>🔬 Advanced Vein Classification</h1>
          <p>AI-powered vein classification using Vision Language Models with fascia-based positioning</p>
        </div>

        <div className="main-content">
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
                  <div className="icon">🖼️</div>
                  <h2>Upload Fascia-Annotated Ultrasound</h2>
                  <p>Drag and drop or click to select image</p>
                  <p className="file-types">PNG, JPG, GIF - Max 50MB</p>
                  <p className="note">📌 Image should have fascia layer clearly marked or visible</p>
                </div>
              </label>
            </div>

            {/* Preview */}
            {preview && (
              <div className="preview-section">
                <h3>📸 Original Image (with Fascia)</h3>
                <img src={preview} alt="Preview" className="preview-image" />
                {preview && <p className="preview-size">Image loaded successfully</p>}
              </div>
            )}

            {/* Fascia Parameters */}
            <div className="options">
              <h3>⚙️ Fascia Detection Settings</h3>
              
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={autoDetectFascia}
                  onChange={(e) => {
                    setAutoDetectFascia(e.target.checked);
                    if (e.target.checked) {
                      setFasciaY('');
                    }
                  }}
                />
                <span>Auto-detect fascia position (recommended)</span>
              </label>
              
              {!autoDetectFascia && (
                <div className="option-group">
                  <label>Fascia Center Y-Coordinate (pixels):</label>
                  <div className="option-input">
                    <input
                      type="number"
                      min="0"
                      value={fasciaY}
                      onChange={(e) => setFasciaY(e.target.value)}
                      placeholder="e.g., 250"
                      disabled={autoDetectFascia}
                    />
                    <span className="help-text">Manual override of fascia Y position (leave empty for auto-detection)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Info Section */}
            <div className="info-box" style={{ 
              background: '#f0f7ff', 
              border: '1px solid #667eea', 
              padding: '15px', 
              borderRadius: '8px',
              marginTop: '15px'
            }}>
              <h4>💡 Classification Rules</h4>
              <ul style={{ margin: '10px 0', paddingLeft: '20px' }}>
                <li><strong style={{color: '#00FF00'}}>N1 (Deep Vein)</strong> - Below the fascia layer</li>
                <li><strong style={{color: '#FF00FF'}}>N2 (GSV)</strong> - Within or very near the fascia (±10px)</li>
                <li><strong style={{color: '#00A5FF'}}>N3 (Superficial)</strong> - Above fascia, near skin surface</li>
              </ul>
              <p style={{ fontSize: '0.9em', color: '#555', marginTop: '10px' }}>
                The system uses Vision Language Models (LLaVA) combined with medical knowledge (RAG) for accurate classification.
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
                  <span className="spinner"></span> Analyzing with VLM...
                </>
              ) : (
                '🔍 Classify Veins'
              )}
            </button>

            {error && <div className="error-message">❌ {error}</div>}
          </div>

          {/* Results Section */}
          {results && (
            <div className="results-section">
              <h2>📊 Classification Results</h2>

              {/* Annotated Image */}
              {results.annotated_image && results.annotated_image.data && (
                <div className="visualization-area">
                  <h3>✅ Annotated Image with Bounding Boxes</h3>
                  <img
                    src={`data:image/png;base64,${results.annotated_image.data}`}
                    alt="Annotated Result"
                    className="result-image"
                    style={{ maxWidth: '100%', border: '2px solid #667eea', borderRadius: '8px' }}
                  />
                </div>
              )}

              {/* Summary Statistics */}
              <div className="summary-stats" style={{ marginTop: '30px' }}>
                <div className="stat">
                  <div className="stat-number">{results.detected_veins?.length || 0}</div>
                  <div className="stat-label">Veins Detected</div>
                </div>
                {results.summary?.by_type && Object.keys(results.summary.by_type).map(type => (
                  <div key={type} className="stat">
                    <div className="stat-number">{results.summary.by_type[type].count}</div>
                    <div className="stat-label">{results.summary.by_type[type].label}</div>
                  </div>
                ))}
                <div className="stat">
                  <div className="stat-number">{(results.summary?.average_confidence * 100).toFixed(0)}%</div>
                  <div className="stat-label">Avg Confidence</div>
                </div>
              </div>

              {/* Detailed Vein Analysis */}
              <div className="detection-results" style={{ marginTop: '30px' }}>
                <h3>🔍 Individual Vein Classifications</h3>
                
                {results.detected_veins && results.detected_veins.length > 0 ? (
                  <div className="veins-list">
                    {results.detected_veins.map((vein, idx) => (
                      <div 
                        key={idx} 
                        className="vein-card"
                        style={{
                          borderLeft: `5px solid ${getVeinColorDot(vein.vein_type)}`,
                          marginBottom: '15px'
                        }}
                      >
                        <div className="vein-header">
                          <span 
                            className="vein-type"
                            style={{
                              backgroundColor: getVeinColorDot(vein.vein_type),
                              color: 'white',
                              padding: '5px 12px',
                              borderRadius: '4px',
                              fontWeight: 'bold'
                            }}
                          >
                            {getVeinTypeDisplay(vein.vein_type)}
                          </span>
                          <span 
                            className="confidence"
                            style={{
                              backgroundColor: vein.confidence > 0.75 ? '#4CAF50' : 
                                              vein.confidence > 0.5 ? '#FF9800' : '#f44336',
                              color: 'white',
                              padding: '5px 12px',
                              borderRadius: '4px',
                              fontWeight: 'bold'
                            }}
                          >
                            {(vein.confidence * 100).toFixed(0)}% Confidence
                          </span>
                        </div>
                        
                        <div className="vein-details" style={{ margin: '12px 0' }}>
                          <div className="detail-row">
                            <span className="label">📍 Center Position:</span>
                            <span className="value">({vein.center[0].toFixed(0)}, {vein.center[1].toFixed(0)})</span>
                          </div>
                          
                          <div className="detail-row">
                            <span className="label">📏 Radius:</span>
                            <span className="value">{vein.radius.toFixed(1)} pixels</span>
                          </div>
                          
                          <div className="detail-row">
                            <span className="label">📍 Position Relative to Fascia:</span>
                            <span className="value" style={{
                              textTransform: 'capitalize',
                              fontWeight: 'bold',
                              color: vein.position === 'above_fascia' ? '#0066cc' :
                                     vein.position === 'within_fascia' ? '#ff6600' :
                                     vein.position === 'below_fascia' ? '#00cc00' : '#999'
                            }}>
                              {vein.position.replace(/_/g, ' ')}
                            </span>
                          </div>

                          {vein.distance_to_fascia !== null && (
                            <div className="detail-row">
                              <span className="label">📐 Distance to Fascia:</span>
                              <span className="value">
                                {vein.distance_to_fascia > 0 ? '↓' : '↑'} {Math.abs(vein.distance_to_fascia).toFixed(1)} pixels
                              </span>
                            </div>
                          )}

                          {vein.in_fascia_region && (
                            <div className="detail-row" style={{
                              backgroundColor: '#ffffcc',
                              padding: '8px',
                              borderRadius: '4px',
                              color: '#ff6600',
                              fontWeight: 'bold'
                            }}>
                              ⚠️ Located within fascia region
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

              {/* VLM Analysis Details */}
              {results.vlm_analysis && results.vlm_analysis.length > 0 && (
                <div className="llm-analysis" style={{
                  background: '#f0f7ff',
                  border: '1px solid #667eea',
                  padding: '15px',
                  borderRadius: '8px',
                  marginTop: '20px'
                }}>
                  <h3>🤖 Vision Language Model Analysis</h3>
                  <p style={{ color: '#555', whiteSpace: 'pre-wrap' }}>
                    {results.vlm_analysis.substring(0, 500)}...
                  </p>
                </div>
              )}

              {/* Legend */}
              <div className="legend" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '15px',
                marginTop: '25px',
                padding: '15px',
                backgroundColor: '#f8f9ff',
                borderRadius: '8px'
              }}>
                <h3 style={{ gridColumn: '1 / -1' }}>🎨 Color Legend</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '20px', height: '20px', backgroundColor: '#00FF00', borderRadius: '3px' }}></div>
                  <span>N1 - Deep Vein (Below Fascia)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '20px', height: '20px', backgroundColor: '#FF00FF', borderRadius: '3px' }}></div>
                  <span>N2 - GSV (Within/Near Fascia)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '20px', height: '20px', backgroundColor: '#00A5FF', borderRadius: '3px' }}></div>
                  <span>N3 - Superficial (Above Fascia)</span>
                </div>
              </div>

              {/* Export Buttons */}
              <div className="export-section" style={{ marginTop: '25px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button
                  className="export-button"
                  onClick={() => {
                    const element = document.createElement('a');
                    element.setAttribute(
                      'href',
                      `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(results, null, 2))}`
                    );
                    element.setAttribute('download', `vein_classification_${new Date().toISOString().slice(0, 10)}.json`);
                    element.style.display = 'none';
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                  }}
                >
                  📥 Export as JSON
                </button>

                <button
                  className="export-button"
                  onClick={() => {
                    if (results.annotated_image?.data) {
                      const link = document.createElement('a');
                      link.href = `data:image/png;base64,${results.annotated_image.data}`;
                      link.download = `vein_annotation_${new Date().toISOString().slice(0, 10)}.png`;
                      link.click();
                    }
                  }}
                  style={{ backgroundColor: '#764ba2' }}
                >
                  🖼️ Download Annotated Image
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
                  setFasciaY('');
                }}
              >
                🔄 Classify Another Image
              </button>
            </div>
          )}
        </div>

        {/* Info Section */}
        <div className="info-section">
          <h3>📖 Advanced Vein Classification System</h3>
          <ul>
            <li><strong>Vision Language Model:</strong> Uses LLaVA (open-source) for intelligent image analysis</li>
            <li><strong>Blob Detection:</strong> Automatically detects vein structures in ultrasound images</li>
            <li><strong>Fascia-Based Classification:</strong> Classifies based on spatial position relative to fascia layer</li>
            <li><strong>Medical Knowledge (RAG):</strong> Integrates medical literature for evidence-based classification</li>
            <li><strong>N1/N2/N3 Classification:</strong>
              <ul>
                <li><strong>N1 (Deep Veins):</strong> Located below the fascia layer</li>
                <li><strong>N2 (GSV):</strong> Great Saphenous Vein, within or near the fascia</li>
                <li><strong>N3 (Superficial):</strong> Superficial veins above the fascia, near skin</li>
              </ul>
            </li>
            <li><strong>Real-time Feedback:</strong> Get confidence scores and positional data for each detected vein</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VeinClassification;
