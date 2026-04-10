import React, { useState } from 'react';
import axios from 'axios';

const ProbeGuidance = () => {
  const [mode, setMode] = useState('single'); // 'single' or 'stream'
  
  // Single mode state - Use same format as Task-1
  const [ultrasoundData, setUltrasoundData] = useState(JSON.stringify({
    sequenceNumber: 1,
    fromType: "N1",
    toType: "N2",
    step: "SFJ-Knee",
    flow: "RP",
    posXRatio: 0.30,
    posYRatio: 0.35,
    clipPath: "video-frame-001.png",
    legSide: "left",
    confidence: 0.92,
    reflux_duration: 1.1,
    description: "SFJ area - reflux detected"
  }, null, 2));
  
  // Stream mode state - SAME input format as Task-1 with EP/RP flow information
  // Probe guidance based on clinical findings (RP/EP), NOT just coordinates
  const [streamData, setStreamData] = useState(JSON.stringify([
    { sequenceNumber: 1, fromType: "N1", toType: "N1", step: "SFJ-Knee", flow: "EP", posXRatio: 0.15, posYRatio: 0.25, clipPath: "frame-001.png", legSide: "left", confidence: 0.95, reflux_duration: 0.0, description: "Groin area - normal GSV" },
    { sequenceNumber: 2, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.25, posYRatio: 0.40, clipPath: "frame-002.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Mid-groin - normal tributaries" },
    { sequenceNumber: 3, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.30, posYRatio: 0.35, clipPath: "frame-003.png", legSide: "left", confidence: 0.88, reflux_duration: 0.6, description: "SFJ area - reflux detected move medial" },
    { sequenceNumber: 4, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.28, posYRatio: 0.38, clipPath: "frame-004.png", legSide: "left", confidence: 0.91, reflux_duration: 1.1, description: "SFJ junction - Type 1 reflux locate SFJ precisely" },
    { sequenceNumber: 5, fromType: "N3", toType: "N3", step: "Knee-Ankle", flow: "EP", posXRatio: 0.35, posYRatio: 0.55, clipPath: "frame-005.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Medial knee - normal calf tributaries" },
    { sequenceNumber: 6, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.40, posYRatio: 0.85, clipPath: "frame-006.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Ankle area - SPJ competent" },
    { sequenceNumber: 7, fromType: "N2", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.32, posYRatio: 0.60, clipPath: "frame-007.png", legSide: "left", confidence: 0.85, reflux_duration: 0.7, description: "Medial thigh - tributary reflux detected" },
    { sequenceNumber: 8, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.30, posYRatio: 0.62, clipPath: "frame-008.png", legSide: "left", confidence: 0.84, reflux_duration: 0.9, description: "Medial knee junction - Type 2 tributary guide toward varicosities" },
    { sequenceNumber: 9, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.25, posYRatio: 0.70, clipPath: "frame-009.png", legSide: "left", confidence: 0.90, reflux_duration: 0.0, description: "Lower thigh - normal deep vein" },
    { sequenceNumber: 10, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.45, posYRatio: 0.45, clipPath: "frame-010.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Mid-thigh - competent flow" },
    { sequenceNumber: 11, fromType: "N1", toType: "N3", step: "SFJ-Knee", flow: "RP", posXRatio: 0.35, posYRatio: 0.40, clipPath: "frame-011.png", legSide: "left", confidence: 0.87, reflux_duration: 0.8, description: "Thigh complex reflux - scan downward obliquely" },
    { sequenceNumber: 12, fromType: "N3", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.33, posYRatio: 0.42, clipPath: "frame-012.png", legSide: "left", confidence: 0.86, reflux_duration: 0.9, description: "Network convergence zone Type 3" },
    { sequenceNumber: 13, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.32, posYRatio: 0.45, clipPath: "frame-013.png", legSide: "left", confidence: 0.85, reflux_duration: 1.2, description: "Type 3 complex multi-vessel confirm all pathways" },
    { sequenceNumber: 14, fromType: "N3", toType: "N3", step: "SFJ-Knee", flow: "EP", posXRatio: 0.38, posYRatio: 0.50, clipPath: "frame-014.png", legSide: "left", confidence: 0.91, reflux_duration: 0.0, description: "Lateral thigh - normal secondary branches" },
    { sequenceNumber: 15, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.42, posYRatio: 0.80, clipPath: "frame-015.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Posterior calf - SPJ normal" },
    { sequenceNumber: 16, fromType: "P", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.20, posYRatio: 0.30, clipPath: "frame-016.png", legSide: "left", confidence: 0.82, reflux_duration: 1.5, description: "Pelvic origin reflux scan external iliac probe medial-superior" },
    { sequenceNumber: 17, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.25, posYRatio: 0.35, clipPath: "frame-017.png", legSide: "left", confidence: 0.83, reflux_duration: 1.7, description: "Pelvic pathway continuation guide distally" },
    { sequenceNumber: 18, fromType: "N1", toType: "N1", step: "SFJ-Knee", flow: "EP", posXRatio: 0.30, posYRatio: 0.40, clipPath: "frame-018.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Main GSV segment - competent" },
    { sequenceNumber: 19, fromType: "N2", toType: "N2", step: "Knee-Ankle", flow: "EP", posXRatio: 0.33, posYRatio: 0.65, clipPath: "frame-019.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Calf GSV - normal" },
    { sequenceNumber: 20, fromType: "N1", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.28, posYRatio: 0.50, clipPath: "frame-020.png", legSide: "left", confidence: 0.78, reflux_duration: 1.1, description: "Perforator reflux deep-to-superficial locate Hunt/Cockett region" },
    { sequenceNumber: 21, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.42, posYRatio: 0.70, clipPath: "frame-021.png", legSide: "left", confidence: 0.79, reflux_duration: 0.9, description: "Perforator cluster Type 4 posterior calf precise location" },
    { sequenceNumber: 22, fromType: "N3", toType: "N3", step: "Knee-Ankle", flow: "EP", posXRatio: 0.38, posYRatio: 0.75, clipPath: "frame-022.png", legSide: "left", confidence: 0.91, reflux_duration: 0.0, description: "Calf tributary - normal" },
    { sequenceNumber: 23, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.40, posYRatio: 0.88, clipPath: "frame-023.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Ankle perforators - competent" },
    { sequenceNumber: 24, fromType: "P", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.22, posYRatio: 0.55, clipPath: "frame-024.png", legSide: "left", confidence: 0.80, reflux_duration: 1.4, description: "Pelvic-tributary complex scan medially toward popliteal" },
    { sequenceNumber: 25, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.30, posYRatio: 0.60, clipPath: "frame-025.png", legSide: "left", confidence: 0.81, reflux_duration: 1.2, description: "Type 5 pelvic complex pathway multi-point assessment" },
    { sequenceNumber: 26, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.28, posYRatio: 0.40, clipPath: "frame-026.png", legSide: "left", confidence: 0.82, reflux_duration: 0.8, description: "Type 5 continuation guide back to groin" },
    { sequenceNumber: 27, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.32, posYRatio: 0.68, clipPath: "frame-027.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Recovery zone - normal" },
    { sequenceNumber: 28, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.35, posYRatio: 0.42, clipPath: "frame-028.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "GSV main - stable" },
    { sequenceNumber: 29, fromType: "N1", toType: "N3", step: "SFJ-Knee", flow: "RP", posXRatio: 0.22, posYRatio: 0.35, clipPath: "frame-029.png", legSide: "left", confidence: 0.74, reflux_duration: 0.9, description: "Direct reflux pathway Type 6 locate perforator entry" },
    { sequenceNumber: 30, fromType: "N3", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.25, posYRatio: 0.38, clipPath: "frame-030.png", legSide: "left", confidence: 0.75, reflux_duration: 0.7, description: "Type 6 completion point map tributary contribution" },
    { sequenceNumber: 31, fromType: "N2", toType: "N2", step: "Knee-Ankle", flow: "EP", posXRatio: 0.30, posYRatio: 0.65, clipPath: "frame-031.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Calf stable segment" },
    { sequenceNumber: 32, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.32, posYRatio: 0.32, clipPath: "frame-032.png", legSide: "left", confidence: 0.89, reflux_duration: 1.3, description: "Recurrent reflux new pathway assess collateral formation" },
    { sequenceNumber: 33, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.30, posYRatio: 0.35, clipPath: "frame-033.png", legSide: "left", confidence: 0.87, reflux_duration: 0.6, description: "Type 1 recurrent staged approach plan first intervention" },
    { sequenceNumber: 34, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.40, posYRatio: 0.85, clipPath: "frame-034.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Ankle reassessment - SPJ intact" },
    { sequenceNumber: 35, fromType: "N2", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.65, posYRatio: 0.60, clipPath: "frame-035.png", legSide: "right", confidence: 0.83, reflux_duration: 0.8, description: "RIGHT LEG tributary reflux locate medial branches" },
    { sequenceNumber: 36, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.62, posYRatio: 0.62, clipPath: "frame-036.png", legSide: "right", confidence: 0.82, reflux_duration: 0.9, description: "RIGHT Type 2 confirm junction anatomy" },
    { sequenceNumber: 37, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.58, posYRatio: 0.70, clipPath: "frame-037.png", legSide: "right", confidence: 0.95, reflux_duration: 0.0, description: "RIGHT deep vein - competent" },
    { sequenceNumber: 38, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.60, posYRatio: 0.35, clipPath: "frame-038.png", legSide: "right", confidence: 0.88, reflux_duration: 0.7, description: "RIGHT LEG reflux at SFJ probe lateral to locate vein" },
    { sequenceNumber: 39, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.58, posYRatio: 0.38, clipPath: "frame-039.png", legSide: "right", confidence: 0.90, reflux_duration: 1.2, description: "RIGHT Type 1 SFJ ligation target confirm competence distally" },
    { sequenceNumber: 40, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.62, posYRatio: 0.85, clipPath: "frame-040.png", legSide: "right", confidence: 0.94, reflux_duration: 0.0, description: "RIGHT ankle assessment - SPJ intact" }
  ], null, 2));
  
  const [bufferInterval, setBufferInterval] = useState(0.5);
  
  const [guidance, setGuidance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisTime, setAnalysisTime] = useState(null);

  // Single mode guidance - Send clinical ultrasound data (like Task-1)
  const handleGetGuidance = async () => {
    setError(null);
    setGuidance(null);
    setLoading(true);

    try {
      const startTime = performance.now();
      
      // Parse input JSON
      let clinicalData;
      try {
        clinicalData = JSON.parse(ultrasoundData);
      } catch (e) {
        throw new Error('Invalid JSON input: ' + e.message);
      }

      const res = await axios.post('/api/probe-guidance', {
        ultrasound_data: clinicalData
      });

      const endTime = performance.now();
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));

      setGuidance(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Guidance failed');
    } finally {
      setLoading(false);
    }
  };

  // Stream mode guidance - Process clinical ultrasound data points sequentially
  const handleStreamGuidance = async () => {
    setError(null);
    setGuidance(null);
    setLoading(true);

    try {
      // Parse stream data
      let dataStream;
      try {
        dataStream = JSON.parse(streamData);
      } catch (e) {
        throw new Error('Invalid JSON input: ' + e.message);
      }

      if (!Array.isArray(dataStream)) {
        throw new Error('Stream data must be an array');
      }

      const startTime = performance.now();
      const results = {
        total_points: dataStream.length,
        processed_points: [],
        current_guidance: null,
        current_instruction: '',
        guidance_history: []
      };

      // Process each clinical data point sequentially with buffer delay
      for (let i = 0; i < dataStream.length; i++) {
        const dataPoint = dataStream[i];
        
        // Add buffer delay between points (except first)
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, bufferInterval * 1000));
        }

        try {
          // Send individual ultrasound data point to /api/probe-guidance (like Task-1)
          const res = await axios.post('/api/probe-guidance', {
            ultrasound_data: dataPoint
          });

          // Update results with the latest processing
          results.processed_points.push({
            point_number: i + 1,
            sequence_number: dataPoint.sequenceNumber,
            location: dataPoint.step,
            flow_type: dataPoint.flow,
            reflux_duration: dataPoint.reflux_duration,
            description: dataPoint.description || ''
          });

          // Update current display with latest guidance result
          results.current_guidance = res.data;
          results.current_instruction = res.data.guidance_instruction || '';
          results.guidance_history.push({
            point: i + 1,
            flow_type: dataPoint.flow,
            instruction: res.data.guidance_instruction,
            clinical_reason: res.data.clinical_reason
          });

          // Update UI in real-time after each point - CREATE NEW OBJECT to trigger React re-render
          setGuidance({ ...results });
          
        } catch (pointError) {
          console.error(`Error processing point ${i + 1}:`, pointError);
          results.processed_points.push({
            point_number: i + 1,
            sequence_number: dataPoint.sequenceNumber,
            error: pointError.response?.data?.error || pointError.message
          });
        }
      }

      const endTime = performance.now();
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));
      setGuidance({ ...results });
      
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Stream guidance failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      {/* Mode Toggle */}
      <div className="section" style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <label style={{ fontWeight: '600' }}>Analysis Mode:</label>
          <button
            onClick={() => { setMode('single'); setGuidance(null); setError(null); }}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: mode === 'single' ? '#3b82f6' : '#e5e7eb',
              color: mode === 'single' ? 'white' : '#1f2937',
              border: 'none',
              borderRadius: '0.375rem',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Single Position
          </button>
          <button
            onClick={() => { setMode('stream'); setGuidance(null); }}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: mode === 'stream' ? '#3b82f6' : '#e5e7eb',
              color: mode === 'stream' ? 'white' : '#1f2937',
              border: 'none',
              borderRadius: '0.375rem',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Stream Multiple Positions
          </button>
        </div>
      </div>

      {mode === 'single' ? (
        // ===== SINGLE MODE - Clinical Ultrasound Data Input =====
        <>
          <div className="section">
            <h2 className="section-title">📊 Single Ultrasound Point Guidance</h2>
            <div className="section-content">
              <div className="form-group">
                <label className="form-label">
                  Ultrasound Data (JSON)
                  <span className="text-muted"> - Include flow (EP/RP), step, reflux_duration for probe guidance</span>
                </label>
                <textarea
                  className="form-textarea"
                  value={ultrasoundData}
                  onChange={(e) => setUltrasoundData(e.target.value)}
                  placeholder="Enter ultrasound clinical data with flow information"
                  style={{ minHeight: '200px' }}
                />
              </div>

              <p className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
                <strong>Required fields for guidance:</strong> flow (EP/RP), step (SFJ-Knee/Knee-Ankle/SPJ-Ankle), reflux_duration, confidence
              </p>

              <div style={{ display: 'flex', gap: '1rem' }}>
                <button 
                  className="btn btn-primary" 
                  onClick={handleGetGuidance}
                  disabled={loading}
                >
                  {loading ? '🔄 Computing...' : '🎯 Get Guidance'}
                </button>
                {guidance && (
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => { setGuidance(null); setError(null); }}
                  >
                    Clear Results
                  </button>
                )}
              </div>

              {analysisTime && (
                <p className="text-muted mt-2">
                  Guidance computed in {analysisTime}s
                </p>
              )}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
              <div className="output-header">
                <h3>❌ Error</h3>
              </div>
              <div className="output-content">
                <p className="text-error">{error}</p>
              </div>
            </div>
          )}

          {/* Guidance Output */}
          {guidance && (
            <>
              {/* Clinical Assessment */}
              <div className="output-container">
                <div className="output-header">
                  <h3>🔍 Clinical Assessment</h3>
                  <span className="output-status success">✓ Complete</span>
                </div>
                <div className="output-content">
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                      <strong>Flow Type:</strong>
                      <p style={{ fontSize: '1.1rem', color: guidance.flow_type === 'RP' ? '#dc2626' : '#059669', marginTop: '0.5rem' }}>
                        {guidance.flow_type === 'RP' ? '❌ Reflux (RP)' : '✓ Normal (EP)'}
                      </p>
                    </div>
                    <div>
                      <strong>Anatomical Location:</strong>
                      <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                        {guidance.anatomical_location || 'Unknown'}
                      </p>
                    </div>
                    <div>
                      <strong>Reflux Duration:</strong>
                      <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                        {guidance.reflux_duration || 'N/A'}s
                      </p>
                    </div>
                    <div>
                      <strong>Confidence Level:</strong>
                      <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                        {guidance.confidence ? (guidance.confidence * 100).toFixed(0) : 'N/A'}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Probe Guidance Instruction */}
              <div className="output-container">
                <div className="output-header">
                  <h3>📍 Probe Guidance Instruction</h3>
                  <span className="output-status success">✓ AI Generated</span>
                </div>
                <div className="output-content">
                  <div style={{ 
                    padding: '1.5rem', 
                    backgroundColor: guidance.flow_type === 'RP' ? '#fef2f2' : '#f0fdf4', 
                    borderLeft: '4px solid ' + (guidance.flow_type === 'RP' ? '#dc2626' : '#059669'),
                    borderRadius: '0.25rem',
                    fontSize: '1.05rem',
                    lineHeight: '1.8',
                    fontWeight: '500'
                  }}>
                    {guidance.guidance_instruction || 'Generating guidance...'}
                  </div>
                </div>
              </div>

              {/* Clinical Reason */}
              {guidance.clinical_reason && (
                <div className="output-container">
                  <div className="output-header">
                    <h3>📋 Clinical Reasoning</h3>
                  </div>
                  <div className="output-content">
                    <p style={{ lineHeight: '1.8', color: '#374151' }}>
                      {guidance.clinical_reason}
                    </p>
                  </div>
                </div>
              )}

              {/* Raw Data */}
              <details style={{ marginTop: '2rem' }}>
                <summary style={{ cursor: 'pointer', fontWeight: '600', color: '#666' }}>
                  Show Complete Response
                </summary>
                <div style={{ 
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#f3f4f6',
                  borderRadius: '0.5rem',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'Monaco, monospace',
                  fontSize: '0.85rem'
                }}>
                  {JSON.stringify(guidance, null, 2)}
                </div>
              </details>
            </>
          )}

          {/* Loading */}
          {loading && (
            <div className="output-container" style={{ textAlign: 'center' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div className="spinner"></div>
              </div>
              <p>🔄 Analyzing flow data and generating probe guidance...</p>
              <p className="text-muted mt-1">Processing ultrasound findings...</p>
            </div>
          )}
        </>
      ) : (
        // ===== STREAM MODE =====
        <>
          <div className="section">
            <h2 className="section-title">📊 Stream Multiple Probe Positions</h2>
            <div className="section-content">
              <p className="text-muted">
                Process multiple ultrasound probe positions sequentially with real-time guidance updates
              </p>

              <div style={{ marginTop: '1.5rem' }}>
                <label style={{ fontWeight: '600', display: 'block', marginBottom: '0.5rem' }}>
                  Stream Data (JSON Array):
                </label>
                <textarea
                  value={streamData}
                  onChange={(e) => setStreamData(e.target.value)}
                  disabled={loading}
                  style={{
                    width: '100%',
                    height: '200px',
                    padding: '0.75rem',
                    fontFamily: 'Monaco, monospace',
                    fontSize: '0.85rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '0.375rem',
                    backgroundColor: loading ? '#f9fafb' : 'white'
                  }}
                />
              </div>

              <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <label style={{ fontWeight: '600' }}>Buffer Interval (seconds):</label>
                <input
                  type="number"
                  min="0.1"
                  max="5"
                  step="0.1"
                  value={bufferInterval}
                  onChange={(e) => setBufferInterval(parseFloat(e.target.value))}
                  disabled={loading}
                  style={{
                    width: '100px',
                    padding: '0.5rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '0.375rem'
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                <button 
                  className="btn btn-primary" 
                  onClick={handleStreamGuidance}
                  disabled={loading}
                >
                  {loading ? '🔄 Processing Stream...' : '▶️ Start Stream Analysis'}
                </button>
              </div>

              {analysisTime && !loading && (
                <p className="text-muted mt-2">
                  Stream analysis completed in {analysisTime}s
                </p>
              )}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
              <div className="output-header">
                <h3>❌ Error</h3>
              </div>
              <div className="output-content">
                <p className="text-error">{error}</p>
              </div>
            </div>
          )}

          {/* Current Guidance Output */}
          {guidance && guidance.current_guidance && (
            <>
              <div className="output-container">
                <div className="output-header">
                  <h3>📋 Current Point Guidance</h3>
                  <span className="output-status success">
                    Point {guidance.processed_points.length} of {guidance.total_points}
                  </span>
                </div>
                <div className="output-content">
                  {guidance.current_guidance && (
                    <>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div>
                          <strong>Flow Type:</strong>
                          <p style={{ fontSize: '1.1rem', color: guidance.current_guidance.flow_type === 'RP' ? '#dc2626' : '#059669', marginTop: '0.5rem' }}>
                            {guidance.current_guidance.flow_type === 'RP' ? '❌ Reflux (RP)' : '✓ Normal (EP)'}
                          </p>
                        </div>
                        <div>
                          <strong>Location:</strong>
                          <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                            {guidance.current_guidance.anatomical_location}
                          </p>
                        </div>
                        <div>
                          <strong>Reflux Duration:</strong>
                          <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                            {guidance.current_guidance.reflux_duration}s
                          </p>
                        </div>
                        <div>
                          <strong>Confidence:</strong>
                          <p style={{ fontSize: '1.1rem', color: '#1e40af', marginTop: '0.5rem' }}>
                            {(guidance.current_guidance.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                      
                      <div style={{ 
                        padding: '1rem', 
                        backgroundColor: guidance.current_guidance.flow_type === 'RP' ? '#fef2f2' : '#f0fdf4', 
                        borderLeft: '4px solid ' + (guidance.current_guidance.flow_type === 'RP' ? '#dc2626' : '#059669'),
                        borderRadius: '0.25rem',
                        marginBottom: '1rem',
                        fontSize: '1.05rem',
                        fontWeight: '500'
                      }}>
                        {guidance.current_instruction || guidance.current_guidance.guidance_instruction}
                      </div>

                      {guidance.current_guidance.clinical_reason && (
                        <div style={{ 
                          padding: '1rem', 
                          backgroundColor: '#f9fafb', 
                          borderLeft: '4px solid #6b7280',
                          borderRadius: '0.25rem',
                          fontSize: '0.95rem',
                          lineHeight: '1.6'
                        }}>
                          {guidance.current_guidance.clinical_reason}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Guidance History */}
              {guidance.guidance_history && guidance.guidance_history.length > 0 && (
                <div className="output-container">
                  <div className="output-header">
                    <h3>📈 Guidance History</h3>
                  </div>
                  <div className="output-content">
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      {guidance.guidance_history.map((item, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: '0.75rem',
                            marginBottom: '0.75rem',
                            backgroundColor: '#f3f4f6',
                            borderRadius: '0.375rem',
                            borderLeft: '3px solid ' + (item.flow_type === 'RP' ? '#dc2626' : '#059669')
                          }}
                        >
                          <div style={{ fontWeight: '600', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Point {item.point}</span>
                            <span style={{ color: item.flow_type === 'RP' ? '#dc2626' : '#059669' }}>
                              {item.flow_type === 'RP' ? '❌ Reflux' : '✓ Normal'}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.95rem', color: '#374151', marginBottom: '0.5rem' }}>
                            {item.instruction}
                          </div>
                          {item.clinical_reason && (
                            <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>
                              {item.clinical_reason}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Loading */}
          {loading && (
            <div className="output-container" style={{ textAlign: 'center' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div className="spinner"></div>
              </div>
              <p>🔄 Processing stream guidance...</p>
              {guidance && guidance.processed_positions && (
                <p className="text-muted mt-1">
                  Processed {guidance.processed_positions.length} of {guidance.total_positions} positions
                </p>
              )}
            </div>
          )}

          {/* Placeholder */}
          {!guidance && !loading && !error && (
            <div className="output-container" style={{ textAlign: 'center', opacity: 0.6 }}>
              <p>Click "Start Stream Analysis" to process multiple probe positions</p>
              <p className="text-muted mt-2">Stream data will be processed sequentially with real-time updates</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ProbeGuidance;
