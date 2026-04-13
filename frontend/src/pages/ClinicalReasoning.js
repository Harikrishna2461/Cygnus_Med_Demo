import React, { useState } from 'react';
import axios from 'axios';

// Default clip_list: 18 clips representing a Type 3 shunt assessment
// (EP N1→N2 at SFJ, EP N2→N3, RP N3→N2, RP N2→N1, RP N3→N1 with No Reflux elim test)
// This is the most clinically interesting / complex single-assessment pattern.
const REPORT_DEFAULT_CLIPS = JSON.stringify([
  {"flow":"EP","fromType":"N1","toType":"N2","posXRatio":0.30,"posYRatio":0.07,"eliminationTest":"","step":"SFJ","legSide":"left"},
  {"flow":"EP","fromType":"N2","toType":"N2","posXRatio":0.50,"posYRatio":0.20,"eliminationTest":"","step":"SFJ-Knee","legSide":"left"},
  {"flow":"EP","fromType":"N2","toType":"N3","posXRatio":0.33,"posYRatio":0.25,"eliminationTest":"","step":"SFJ-Knee","legSide":"left"},
  {"flow":"RP","fromType":"N3","toType":"N2","posXRatio":0.36,"posYRatio":0.40,"eliminationTest":"","step":"Knee","legSide":"left"},
  {"flow":"EP","fromType":"N1","toType":"N1","posXRatio":0.45,"posYRatio":0.10,"eliminationTest":"","step":"SFJ","legSide":"left"},
  {"flow":"RP","fromType":"N2","toType":"N1","posXRatio":0.32,"posYRatio":0.15,"eliminationTest":"","step":"SFJ-Knee","legSide":"left"},
  {"flow":"EP","fromType":"N3","toType":"N3","posXRatio":0.52,"posYRatio":0.68,"eliminationTest":"","step":"Knee-Ankle","legSide":"left"},
  {"flow":"RP","fromType":"N3","toType":"N2","posXRatio":0.37,"posYRatio":0.55,"eliminationTest":"","step":"Knee-Ankle","legSide":"left"},
  {"flow":"EP","fromType":"N2","toType":"N2","posXRatio":0.48,"posYRatio":0.22,"eliminationTest":"","step":"Knee","legSide":"left"},
  {"flow":"RP","fromType":"N3","toType":"N1","posXRatio":0.38,"posYRatio":0.45,"eliminationTest":"No Reflux","step":"Knee-Ankle","legSide":"left"},
  {"flow":"RP","fromType":"N2","toType":"N1","posXRatio":0.31,"posYRatio":0.18,"eliminationTest":"","step":"SFJ-Knee","legSide":"left"},
  {"flow":"EP","fromType":"N1","toType":"N1","posXRatio":0.44,"posYRatio":0.75,"eliminationTest":"","step":"Ankle","legSide":"left"},
  {"flow":"EP","fromType":"N2","toType":"N3","posXRatio":0.35,"posYRatio":0.30,"eliminationTest":"","step":"SFJ-Knee","legSide":"left"},
  {"flow":"RP","fromType":"N3","toType":"N2","posXRatio":0.38,"posYRatio":0.50,"eliminationTest":"","step":"Knee","legSide":"left"},
  {"flow":"EP","fromType":"N3","toType":"N3","posXRatio":0.50,"posYRatio":0.60,"eliminationTest":"","step":"Knee-Ankle","legSide":"left"},
  {"flow":"RP","fromType":"N2","toType":"N1","posXRatio":0.34,"posYRatio":0.28,"eliminationTest":"","step":"Knee","legSide":"left"},
  {"flow":"EP","fromType":"N1","toType":"N1","posXRatio":0.55,"posYRatio":0.95,"eliminationTest":"","step":"SPJ","legSide":"left"},
  {"flow":"EP","fromType":"N2","toType":"N2","posXRatio":0.48,"posYRatio":0.65,"eliminationTest":"","step":"Ankle","legSide":"left"}
], null, 2);

const ClinicalReasoning = () => {
  const [mode, setMode] = useState('single'); // 'single', 'stream', or 'report'

  // Post-assessment report state
  const [reportClips, setReportClips] = useState(REPORT_DEFAULT_CLIPS);
  const [reportPatientInfo, setReportPatientInfo] = useState(JSON.stringify({
    "patient_id": "PAT-001",
    "assessor": "Dr. Smith",
    "leg_side": "Left",
    "notes": ""
  }, null, 2));
  const [reportResult, setReportResult] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  
  // Single mode state - Task-1: Temporal Flow Analysis with Ligation
  const [inputData, setInputData] = useState(JSON.stringify({
    sequenceNumber: 1,
    fromType: "N1",
    toType: "N2",
    step: "SFJ-Knee",
    flow: "RP",
    posXRatio: 0.45,
    posYRatio: 0.08,
    clipPath: "video-frame-001.png",
    legSide: "left",
    confidence: 0.92,
    reflux_duration: 2.1,
    description: "GSV reflux with SFJ incompetence",
    ligation: {
      shunt_type: "Type 1 - Simple N1-N2-N1",
      primary_target: "Saphenofemoral junction (SFJ)",
      technique: "Endovenous Laser Ablation (EVLA)",
      wavelength: "1470nm",
      power: "12W",
      pullback_speed: "1cm/min",
      total_duration: "90-120 seconds",
      compression_protocol: {
        phase1: "Class III 40-50mmHg, weeks 1-2",
        phase2: "Class II 23-32mmHg, weeks 3-6"
      },
      follow_up: "Week 2 and Week 6 ultrasound",
      contraindications: "Active thrombosis, acute cellulitis, uncontrolled hypercoagulability",
      saphenous_nerve_protection: "Tumescent anesthesia with 0.1% lidocaine in 500mL NS",
      complications_to_monitor: "Paresthesia (2%), DVT risk <1%, skin burns <0.5%",
      clinical_rationale: "Type 1 (simple GSV reflux) responds excellently to SFJ ablation alone. EVLA preferred for saphenous nerve preservation compared to stripping. Success rate 95% at 1 year with saphenous-sparing approach. Avoid unnecessary tributary ablation per book guidelines."
    }
  }, null, 2));

  // Stream mode state - Mixed normal (EP) and abnormal (RP) flows, LLM generates reasoning dynamically
  const [streamData, setStreamData] = useState(JSON.stringify([
    // ── Cluster 1: Type 1 pattern (EP N1→N2 at SFJ, RP N2→N1) ──────────────
    { sequenceNumber:1,  flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.45, posYRatio:0.10, step:"SFJ",       legSide:"left",  confidence:0.95, reflux_duration:0.0, description:"Baseline", clipPath:"frame-001.png" },
    { sequenceNumber:2,  flow:"EP", fromType:"N1", toType:"N2", posXRatio:0.30, posYRatio:0.07, step:"SFJ",       legSide:"left",  confidence:0.94, reflux_duration:0.0, description:"SFJ entry point detected", clipPath:"frame-002.png" },
    { sequenceNumber:3,  flow:"RP", fromType:"N1", toType:"N2", posXRatio:0.30, posYRatio:0.07, step:"SFJ",       legSide:"left",  confidence:0.90, reflux_duration:0.8, description:"RP at SFJ — N1→N2 reflux", clipPath:"frame-003.png", ligation:{ procedure_name:"SFJ Ligation (Crossectomy)", technique:"Open groin incision, 4-0 Vicryl at SFJ", location:"Saphenofemoral junction", vessels_ligated:["GSV at SFJ","Superficial epigastric vein"], compression_post_op:"Class III 40-50mmHg wk 1-2, Class II wk 3-6" } },
    { sequenceNumber:4,  flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.32, posYRatio:0.15, step:"SFJ-Knee",  legSide:"left",  confidence:0.88, reflux_duration:1.1, description:"RP N2→N1 proximal GSV", clipPath:"frame-004.png" },
    { sequenceNumber:5,  flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.34, posYRatio:0.28, step:"SFJ-Knee",  legSide:"left",  confidence:0.86, reflux_duration:0.9, description:"RP N2→N1 mid GSV", clipPath:"frame-005.png" },
    { sequenceNumber:6,  flow:"EP", fromType:"N2", toType:"N2", posXRatio:0.50, posYRatio:0.45, step:"Knee",      legSide:"left",  confidence:0.93, reflux_duration:0.0, description:"Normal flow below knee", clipPath:"frame-006.png" },
    // ── Cluster 2: Type 3 pattern (EP N2→N3, RP N3→N2, RP N2→N1) ──────────
    { sequenceNumber:7,  flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.33, posYRatio:0.25, step:"SFJ-Knee",  legSide:"left",  confidence:0.92, reflux_duration:0.0, description:"Entry point N2→N3 tributary", clipPath:"frame-007.png" },
    { sequenceNumber:8,  flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.36, posYRatio:0.40, step:"Knee",      legSide:"left",  confidence:0.87, reflux_duration:1.2, description:"RP N3→N2 tributary reflux", clipPath:"frame-008.png", ligation:{ procedure_name:"Tributary Ligation at N2→N3", technique:"Small 2cm incision, 3-0 Vicryl at tributary junction", location:"Medial thigh tributary", vessels_ligated:["N3 tributary at N2 junction"], compression_post_op:"Class III wk 1-2, Class II wk 3-8" } },
    { sequenceNumber:9,  flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.38, posYRatio:0.55, step:"Knee-Ankle", legSide:"left",  confidence:0.85, reflux_duration:0.9, description:"RP second N3→N2 branch", clipPath:"frame-009.png" },
    { sequenceNumber:10, flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.31, posYRatio:0.18, step:"SFJ-Knee",  legSide:"left",  confidence:0.84, reflux_duration:1.3, description:"RP N2→N1 via GSV after tributary loop", clipPath:"frame-010.png" },
    { sequenceNumber:11, flow:"EP", fromType:"N3", toType:"N3", posXRatio:0.52, posYRatio:0.68, step:"Knee-Ankle", legSide:"left",  confidence:0.93, reflux_duration:0.0, description:"Normal distal tributary", clipPath:"frame-011.png" },
    { sequenceNumber:12, flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.55, posYRatio:0.95, step:"SPJ",        legSide:"left",  confidence:0.94, reflux_duration:0.0, description:"SPJ — normal", clipPath:"frame-012.png" },
    // ── Cluster 3: Type 2A pattern (EP N2→N3 only, no SFJ entry, RP N3) ────
    { sequenceNumber:13, flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.35, posYRatio:0.30, step:"SFJ-Knee",  legSide:"right", confidence:0.91, reflux_duration:0.0, description:"Right leg: entry point N2→N3", clipPath:"frame-013.png" },
    { sequenceNumber:14, flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.37, posYRatio:0.38, step:"SFJ-Knee",  legSide:"right", confidence:0.90, reflux_duration:0.0, description:"Second N2→N3 branch", clipPath:"frame-014.png" },
    { sequenceNumber:15, flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.39, posYRatio:0.48, step:"Knee",      legSide:"right", confidence:0.86, reflux_duration:1.0, description:"RP N3→N2 right leg", clipPath:"frame-015.png", ligation:{ procedure_name:"Ligate highest EP at N2→N3", technique:"Small incision at highest EP entry, 3-0 absorbable sutures", location:"Right medial knee", vessels_ligated:["Highest N2→N3 entry point"], compression_post_op:"Class II 23-32mmHg wk 1-4" } },
    { sequenceNumber:16, flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.41, posYRatio:0.58, step:"Knee-Ankle", legSide:"right", confidence:0.84, reflux_duration:0.8, description:"RP N3→N2 distal tributary", clipPath:"frame-016.png" },
    { sequenceNumber:17, flow:"EP", fromType:"N2", toType:"N2", posXRatio:0.48, posYRatio:0.22, step:"Knee",      legSide:"right", confidence:0.93, reflux_duration:0.0, description:"GSV competent — no SFJ reflux right", clipPath:"frame-017.png" },
    { sequenceNumber:18, flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.44, posYRatio:0.75, step:"Ankle",     legSide:"right", confidence:0.92, reflux_duration:0.0, description:"Normal deep system right ankle", clipPath:"frame-018.png" }
  ], null, 2));
  
  const [bufferInterval, setBufferInterval] = useState(0.5);

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [analysisTime, setAnalysisTime] = useState(null);

  // Single mode analysis
  const handleAnalyze = async () => {
    setError(null);
    setResponse(null);
    setLoading(true);
    
    try {
      const startTime = performance.now();
      
      // Parse input JSON
      let ultrasoundData;
      try {
        ultrasoundData = JSON.parse(inputData);
      } catch (e) {
        throw new Error('Invalid JSON input: ' + e.message);
      }

      // Make API call
      const res = await axios.post('/api/analyze', {
        ultrasound_data: ultrasoundData
      });

      const endTime = performance.now();
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));
      
      setResponse(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  // Stream mode analysis - PROCESS ONE POINT AT A TIME WITH UPDATES
  const handleStreamAnalysis = async () => {
    setError(null);
    setResponse(null);
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
        current_reasoning: '',
        current_assessment: '',
        current_treatment: '',
        shunt_classifications: []
      };

      // Process each data point sequentially with buffer delay
      for (let i = 0; i < dataStream.length; i++) {
        const dataPoint = dataStream[i];
        
        // Add buffer delay between points (except first)
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, bufferInterval * 1000));
        }

        try {
          // ONLY populate sections when abnormal flow (RP) detected
          const isAbnormal = dataPoint.flow === 'RP';
          
          // Generate dynamic LLM-based reasoning for this flow point
          let dynamicReasoning = dataPoint.description || '';
          if (dataPoint.flow === 'EP') {
            dynamicReasoning = 'Normal forward flow without reflux features';
          } else if (dataPoint.flow === 'RP') {
            // For abnormal flows, call LLM to generate intelligent reasoning
            try {
              const reasoningRes = await axios.post('/api/generate-flow-reasoning', {
                flow_data: dataPoint
              });
              dynamicReasoning = reasoningRes.data.reasoning || dataPoint.description || 'Abnormal reflux detected';
            } catch (reasoningErr) {
              console.warn('Failed to generate reasoning, using default:', reasoningErr);
              dynamicReasoning = dataPoint.description || 'Abnormal reflux detected';
            }
          }
          
          // Send individual data point to /api/analyze for treatment plan
          const res = await axios.post('/api/analyze', {
            ultrasound_data: dataPoint
          });

          // Update results with the latest processing
          results.processed_points.push({
            point_number: i + 1,
            sequence_number: dataPoint.sequenceNumber,
            location: dataPoint.step,
            reflux_duration: dataPoint.reflux_duration,
            description: dynamicReasoning,  // Use LLM-generated reasoning
            flow_type: dataPoint.flow,
            ligation: isAbnormal ? dataPoint.ligation || null : null
          });

          // Update current display ONLY for abnormal flow (RP)
          if (isAbnormal) {
            results.current_reasoning = res.data.reasoning || '';
            results.current_assessment = res.data.shunt_type_assessment || '';
            results.current_treatment = res.data.treatment_plan || '';
            results.shunt_classifications.push({
              point: i + 1,
              classification: res.data.shunt_classification
            });
          } else {
            // Clear sections for normal flow (EP)
            results.current_reasoning = '';
            results.current_assessment = '';
            results.current_treatment = '';
          }

          // Update UI in real-time after each point - CREATE NEW OBJECT to trigger React re-render
          setResponse({ ...results });
          
        } catch (pointError) {
          console.error(`Error processing point ${i + 1}:`, pointError);
          results.processed_points.push({
            point_number: i + 1,
            sequence_number: dataStream[i].sequenceNumber,
            error: pointError.response?.data?.error || pointError.message
          });
        }
      }

      const endTime = performance.now();
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));
      setResponse({ ...results });
      
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Stream analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setResponse(null);
    setError(null);
    setAnalysisTime(null);
  };

  // Post-assessment report handlers
  const handleGenerateReport = async () => {
    setReportError(null);
    setReportResult(null);
    setReportLoading(true);
    try {
      let clips, patientInfo;
      try { clips = JSON.parse(reportClips); } catch(e) { throw new Error('Invalid clip_list JSON: ' + e.message); }
      try { patientInfo = JSON.parse(reportPatientInfo); } catch(e) { patientInfo = {}; }
      const res = await axios.post('/api/shunt/classify-report', { clip_list: clips, patient_info: patientInfo });
      setReportResult(res.data.classification);
    } catch(err) {
      setReportError(err.response?.data?.error || err.message || 'Report generation failed');
    } finally {
      setReportLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    try {
      let clips, patientInfo;
      try { clips = JSON.parse(reportClips); } catch(e) { throw new Error('Invalid clip_list JSON'); }
      try { patientInfo = JSON.parse(reportPatientInfo); } catch(e) { patientInfo = {}; }
      const res = await axios.post(
        '/api/shunt/classify-report?format=pdf',
        { clip_list: clips, patient_info: patientInfo },
        { responseType: 'blob' }
      );
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `shunt_report_${new Date().toISOString().slice(0,10)}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch(err) {
      setReportError('PDF download failed: ' + (err.message || 'Unknown error'));
    }
  };

  return (
    <div className="page-container">
      {/* Mode Selection */}
      <div className="section">
        <h2 className="section-title">🔄 Analysis Mode</h2>
        <div className="section-content">
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="single"
                checked={mode === 'single'}
                onChange={(e) => setMode(e.target.value)}
              />
              Single Data Point
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="stream"
                checked={mode === 'stream'}
                onChange={(e) => setMode(e.target.value)}
              />
              Continuous Stream (with buffer)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="report"
                checked={mode === 'report'}
                onChange={(e) => setMode(e.target.value)}
              />
              Post-Assessment Report (PDF)
            </label>
          </div>
        </div>
      </div>

      {/* Single Mode Input */}
      {mode === 'single' && (
        <div className="section">
          <h2 className="section-title">📊 Single Ultrasound Input</h2>
          <div className="section-content">
            <div className="form-group">
              <label className="form-label">
                Ultrasound Data (JSON)
                <span className="text-muted"> - Include reflux_type and description for shunt classification</span>
              </label>
              <textarea
                className="form-textarea"
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                placeholder="Enter ultrasound JSON data"
              />
            </div>

            <p className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
              <strong>Required fields for shunt classification:</strong> reflux_type, location, reflux_duration, description
            </p>

            <div style={{ display: 'flex', gap: '1rem' }}>
              <button 
                className="btn btn-primary" 
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? '🔄 Analyzing...' : '🔍 Analyze'}
              </button>
              {response && (
                <button 
                  className="btn btn-secondary" 
                  onClick={handleClear}
                >
                  Clear Results
                </button>
              )}
            </div>

            {analysisTime && (
              <p className="text-muted mt-2">
                Analysis completed in {analysisTime}s
              </p>
            )}
          </div>
        </div>
      )}

      {/* Stream Mode Input */}
      {mode === 'stream' && (
        <div className="section">
          <h2 className="section-title">🌊 Continuous Data Stream</h2>
          <div className="section-content">
            <div className="form-group">
              <label className="form-label">
                Data Stream (JSON Array)
                <span className="text-muted"> - Each element will be processed with buffer delay</span>
              </label>
              <textarea
                className="form-textarea"
                value={streamData}
                onChange={(e) => setStreamData(e.target.value)}
                placeholder="Enter array of ultrasound data points"
                style={{ minHeight: '300px' }}
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Buffer Interval (seconds)
                <span className="text-muted"> - Delay between processing each data point</span>
              </label>
              <input
                type="number"
                min="0.2"
                max="5"
                step="0.1"
                value={bufferInterval}
                onChange={(e) => setBufferInterval(parseFloat(e.target.value))}
                style={{
                  padding: '0.5rem',
                  borderRadius: '0.25rem',
                  border: '1px solid #ddd',
                  width: '100px'
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: '1rem' }}>
              <button 
                className="btn btn-primary" 
                onClick={handleStreamAnalysis}
                disabled={loading}
              >
                {loading ? '🌊 Processing Stream...' : '🌊 Process Stream'}
              </button>
              {response && (
                <button 
                  className="btn btn-secondary" 
                  onClick={handleClear}
                >
                  Clear Results
                </button>
              )}
            </div>

            {analysisTime && (
              <p className="text-muted mt-2">
                Stream processing completed in {analysisTime}s
              </p>
            )}
          </div>
        </div>
      )}

      {/* ── Post-Assessment Report Mode ── */}
      {mode === 'report' && (
        <>
          <div className="section">
            <h2 className="section-title">📋 Post-Assessment Shunt Report</h2>
            <div className="section-content">
              <p className="text-muted" style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
                Paste 15–20 EP/RP clips from the completed duplex assessment. The LLM will classify
                the shunt type using few-shot examples from the CHIVA cheatsheet and generate a
                downloadable PDF report for clinical review.
              </p>

              <div className="form-group">
                <label className="form-label">
                  Clip List (JSON array, 15–20 points)
                </label>
                <textarea
                  className="form-textarea"
                  value={reportClips}
                  onChange={(e) => setReportClips(e.target.value)}
                  style={{ minHeight: '280px', fontFamily: 'monospace', fontSize: '0.82rem' }}
                  placeholder='[{"flow":"EP","fromType":"N1","toType":"N2","posXRatio":0.30,"posYRatio":0.07,...}]'
                />
              </div>

              <div className="form-group">
                <label className="form-label">Patient / Assessment Info (optional)</label>
                <textarea
                  className="form-textarea"
                  value={reportPatientInfo}
                  onChange={(e) => setReportPatientInfo(e.target.value)}
                  style={{ minHeight: '100px', fontFamily: 'monospace', fontSize: '0.82rem' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleGenerateReport}
                  disabled={reportLoading}
                  style={{ backgroundColor: '#C01C1C', borderColor: '#C01C1C' }}
                >
                  {reportLoading ? '🔄 Classifying...' : '🔬 Classify Shunt'}
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleDownloadPDF}
                  disabled={reportLoading}
                  style={{ backgroundColor: '#8B0000', borderColor: '#8B0000' }}
                >
                  📄 Download PDF Report
                </button>
                {reportResult && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => { setReportResult(null); setReportError(null); }}
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Report error */}
          {reportError && (
            <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
              <div className="output-header"><h3>❌ Error</h3></div>
              <div className="output-content"><p className="text-error">{reportError}</p></div>
            </div>
          )}

          {/* Report results */}
          {reportResult && (
            <>
              {/* Multi-finding overview badges */}
              {reportResult.findings && reportResult.findings.length > 1 && (
                <div className="output-container" style={{ borderLeft: '4px solid #C01C1C' }}>
                  <div className="output-header">
                    <h3>🔬 Findings Overview — {reportResult.findings.length} Legs</h3>
                    <span className="output-status success">✓ Rule-Based</span>
                  </div>
                  <div className="output-content">
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                      {reportResult.findings.map((f, i) => (
                        <div key={i} style={{ flex: 1, minWidth: 160, background: '#C01C1C', color: 'white', borderRadius: 8, padding: '0.9rem 1.1rem', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.75rem', opacity: 0.85, textTransform: 'uppercase', letterSpacing: 1 }}>{f.leg} Leg</div>
                          <div style={{ fontSize: '1.05rem', fontWeight: 700, margin: '0.3rem 0' }}>{f.shunt_type}</div>
                          <div style={{ fontSize: '0.78rem', opacity: 0.85 }}>Confidence: {((f.confidence||0)*100).toFixed(0)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Per-finding detail */}
              {(reportResult.findings || [{ ...reportResult, leg: 'Assessment', num_clips: reportResult.num_clips }]).map((f, fi) => (
                <div key={fi} className="output-container" style={{ borderLeft: '4px solid #C01C1C' }}>
                  <div className="output-header">
                    <h3>🦵 {f.leg} Leg — {f.shunt_type}</h3>
                    <span className="output-status success">Confidence: {((f.confidence||0)*100).toFixed(0)}%</span>
                  </div>
                  <div className="output-content">
                    {f.summary && <p style={{ marginBottom: '0.8rem', color: '#333', fontStyle: 'italic', fontSize: '0.92rem' }}>{f.summary}</p>}

                    {(f.needs_elim_test || f.ask_diameter || f.ask_branching) && (
                      <div style={{ background: '#FFFBEB', border: '1px solid #F59E0B', borderRadius: 4, padding: '0.5rem 0.8rem', marginBottom: '0.8rem', fontSize: '0.85rem', color: '#92400E' }}>
                        {f.needs_elim_test && <div>⚠ Elimination test required before ligation decision</div>}
                        {f.ask_diameter    && <div>ℹ Specify RP diameter at N2: Small or Large</div>}
                        {f.ask_branching   && <div>ℹ Specify N3 branching pattern</div>}
                      </div>
                    )}

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                      <div style={{ background: '#eff6ff', borderLeft: '3px solid #C01C1C', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#C01C1C', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Clinical Reasoning</div>
                        {(f.reasoning||[]).map((r,i) => <div key={i} style={{ fontSize: '0.82rem', marginBottom: '0.2rem' }}>• {String(r).replace(/^[•\-\s]+/,'')}</div>)}
                        {(!f.reasoning||f.reasoning.length===0) && <div style={{ fontSize: '0.82rem', color: '#888' }}>No pattern detected.</div>}
                      </div>
                      <div style={{ background: '#fff5f5', borderLeft: '3px solid #8B0000', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#8B0000', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Proposed Ligation</div>
                        {(f.ligation||[]).map((l,i) => <div key={i} style={{ fontSize: '0.82rem', marginBottom: '0.2rem', fontWeight: i===0?600:400 }}>• {String(l).replace(/^[•\-\s]+/,'')}</div>)}
                        {(!f.ligation||f.ligation.length===0) && <div style={{ fontSize: '0.82rem', color: '#888' }}>No ligation required.</div>}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              <div style={{ textAlign: 'center', marginTop: '1rem', marginBottom: '1.5rem' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleDownloadPDF}
                  style={{
                    backgroundColor: '#8B0000', borderColor: '#8B0000',
                    padding: '0.75rem 2rem', fontSize: '1rem'
                  }}
                >
                  📄 Download Full PDF Report
                </button>
              </div>
            </>
          )}

          {/* Loading */}
          {reportLoading && (
            <div className="output-container" style={{ textAlign: 'center' }}>
              <div className="spinner" style={{ marginBottom: '1rem' }}></div>
              <p>🔄 Running LLM shunt classification with few-shot examples...</p>
              <p className="text-muted mt-1">Analysing {(() => { try { return JSON.parse(reportClips).length; } catch { return '?'; } })()} clips</p>
            </div>
          )}
        </>
      )}

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

      {/* Single Mode Response Display */}
      {mode === 'single' && response && !response.results && (
        <>
          {/* Shunt Classification */}
          <div className="output-container">
            <div className="output-header">
              <h3>🔬 Shunt Type Classification (Task-1)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.shunt_classification ? (
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#f0fdf4', 
                  borderLeft: '4px solid #059669',
                  borderRadius: '0.25rem',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui'
                }}>
                  <div><strong>Type:</strong> {response.shunt_classification.shunt_type}</div>
                  <div><strong>Pathway:</strong> {response.shunt_classification.vein_path}</div>
                  <div><strong>Confidence:</strong> {(response.shunt_classification.confidence * 100).toFixed(0)}%</div>
                  <div style={{ marginTop: '0.5rem' }}><strong>Analysis:</strong> {response.shunt_classification.reasoning}</div>
                </div>
              ) : (
                <p className="text-muted">No classification available</p>
              )}
            </div>
          </div>

          {/* Shunt Type Assessment */}
          <div className="output-container">
            <div className="output-header">
              <h3>🔍 Shunt Type Assessment (RAG)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.shunt_type_assessment ? (
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#f0fdf4', 
                  borderLeft: '4px solid #059669',
                  borderRadius: '0.25rem',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui'
                }}>
                  {response.shunt_type_assessment}
                </div>
              ) : (
                <p className="text-muted">No assessment available</p>
              )}
            </div>
          </div>

          {/* Reasoning */}
          <div className="output-container">
            <div className="output-header">
              <h3>💭 Reasoning (Task-2)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.reasoning ? (
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#eff6ff', 
                  borderLeft: '4px solid #1e40af',
                  borderRadius: '0.25rem',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui'
                }}>
                  {response.reasoning}
                </div>
              ) : (
                <p className="text-muted">No reasoning available</p>
              )}
            </div>
          </div>

          {/* Treatment Plan */}
          <div className="output-container">
            <div className="output-header">
              <h3>💊 Treatment Plan</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.treatment_plan ? (
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#fef3c7', 
                  borderLeft: '4px solid #d97706',
                  borderRadius: '0.25rem',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui'
                }}>
                  {response.treatment_plan}
                </div>
              ) : (
                <p className="text-muted">No treatment plan available</p>
              )}
            </div>
          </div>

          {/* Raw Response */}
          <details style={{ marginTop: '2rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '600', color: '#666' }}>
              Show Raw LLM Response
            </summary>
            <div style={{ 
              marginTop: '1rem',
              padding: '1rem',
              backgroundColor: '#f3f4f6',
              borderRadius: '0.5rem',
              whiteSpace: 'pre-wrap',
              fontFamily: 'Monaco, monospace',
              fontSize: '0.85rem',
              overflow: 'auto',
              maxHeight: '400px'
            }}>
              {response.raw_response}
            </div>
          </details>
        </>
      )}

      {/* Stream Mode Response Display - LIVE UPDATES */}
      {mode === 'stream' && response && response.processed_points && (
        <>
          <div className="output-container">
            <div className="output-header">
              <h3>📊 Stream Processing - Live Results</h3>
              <span className="output-status success">✓ Processing: {response.processed_points.length}/{response.total_points} points</span>
            </div>
            <div className="output-content">
              <p><strong>Total Data Points:</strong> {response.total_points}</p>
              <p><strong>Processed:</strong> {response.processed_points.length} points</p>
              <p><strong>Buffer Interval:</strong> {bufferInterval}s between points</p>
              {analysisTime && <p><strong>Total Time:</strong> {analysisTime}s</p>}
            </div>
          </div>

          {/* Current Latest Result (LATEST POINT) */}
          <div className="output-container">
            <div className="output-header">
              <h3>⏱️ Latest Result</h3>
              <span className="output-status success">✓ Active</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && (
                <>
                  <div style={{ 
                    padding: '0.75rem', 
                    backgroundColor: '#f0fdf4', 
                    borderLeft: '4px solid #059669',
                    borderRadius: '0.25rem',
                    marginBottom: '1rem',
                    fontSize: '0.9rem'
                  }}>
                    <div><strong>Point #</strong> {response.processed_points[response.processed_points.length - 1].sequence_number}</div>
                    <div><strong>Location:</strong> {response.processed_points[response.processed_points.length - 1].location}</div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Shunt Type Assessment - LATEST */}
          <div className="output-container">
            <div className="output-header">
              <h3>🔍 Shunt Type Assessment (Latest)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_assessment ? (
                  <div style={{ 
                    padding: '1rem', 
                    backgroundColor: '#f0fdf4', 
                    borderLeft: '4px solid #059669',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_assessment}
                  </div>
                ) : (
                  <p className="text-muted">Processing...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - sections remain empty</p>
              )}
            </div>
          </div>

          {/* Reasoning - LATEST */}
          <div className="output-container">
            <div className="output-header">
              <h3>💭 Reasoning (Latest Point)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_reasoning ? (
                  <div style={{ 
                    padding: '1rem', 
                    backgroundColor: '#eff6ff', 
                    borderLeft: '4px solid #1e40af',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_reasoning}
                  </div>
                ) : (
                  <p className="text-muted">Processing...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - sections remain empty</p>
              )}
            </div>
          </div>

          {/* Ligation - CHIVA SURGICAL PROCEDURE */}
          <div className="output-container">
            <div className="output-header">
              <h3>⚕️ Ligation (Latest Point)</h3>
              <span className="output-status success">✓ CHIVA Surgical Strategy</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_treatment ? (
                  <div style={{ 
                    padding: '1rem', 
                    backgroundColor: '#fdf2f8', 
                    borderLeft: '4px solid #8b5cf6',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_treatment}
                  </div>
                ) : (
                  <p className="text-muted">Processing surgical ligation procedures...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - ligation not required</p>
              )}
            </div>
          </div>

          {/* All Processed Points Summary */}
          <details style={{ marginTop: '2rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '600', color: '#666' }}>
              📋 Show All {response.processed_points.length} Processed Data Points
            </summary>
            <div style={{ marginTop: '1rem' }}>
              {response.processed_points.map((point, idx) => (
                <div key={idx} style={{
                  padding: '1rem',
                  marginBottom: '0.75rem',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.375rem',
                  fontSize: '0.9rem'
                }}>
                  <div><strong>Point #{point.point_number}:</strong> {point.timestamp}</div>
                  <div><strong>Reflux Type:</strong> {point.reflux_type}</div>
                  <div><strong>Location:</strong> {point.location}</div>
                  <div><strong>Duration:</strong> {point.reflux_duration}s</div>
                  <div><strong>Description:</strong> {point.description}</div>
                  {point.error && <div style={{ color: '#dc2626', marginTop: '0.5rem' }}><strong>Error:</strong> {point.error}</div>}
                </div>
              ))}
            </div>
          </details>
        </>
      )}

      {/* Placeholder when no response */}
      {!response && !error && !loading && (
        <div className="output-container" style={{ textAlign: 'center', opacity: 0.6 }}>
          {mode === 'single' ? (
            <>
              <p>Enter ultrasound data with reflux_type and description, then click "Analyze"</p>
              <p className="text-muted mt-2">Task-1: Shunt classification → Task-2: Clinical reasoning & treatment</p>
            </>
          ) : (
            <>
              <p>Enter a JSON array of data points and click "Process Stream"</p>
              <p className="text-muted mt-2">Each data point will be processed with the specified buffer interval (0.2-3s, default 0.5s)</p>
            </>
          )}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="output-container" style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '1rem' }}>
            <div className="spinner"></div>
          </div>
          {mode === 'single' ? (
            <>
              <p>🔄 Analyzing single data point...</p>
              <p className="text-muted mt-1">Task-1: Classifying shunt type | Task-2: Retrieving medical context...</p>
            </>
          ) : (
            <>
              <p>🌊 Processing continuous data stream...</p>
              <p className="text-muted mt-1">Applying {bufferInterval}s buffer between each data point</p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ClinicalReasoning;
