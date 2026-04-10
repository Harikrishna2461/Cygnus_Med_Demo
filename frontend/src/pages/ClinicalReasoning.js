import React, { useState } from 'react';
import axios from 'axios';

const ClinicalReasoning = () => {
  const [mode, setMode] = useState('single'); // 'single' or 'stream'
  
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
    { sequenceNumber: 1, fromType: "N1", toType: "N1", step: "SFJ-Knee", flow: "EP", posXRatio: 0.45, posYRatio: 0.10, clipPath: "frame-001.png", legSide: "left", confidence: 0.95, reflux_duration: 0.0, description: "Setup" },
    { sequenceNumber: 2, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.50, posYRatio: 0.20, clipPath: "frame-002.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Setup" },
    { sequenceNumber: 3, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.45, posYRatio: 0.08, clipPath: "frame-003.png", legSide: "left", confidence: 0.88, reflux_duration: 0.6, description: "Analyzing..." },
    { sequenceNumber: 4, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.48, posYRatio: 0.12, clipPath: "frame-004.png", legSide: "left", confidence: 0.91, reflux_duration: 1.1, description: "Analyzing...", "ligation": { "procedure_name": "SFJ Ligation with Crossectomy", "description": "Surgical tying of saphenofemoral junction to close Type 1 reflux pathway", "technique": "Open surgical exposure with 4-0 absorbable sutures at SFJ", "location": "Saphenofemoral junction in groin crease", "vessels_ligated": ["Great Saphenous Vein at SFJ", "Superficial epigastric vein", "Superficial circumflex iliac vein"], "compression_post_op": "Class III 40-50mmHg weeks 1-2, then Class II 23-32mmHg weeks 3-6" } },
    { sequenceNumber: 5, fromType: "N3", toType: "N3", step: "Knee-Ankle", flow: "EP", posXRatio: 0.52, posYRatio: 0.68, clipPath: "frame-005.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 6, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.55, posYRatio: 0.95, clipPath: "frame-006.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 7, fromType: "N2", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.50, posYRatio: 0.62, clipPath: "frame-007.png", legSide: "left", confidence: 0.85, reflux_duration: 0.7, description: "Analyzing..." },
    { sequenceNumber: 8, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.48, posYRatio: 0.68, clipPath: "frame-008.png", legSide: "left", confidence: 0.84, reflux_duration: 0.9, description: "Analyzing...", "ligation": { "procedure_name": "Tributary Ligation - Isolated", "description": "Ligation of incompetent tributary branches while preserving competent GSV", "technique": "Small 2-3cm incisions above medial knee with 3-0 absorbable sutures", "location": "Medial knee above tributary-GSV junction", "vessels_ligated": ["Medial tributary branches from GSV", "Reverse flow perforators"], "suture_material": "Polyglactin 910 (Vicryl) 3-0", "GSV_status": "Competent - preserved for potential future use", "compression_post_op": "Class III 40-50mmHg weeks 1-2, then Class II weeks 3-4" } },
    { sequenceNumber: 9, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.45, posYRatio: 0.75, clipPath: "frame-009.png", legSide: "left", confidence: 0.90, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 10, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.50, posYRatio: 0.22, clipPath: "frame-010.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 11, fromType: "N1", toType: "N3", step: "SFJ-Knee", flow: "RP", posXRatio: 0.42, posYRatio: 0.15, clipPath: "frame-011.png", legSide: "left", confidence: 0.87, reflux_duration: 0.8, description: "Analyzing..." },
    { sequenceNumber: 12, fromType: "N3", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.38, posYRatio: 0.18, clipPath: "frame-012.png", legSide: "left", confidence: 0.86, reflux_duration: 0.9, description: "Analyzing..." },
    { sequenceNumber: 13, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.52, posYRatio: 0.25, clipPath: "frame-013.png", legSide: "left", confidence: 0.85, reflux_duration: 1.2, description: "Analyzing...", "ligation": { "procedure_name": "Dual Ligation - SFJ with Tributary", "description": "Two-stage ligation for Type 3 complex multi-vessel reflux", "technique": "Primary: Groin incision 4-0 sutures SFJ; Secondary: Medial knee 3-0 sutures tributary junction", "location_primary": "Groin saphenofemoral junction", "location_secondary": "Medial knee tributary convergence", "vessels_ligated_primary": ["Great Saphenous Vein at SFJ", "All SFJ tributaries"], "vessels_ligated_secondary": ["Major tributary branches"], "timing": "Primary immediate, secondary 2-4 weeks later", "compression_post_op": "Class III 40mmHg weeks 1-3, then Class II weeks 4-8" } },
    { sequenceNumber: 14, fromType: "N3", toType: "N3", step: "SFJ-Knee", flow: "EP", posXRatio: 0.45, posYRatio: 0.35, clipPath: "frame-014.png", legSide: "left", confidence: 0.91, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 15, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.55, posYRatio: 0.90, clipPath: "frame-015.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 16, fromType: "P", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.40, posYRatio: 0.08, clipPath: "frame-016.png", legSide: "left", confidence: 0.82, reflux_duration: 1.5, description: "Analyzing..." },
    { sequenceNumber: 17, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.45, posYRatio: 0.20, clipPath: "frame-017.png", legSide: "left", confidence: 0.83, reflux_duration: 1.7, description: "Analyzing...", "ligation": null },
    { sequenceNumber: 18, fromType: "N1", toType: "N1", step: "SFJ-Knee", flow: "EP", posXRatio: 0.45, posYRatio: 0.12, clipPath: "frame-018.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 19, fromType: "N2", toType: "N2", step: "Knee-Ankle", flow: "EP", posXRatio: 0.48, posYRatio: 0.65, clipPath: "frame-019.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 20, fromType: "N1", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.35, posYRatio: 0.32, clipPath: "frame-020.png", legSide: "left", confidence: 0.78, reflux_duration: 1.1, description: "Analyzing..." },
    { sequenceNumber: 21, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.55, posYRatio: 0.75, clipPath: "frame-021.png", legSide: "left", confidence: 0.79, reflux_duration: 0.9, description: "Analyzing...", "ligation": { "procedure_name": "Perforator Ligation - SEPS", "description": "Subfascial endoscopic closure of perforator reflux pathway", "technique": "Subfascial endoscopic approach via calf incision, titanium clips or absorbable sutures", "location": "Calf perforators (Cockett/Hunt region)", "vessels_ligated": ["Incompetent perforating veins N1-N3 connection"], "suture_material": "Titanium clips or 2-0/3-0 absorbable sutures", "compression_post_op": "Class III 40-50mmHg weeks 1-2, then Class II weeks 3-6" } },
    { sequenceNumber: 22, fromType: "N3", toType: "N3", step: "Knee-Ankle", flow: "EP", posXRatio: 0.52, posYRatio: 0.72, clipPath: "frame-022.png", legSide: "left", confidence: 0.91, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 23, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.55, posYRatio: 0.88, clipPath: "frame-023.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 24, fromType: "P", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.42, posYRatio: 0.60, clipPath: "frame-024.png", legSide: "left", confidence: 0.80, reflux_duration: 1.4, description: "Analyzing..." },
    { sequenceNumber: 25, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.50, posYRatio: 0.70, clipPath: "frame-025.png", legSide: "left", confidence: 0.81, reflux_duration: 1.2, description: "Analyzing...", "ligation": null },
    { sequenceNumber: 26, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.48, posYRatio: 0.18, clipPath: "frame-026.png", legSide: "left", confidence: 0.82, reflux_duration: 0.8, description: "Analyzing..." },
    { sequenceNumber: 27, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.45, posYRatio: 0.72, clipPath: "frame-027.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 28, fromType: "N2", toType: "N2", step: "SFJ-Knee", flow: "EP", posXRatio: 0.50, posYRatio: 0.22, clipPath: "frame-028.png", legSide: "left", confidence: 0.93, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 29, fromType: "N1", toType: "N3", step: "SFJ-Knee", flow: "RP", posXRatio: 0.32, posYRatio: 0.30, clipPath: "frame-029.png", legSide: "left", confidence: 0.74, reflux_duration: 0.9, description: "Analyzing..." },
    { sequenceNumber: 30, fromType: "N3", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.52, posYRatio: 0.28, clipPath: "frame-030.png", legSide: "left", confidence: 0.75, reflux_duration: 0.7, description: "Analyzing...", "ligation": null },
    { sequenceNumber: 31, fromType: "N2", toType: "N2", step: "Knee-Ankle", flow: "EP", posXRatio: 0.48, posYRatio: 0.65, clipPath: "frame-031.png", legSide: "left", confidence: 0.92, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 32, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.45, posYRatio: 0.10, clipPath: "frame-032.png", legSide: "left", confidence: 0.89, reflux_duration: 1.3, description: "Analyzing..." },
    { sequenceNumber: 33, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.48, posYRatio: 0.12, clipPath: "frame-033.png", legSide: "left", confidence: 0.87, reflux_duration: 0.6, description: "Analyzing...", "ligation": { "procedure_name": "Dual Ligation - Staged Approach", "description": "SFJ ligation initially, then staged tributary ligation after 4-6 weeks", "technique": "Primary: SFJ crossectomy 4-0 sutures; Secondary: Tributary incisions 3-0 sutures", "location_primary": "Groin saphenofemoral junction", "location_secondary": "Medial knee after primary healing", "timing": "Primary immediate, secondary 4-6 weeks post-primary", "compression_post_op": "Class III 40mmHg primary weeks 1-2, then Class II weeks 3-6; Secondary phase: Class III week 1, Class II weeks 2-4" } },
    { sequenceNumber: 34, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.55, posYRatio: 0.90, clipPath: "frame-034.png", legSide: "left", confidence: 0.94, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 35, fromType: "N2", toType: "N3", step: "Knee-Ankle", flow: "RP", posXRatio: 0.50, posYRatio: 0.62, clipPath: "frame-035.png", legSide: "right", confidence: 0.83, reflux_duration: 0.8, description: "Analyzing..." },
    { sequenceNumber: 36, fromType: "N3", toType: "N2", step: "Knee-Ankle", flow: "RP", posXRatio: 0.48, posYRatio: 0.68, clipPath: "frame-036.png", legSide: "right", confidence: 0.82, reflux_duration: 0.9, description: "Analyzing...", "ligation": null },
    { sequenceNumber: 37, fromType: "N1", toType: "N1", step: "Knee-Ankle", flow: "EP", posXRatio: 0.45, posYRatio: 0.75, clipPath: "frame-037.png", legSide: "right", confidence: 0.95, reflux_duration: 0.0, description: "Separator" },
    { sequenceNumber: 38, fromType: "N1", toType: "N2", step: "SFJ-Knee", flow: "RP", posXRatio: 0.45, posYRatio: 0.08, clipPath: "frame-038.png", legSide: "right", confidence: 0.88, reflux_duration: 0.7, description: "Analyzing..." },
    { sequenceNumber: 39, fromType: "N2", toType: "N1", step: "SFJ-Knee", flow: "RP", posXRatio: 0.48, posYRatio: 0.12, clipPath: "frame-039.png", legSide: "right", confidence: 0.90, reflux_duration: 1.2, description: "Analyzing...", "ligation": { "procedure_name": "SFJ Ligation with Crossectomy", "description": "Surgical tying of right saphenofemoral junction for Type 1 reflux", "technique": "Open surgical approach with 4-0 absorbable sutures", "location": "Right groin saphenofemoral junction", "vessels_ligated": ["Right Great Saphenous Vein at SFJ", "Right superficial tributaries"], "suture_material": "Polyglactin 910 (Vicryl) 4-0", "compression_post_op": "Class III 40-50mmHg weeks 1-2, then Class II weeks 3-6" } },
    { sequenceNumber: 40, fromType: "N1", toType: "N1", step: "SPJ-Ankle", flow: "EP", posXRatio: 0.55, posYRatio: 0.92, clipPath: "frame-040.png", legSide: "right", confidence: 0.94, reflux_duration: 0.0, description: "Separator" }
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
