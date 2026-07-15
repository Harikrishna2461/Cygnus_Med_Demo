import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

const ProbeGuidance = () => {
  const { sonographerId } = useParams();
  const navigate = useNavigate();

  // Sonographer profile & session history
  const [sonographer, setSonographer] = useState(null);
  const [pastSessions, setPastSessions] = useState([]);
  const [showSessions, setShowSessions] = useState(false);

  // Generate REALISTIC mock stream data based on CHIVA shunt types from RAG database
  const generateMockStreamData = () => {
    const zones = {
      left: {
        'SFJ-Knee': { xMin: 0.4985, xMax: 0.909, yMin: 0, yMax: 0.5497 },
        'Knee-Ankle': { xMin: 0.7081, xMax: 0.91, yMin: 0.5497, yMax: 1 },
        'SPJ-Ankle': { xMin: 0.588, xMax: 0.714, yMin: 0.5497, yMax: 1 }
      },
      right: {
        'SFJ-Knee': { xMin: 0.0931, xMax: 0.475, yMin: 0, yMax: 0.5497 },
        'Knee-Ankle': { xMin: 0.105, xMax: 0.2947, yMin: 0.5497, yMax: 1 },
        'SPJ-Ankle': { xMin: 0.2827, xMax: 0.4386, yMin: 0.5497, yMax: 1 }
      }
    };

    // CHIVA Shunt Type patterns from medical knowledge base
    const shuntPatterns = [
      // Type 1: N1-N2-N1 (GSV reflux) - Most common
      {
        type: 'Type 1 (N1-N2-N1)',
        sequence: [
          { fromType: 'N1', toType: 'N1', flow: 'EP', zone: 'SFJ-Knee', desc: 'Femoral vein competent at inguinal ligament' },
          { fromType: 'N1', toType: 'N2', flow: 'RP', zone: 'SFJ-Knee', desc: 'SFJ incompetence - N1→N2 reflux detected', duration: 0.8 },
          { fromType: 'N2', toType: 'N1', flow: 'RP', zone: 'SFJ-Knee', desc: 'GSV reflux pathway confirmed proximal thigh', duration: 1.0 },
          { fromType: 'N2', toType: 'N2', flow: 'EP', zone: 'Knee-Ankle', desc: 'GSV competent in calf region' }
        ]
      },
      // Type 2: N2-N3 (Tributary reflux)
      {
        type: 'Type 2 (N2-N3)',
        sequence: [
          { fromType: 'N2', toType: 'N2', flow: 'EP', zone: 'SFJ-Knee', desc: 'GSV trunk normal at groin' },
          { fromType: 'N2', toType: 'N3', flow: 'RP', zone: 'Knee-Ankle', desc: 'Saphenous to tributary reflux in calf', duration: 0.7 },
          { fromType: 'N3', toType: 'N2', flow: 'RP', zone: 'Knee-Ankle', desc: 'Tributary incompetence at Cockett zone', duration: 0.6 },
          { fromType: 'N3', toType: 'N3', flow: 'EP', zone: 'Knee-Ankle', desc: 'Deep calf veins competent' }
        ]
      },
      // Type 3: N1-N2-N3-N1 (Complex multi-vessel)
      {
        type: 'Type 3 (N1-N2-N3-N1)',
        sequence: [
          { fromType: 'N1', toType: 'N2', flow: 'RP', zone: 'SFJ-Knee', desc: 'SFJ reflux N1→N2 at groin', duration: 0.9 },
          { fromType: 'N2', toType: 'N3', flow: 'RP', zone: 'Knee-Ankle', desc: 'GSV to tributary loop detected', duration: 0.8 },
          { fromType: 'N3', toType: 'N1', flow: 'RP', zone: 'Knee-Ankle', desc: 'Tributary back to deep vein pathway', duration: 0.7 },
          { fromType: 'N1', toType: 'N1', flow: 'EP', zone: 'SPJ-Ankle', desc: 'Popliteal vein competent' }
        ]
      },
      // Type 4 Pelvic: P-N2-N1 (Pelvic origin)
      {
        type: 'Type 4 Pelvic (P-N2-N1)',
        sequence: [
          { fromType: 'P', toType: 'N2', flow: 'RP', zone: 'SFJ-Knee', desc: 'Pelvic origin reflux detected superomedial', duration: 1.4 },
          { fromType: 'N2', toType: 'N1', flow: 'RP', zone: 'SFJ-Knee', desc: 'Saphenous axis reflux from pelvic source', duration: 1.2 },
          { fromType: 'N1', toType: 'N1', flow: 'EP', zone: 'Knee-Ankle', desc: 'Calf deep veins competent' }
        ]
      },
      // Type 4 Perforator: N1-N3-N2-N1 (Perforator involvement)
      {
        type: 'Type 4 Perforator (N1-N3-N2)',
        sequence: [
          { fromType: 'N1', toType: 'N1', flow: 'EP', zone: 'SFJ-Knee', desc: 'Femoral vein normal at Hunterian canal' },
          { fromType: 'N1', toType: 'N3', flow: 'RP', zone: 'Knee-Ankle', desc: 'Perforator reflux from deep vein', duration: 0.8 },
          { fromType: 'N3', toType: 'N2', flow: 'RP', zone: 'Knee-Ankle', desc: 'Perforator to saphenous tributary pathway', duration: 0.7 },
          { fromType: 'N2', toType: 'N2', flow: 'EP', zone: 'Knee-Ankle', desc: 'Saphenous axis itself competent' }
        ]
      },
      // Type 5 Pelvic: P-N3-N2-N1 (Complex pelvic)
      {
        type: 'Type 5 Pelvic (P-N3-N2)',
        sequence: [
          { fromType: 'P', toType: 'N3', flow: 'RP', zone: 'Knee-Ankle', desc: 'Pelvic to tributary pathway detected', duration: 1.5 },
          { fromType: 'N3', toType: 'N2', flow: 'RP', zone: 'Knee-Ankle', desc: 'Tributary to saphenous connection', duration: 1.2 },
          { fromType: 'N2', toType: 'N1', flow: 'RP', zone: 'SFJ-Knee', desc: 'Saphenous reflux extending proximally', duration: 1.0 },
          { fromType: 'N1', toType: 'N1', flow: 'EP', zone: 'SPJ-Ankle', desc: 'Popliteal vein assessment normal' }
        ]
      },
      // Type 6: N1-N3-N2 (Direct perforator)
      {
        type: 'Type 6 (N1-N3-N2)',
        sequence: [
          { fromType: 'N1', toType: 'N3', flow: 'RP', zone: 'SFJ-Knee', desc: 'Direct perforator reflux from femoral', duration: 0.9 },
          { fromType: 'N3', toType: 'N2', flow: 'RP', zone: 'Knee-Ankle', desc: 'Tributary to saphenous axis pathway', duration: 0.8 },
          { fromType: 'N2', toType: 'N2', flow: 'EP', zone: 'SPJ-Ankle', desc: 'Saphenous axis competent in ankle region' }
        ]
      }
    ];

    const randomNum = (min, max) => Math.random() * (max - min) + min;
    const randomChoice = (arr) => arr[Math.floor(Math.random() * arr.length)];

    // Pick a realistic shunt type
    const selectedShunt = randomChoice(shuntPatterns);
    const legSide = randomChoice(['left', 'right']);
    const data = [];
    let seqNum = 1;

    // Add initial scanning context point (normal flow assessment)
    data.push({
      sequenceNumber: seqNum++,
      fromType: 'N1',
      toType: 'N1',
      step: 'SFJ-Knee',
      flow: 'EP',
      posXRatio: randomNum(zones[legSide]['SFJ-Knee'].xMin, zones[legSide]['SFJ-Knee'].xMax),
      posYRatio: randomNum(zones[legSide]['SFJ-Knee'].yMin, zones[legSide]['SFJ-Knee'].yMax),
      clipPath: `frame-001.png`,
      legSide,
      confidence: 0.95,
      reflux_duration: 0.0,
      description: `Initial assessment - ${legSide} leg femoral vein screening`
    });

    // Add pattern-specific sequence
    for (const point of selectedShunt.sequence) {
      const zoneData = zones[legSide][point.zone];
      data.push({
        sequenceNumber: seqNum++,
        fromType: point.fromType,
        toType: point.toType,
        step: point.zone,
        flow: point.flow,
        posXRatio: randomNum(zoneData.xMin, zoneData.xMax),
        posYRatio: randomNum(zoneData.yMin, zoneData.yMax),
        clipPath: `frame-${String(seqNum).padStart(3, '0')}.png`,
        legSide,
        confidence: parseFloat(randomNum(0.8, 0.98).toFixed(2)),
        reflux_duration: point.duration || 0.0,
        description: `[${selectedShunt.type}] ${point.desc}`
      });
    }

    // Add final assessment point
    const finalZone = randomChoice(['Knee-Ankle', 'SPJ-Ankle']);
    const finalZoneData = zones[legSide][finalZone];
    data.push({
      sequenceNumber: seqNum++,
      fromType: 'N1',
      toType: 'N1',
      step: finalZone,
      flow: 'EP',
      posXRatio: randomNum(finalZoneData.xMin, finalZoneData.xMax),
      posYRatio: randomNum(finalZoneData.yMin, finalZoneData.yMax),
      clipPath: `frame-final.png`,
      legSide,
      confidence: 0.93,
      reflux_duration: 0.0,
      description: `Final assessment - Complete evaluation in ${finalZone} zone`
    });

    return JSON.stringify(data, null, 2);
  };

  useEffect(() => {
    if (!sonographerId) return;
    axios.get(`/api/sonographers/${sonographerId}`)
      .then(r => setSonographer(r.data))
      .catch(() => {});
    axios.get(`/api/sonographers/${sonographerId}/sessions?limit=100`)
      .then(r => setPastSessions(r.data))
      .catch(() => {});
  }, [sonographerId]);

  const [mode, setMode] = useState('single'); // 'single', 'stream', or 'analyze-session'
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionComparison, setSessionComparison] = useState(null);
  
  // Single mode state - Use same format as Task-1
  // Left leg SFJ-Knee zone: X 0.4985–0.909, Y 0.10–0.50
  const [ultrasoundData, setUltrasoundData] = useState(JSON.stringify({
    sequenceNumber: 1,
    fromType: "N1",
    toType: "N2",
    step: "SFJ-Knee",
    flow: "RP",
    posXRatio: 0.63,
    posYRatio: 0.18,
    clipPath: "video-frame-001.png",
    legSide: "left",
    confidence: 0.92,
    reflux_duration: 1.1,
    description: "Left SFJ-Knee zone — GSV reflux detected at Hunterian canal level"
  }, null, 2));
  
  // Stream mode state — anatomically grounded posX/posY per the CHIVA scanning coordinate system:
  // (0,0)=top-left, (1,1)=bottom-right.
  // Right leg (left side of image): SFJ-Knee X 0.0931-0.475 Y 0-0.5497 | Knee-Ankle X 0.105-0.2947 Y 0.5497-1 | SPJ-Ankle X 0.2827-0.4386 Y 0.5497-1
  // Left leg (right side of image): SFJ-Knee X 0.4985-0.909 Y 0-0.5497 | Knee-Ankle X 0.7081-0.91 Y 0.5497-1  | SPJ-Ankle X 0.588-0.714 Y 0.5497-1


  const [streamData, setStreamData] = useState(generateMockStreamData());

  // Regenerate stream data when component mounts (page refresh)
  useEffect(() => {
    setStreamData(generateMockStreamData());
  }, []);

  
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
        ultrasound_data: clinicalData,
        ...(sonographerId && { sonographer_id: sonographerId }),
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

  // Analyze previous session - Show historical patterns and trends
  const handleAnalyzePreviousSession = async () => {
    if (!selectedSession) {
      setError('Please select a session to analyze');
      return;
    }

    setError(null);
    setSessionComparison(null);
    setLoading(true);

    try {
      const startTime = performance.now();

      // Get sonographer context + session data
      const comparison = {
        session_id: selectedSession.session_id,
        session_date: selectedSession.session_date,
        total_points: selectedSession.total_points,
        reflux_count: selectedSession.reflux_count,
        summary: selectedSession.session_summary,
        guidance_history: selectedSession.guidance_history || [],
      };

      // Analyze patterns in the session
      const refluxPoints = comparison.guidance_history.filter(h => h.flow_type === 'RP');
      const normalPoints = comparison.guidance_history.filter(h => h.flow_type === 'EP');

      comparison.analysis = {
        total_guidance_points: comparison.guidance_history.length,
        reflux_percentage: ((refluxPoints.length / comparison.guidance_history.length) * 100).toFixed(1),
        normal_percentage: ((normalPoints.length / comparison.guidance_history.length) * 100).toFixed(1),
        reflux_locations: refluxPoints.map(p => p.instruction).slice(0, 3),
        normal_areas: normalPoints.map(p => p.instruction).slice(0, 3),
        session_duration_min: (comparison.guidance_history.length * 0.5).toFixed(1),
      };

      const endTime = performance.now();
      comparison.analysis_time_ms = endTime - startTime;

      setSessionComparison(comparison);
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Session analysis failed');
    } finally {
      setLoading(false);
    }
  };

  // Stream mode guidance - Send clinical data stream with buffer delay
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
          const res = await axios.post('/api/probe-guidance', {
            ultrasound_data: dataPoint,
            ...(sonographerId && { sonographer_id: sonographerId }),
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

      // Persist session to sonographer history
      if (sonographerId && results.guidance_history.length > 0) {
        const refluxCount = results.guidance_history.filter(h => h.flow_type === 'RP').length;
        const summary = `${results.guidance_history.length} positions scanned, ${refluxCount} reflux detections.`;
        try {
          await axios.post(`/api/sonographers/${sonographerId}/sessions`, {
            mode: 'stream',
            guidance_history: results.guidance_history,
            session_summary: summary,
          });
          // Refresh session list
          const r = await axios.get(`/api/sonographers/${sonographerId}/sessions?limit=5`);
          setPastSessions(r.data);
        } catch (_) { /* session save is non-critical */ }
      }

    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Stream guidance failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">

      {/* Sonographer header */}
      {sonographer && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '1rem',
          background: 'white', border: '1px solid #e5e7eb',
          borderRadius: '0.75rem', padding: '1rem 1.25rem', marginBottom: '1.25rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        }}>
          <button
            onClick={() => navigate('/probe')}
            style={{
              background: 'none', border: '1px solid #d1d5db', borderRadius: '0.375rem',
              padding: '0.35rem 0.75rem', cursor: 'pointer', fontSize: '0.85rem', color: '#374151',
            }}
          >
            ← Back
          </button>
          <div style={{
            width: '40px', height: '40px', borderRadius: '50%',
            backgroundColor: sonographer.avatar_color || '#3b82f6',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: 'white', fontWeight: '700', flexShrink: 0,
          }}>
            {sonographer.name.split(' ').map(n => n[0]).slice(0, 2).join('')}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: '700', fontSize: '1rem', color: '#111827' }}>{sonographer.name}</div>
            <div style={{ fontSize: '0.8rem', color: '#6b7280' }}>{sonographer.title} · {sonographer.experience_years} yrs</div>
          </div>
          <button
            onClick={() => setShowSessions(v => !v)}
            style={{
              background: '#eff6ff', border: '1px solid #bfdbfe',
              borderRadius: '0.375rem', padding: '0.35rem 0.85rem',
              cursor: 'pointer', fontSize: '0.85rem', color: '#1d4ed8', fontWeight: '600',
            }}
          >
            {showSessions ? 'Hide' : 'Past Sessions'} ({pastSessions.length})
          </button>
        </div>
      )}

      {/* Past sessions panel */}
      {showSessions && pastSessions.length > 0 && (
        <div className="section" style={{ marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '700', marginBottom: '0.75rem', color: '#1e40af' }}>
            📋 Session History — {sonographer?.name}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {pastSessions.map((s, i) => (
              <div key={s.session_id} style={{
                background: '#f9fafb', border: '1px solid #e5e7eb',
                borderRadius: '0.5rem', padding: '0.75rem 1rem',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: '600', fontSize: '0.9rem' }}>
                    Session {pastSessions.length - i} — {new Date(s.session_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>
                    {s.total_points} clips · {s.reflux_count} reflux
                  </span>
                </div>
                {s.session_summary && (
                  <p style={{ fontSize: '0.85rem', color: '#374151', margin: 0 }}>{s.session_summary}</p>
                )}
                {s.guidance_history?.slice(0, 2).map((h, j) => (
                  <div key={j} style={{
                    marginTop: '0.35rem', fontSize: '0.8rem', color: '#6b7280',
                    borderLeft: `3px solid ${h.flow_type === 'RP' ? '#dc2626' : '#059669'}`,
                    paddingLeft: '0.5rem',
                  }}>
                    [{h.flow_type}] {h.instruction}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {showSessions && pastSessions.length === 0 && (
        <div style={{ background: '#f9fafb', borderRadius: '0.5rem', padding: '0.75rem 1rem', marginBottom: '1rem', fontSize: '0.9rem', color: '#6b7280' }}>
          No past sessions yet for {sonographer?.name}. Complete a stream analysis to save a session.
        </div>
      )}

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
          <button
            onClick={() => { setMode('analyze-session'); setSessionComparison(null); setSelectedSession(null); setError(null); }}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: mode === 'analyze-session' ? '#7c3aed' : '#e5e7eb',
              color: mode === 'analyze-session' ? 'white' : '#1f2937',
              border: 'none',
              borderRadius: '0.375rem',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            📊 Analyze Previous Session
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
      ) : mode === 'stream' ? (
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
                            borderLeft: '3px solid ' + (item.flow_type === 'RP' ? '#dc2626' : '#059669'),
                            wordWrap: 'break-word',
                            overflowWrap: 'break-word'
                          }}
                        >
                          <div style={{ fontWeight: '600', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                            <span>Point {item.point}</span>
                            <span style={{ color: item.flow_type === 'RP' ? '#dc2626' : '#059669' }}>
                              {item.flow_type === 'RP' ? '❌ Reflux' : '✓ Normal'}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.95rem', color: '#374151', marginBottom: '0.5rem', wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                            {item.instruction}
                          </div>
                          {item.clinical_reason && (
                            <div style={{ fontSize: '0.85rem', color: '#6b7280', wordWrap: 'break-word', overflowWrap: 'break-word' }}>
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
      ) : (
        // ===== ANALYZE PREVIOUS SESSION MODE =====
        <>
          <div className="section">
            <h2 className="section-title">📊 Analyze Previous Sonographer Session</h2>
            <p className="text-muted" style={{ marginBottom: '1.5rem' }}>
              Review past session patterns and get personalized guidance based on {sonographer?.name}'s scanning history.
            </p>

            {pastSessions.length === 0 ? (
              <div className="output-container" style={{ textAlign: 'center', padding: '2rem' }}>
                <p style={{ fontSize: '1.1rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  No previous sessions available
                </p>
                <p className="text-muted">
                  Complete a stream or single position analysis to create your first session record.
                </p>
              </div>
            ) : (
              <>
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ fontWeight: '600', display: 'block', marginBottom: '0.75rem' }}>
                    Select Session to Analyze:
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
                    {pastSessions.map((session) => (
                      <div
                        key={session.session_id}
                        onClick={() => setSelectedSession(session)}
                        style={{
                          padding: '1rem',
                          backgroundColor: selectedSession?.session_id === session.session_id ? '#eff6ff' : '#f9fafb',
                          border: selectedSession?.session_id === session.session_id ? '2px solid #3b82f6' : '1px solid #e5e7eb',
                          borderRadius: '0.5rem',
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          if (selectedSession?.session_id !== session.session_id) {
                            e.currentTarget.style.backgroundColor = '#f0fdf4';
                            e.currentTarget.style.borderColor = '#d1fae5';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (selectedSession?.session_id !== session.session_id) {
                            e.currentTarget.style.backgroundColor = '#f9fafb';
                            e.currentTarget.style.borderColor = '#e5e7eb';
                          }
                        }}
                      >
                        <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#1e40af' }}>
                          {new Date(session.session_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                        </div>
                        <div style={{ fontSize: '0.9rem', color: '#374151', marginBottom: '0.5rem' }}>
                          📍 {session.total_points} clips · 🔴 {session.reflux_count} reflux detections
                        </div>
                        {session.session_summary && (
                          <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>
                            {session.session_summary}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  className="btn btn-primary"
                  onClick={handleAnalyzePreviousSession}
                  disabled={!selectedSession || loading}
                  style={{ marginTop: '1rem' }}
                >
                  {loading ? '🔄 Analyzing Session...' : '🔍 Analyze Selected Session'}
                </button>

                {analysisTime && !loading && (
                  <p className="text-muted mt-2">
                    Analysis completed in {analysisTime}s
                  </p>
                )}
              </>
            )}
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

          {/* Session Analysis Output */}
          {sessionComparison && (
            <>
              {/* Session Overview */}
              <div className="output-container">
                <div className="output-header">
                  <h3>📋 Session Overview</h3>
                  <span className="output-status success">✓ Analysis Complete</span>
                </div>
                <div className="output-content">
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2rem', fontWeight: '700', color: '#3b82f6' }}>
                        {sessionComparison.total_points}
                      </div>
                      <div style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.5rem' }}>
                        Total Clips Analyzed
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2rem', fontWeight: '700', color: '#dc2626' }}>
                        {sessionComparison.reflux_count}
                      </div>
                      <div style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.5rem' }}>
                        Reflux Detections
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2rem', fontWeight: '700', color: '#059669' }}>
                        {(sessionComparison.total_points - sessionComparison.reflux_count)}
                      </div>
                      <div style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.5rem' }}>
                        Normal Areas
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div style={{ padding: '1rem', backgroundColor: '#fef2f2', borderRadius: '0.5rem', borderLeft: '4px solid #dc2626' }}>
                      <div style={{ fontWeight: '600', color: '#991b1b', marginBottom: '0.5rem' }}>
                        Reflux Percentage
                      </div>
                      <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#dc2626' }}>
                        {sessionComparison.analysis.reflux_percentage}%
                      </div>
                    </div>
                    <div style={{ padding: '1rem', backgroundColor: '#f0fdf4', borderRadius: '0.5rem', borderLeft: '4px solid #059669' }}>
                      <div style={{ fontWeight: '600', color: '#155e3b', marginBottom: '0.5rem' }}>
                        Normal Percentage
                      </div>
                      <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#059669' }}>
                        {sessionComparison.analysis.normal_percentage}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Findings */}
              <div className="output-container">
                <div className="output-header">
                  <h3>🔎 Key Findings from Session</h3>
                </div>
                <div className="output-content">
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                    <div>
                      <div style={{ fontWeight: '600', color: '#dc2626', marginBottom: '0.75rem' }}>
                        Top Reflux Locations Detected:
                      </div>
                      <ul style={{ margin: 0, paddingLeft: '1rem', color: '#374151' }}>
                        {sessionComparison.analysis.reflux_locations.map((loc, i) => (
                          <li key={i} style={{ marginBottom: '0.5rem', fontSize: '0.95rem' }}>
                            {loc}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <div style={{ fontWeight: '600', color: '#059669', marginBottom: '0.75rem' }}>
                        Normal Areas Assessed:
                      </div>
                      <ul style={{ margin: 0, paddingLeft: '1rem', color: '#374151' }}>
                        {sessionComparison.analysis.normal_areas.map((area, i) => (
                          <li key={i} style={{ marginBottom: '0.5rem', fontSize: '0.95rem' }}>
                            {area}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Personalized Insights */}
              <div className="output-container">
                <div className="output-header">
                  <h3>💡 Personalized Scanning Insights</h3>
                </div>
                <div className="output-content">
                  <div style={{ padding: '1rem', backgroundColor: '#f5f3ff', borderRadius: '0.5rem', borderLeft: '4px solid #7c3aed' }}>
                    <div style={{ fontSize: '1rem', lineHeight: '1.8', color: '#374151' }}>
                      <p>
                        Based on <strong>{sonographer?.name}</strong>'s scanning style and this session:
                      </p>
                      <ul style={{ marginTop: '1rem', paddingLeft: '1.5rem', color: '#374151' }}>
                        <li style={{ marginBottom: '0.7rem' }}>
                          <strong>Scanning Pattern:</strong> {sonographer?.scanning_style}
                        </li>
                        <li style={{ marginBottom: '0.7rem' }}>
                          <strong>Session Duration:</strong> Approximately {sessionComparison.analysis.session_duration_min} minutes
                        </li>
                        <li style={{ marginBottom: '0.7rem' }}>
                          <strong>Focus Areas:</strong> This session focused on comprehensive bilateral assessment with {sessionComparison.total_points} probe positions
                        </li>
                        <li>
                          <strong>Next Steps:</strong> Use these historical patterns for personalized guidance in future sessions. The system will adapt recommendations based on this sonographer's technique and habits.
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Detailed History */}
              {sessionComparison.guidance_history.length > 0 && (
                <div className="output-container">
                  <div className="output-header">
                    <h3>📈 Complete Guidance History</h3>
                  </div>
                  <div className="output-content">
                    <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                      {sessionComparison.guidance_history.map((item, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: '0.75rem',
                            marginBottom: '0.75rem',
                            backgroundColor: '#f3f4f6',
                            borderRadius: '0.375rem',
                            borderLeft: '3px solid ' + (item.flow_type === 'RP' ? '#dc2626' : '#059669'),
                            wordWrap: 'break-word',
                            overflowWrap: 'break-word'
                          }}
                        >
                          <div style={{ fontWeight: '600', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                            <span>Position {idx + 1} of {sessionComparison.guidance_history.length}</span>
                            <span style={{ color: item.flow_type === 'RP' ? '#dc2626' : '#059669' }}>
                              {item.flow_type === 'RP' ? '❌ Reflux Detected' : '✓ Normal Flow'}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.95rem', color: '#374151', marginBottom: '0.5rem', wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                            {item.instruction}
                          </div>
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
              <p>🔄 Analyzing session and building personalized profile...</p>
              <p className="text-muted mt-1">Processing {selectedSession?.total_points || 0} probe positions...</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ProbeGuidance;
