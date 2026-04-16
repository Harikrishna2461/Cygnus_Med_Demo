import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const MLOpsDashboard = () => {
  const [selectedTask, setSelectedTask] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);
  const [runs, setRuns] = useState([]);
  const [runDetails, setRunDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch runs for selected task
  useEffect(() => {
    if (!selectedTask) return;

    setLoading(true);
    setError(null);
    setSelectedRun(null);
    setRunDetails(null);

    axios
      .get(`/api/mlops/runs/${selectedTask}`)
      .then((res) => {
        setRuns(res.data.runs);
      })
      .catch((err) => {
        setError(err.response?.data?.error || err.message);
      })
      .finally(() => setLoading(false));
  }, [selectedTask]);

  // Fetch run details for selected run
  useEffect(() => {
    if (!selectedRun) return;

    setLoading(true);
    setError(null);

    axios
      .get(`/api/mlops/run-details/${selectedRun}`)
      .then((res) => {
        setRunDetails(res.data);
      })
      .catch((err) => {
        setError(err.response?.data?.error || err.message);
      })
      .finally(() => setLoading(false));
  }, [selectedRun]);

  return (
    <div className="page-container" style={{ paddingTop: '2rem' }}>
      <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: '700' }}>
        🎬 MLOps & LLMOps Dashboard
      </h1>

      {/* Task Selection */}
      {!selectedTask && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
          <button
            onClick={() => setSelectedTask('Clinical Reasoning')}
            style={{
              padding: '2rem',
              fontSize: '1.2rem',
              fontWeight: '600',
              backgroundColor: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
            onMouseOver={(e) => (e.target.style.backgroundColor = '#059669')}
            onMouseOut={(e) => (e.target.style.backgroundColor = '#10b981')}
          >
            📊 Task 1: Clinical Reasoning
          </button>
          <button
            onClick={() => setSelectedTask('Probe Guidance')}
            style={{
              padding: '2rem',
              fontSize: '1.2rem',
              fontWeight: '600',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
            onMouseOver={(e) => (e.target.style.backgroundColor = '#1e40af')}
            onMouseOut={(e) => (e.target.style.backgroundColor = '#3b82f6')}
          >
            🎯 Task 2: Probe Guidance
          </button>
        </div>
      )}

      {/* Runs List */}
      {selectedTask && !selectedRun && (
        <>
          <button
            onClick={() => {
              setSelectedTask(null);
              setRuns([]);
            }}
            style={{
              padding: '0.5rem 1rem',
              marginBottom: '1.5rem',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '0.375rem',
              cursor: 'pointer'
            }}
          >
            ← Back to Tasks
          </button>

          <div className="section">
            <h2 className="section-title">📈 {selectedTask} - Run Executions</h2>
            <div className="section-content">
              {loading && <p>Loading runs...</p>}
              {error && <p style={{ color: '#dc2626' }}>❌ {error}</p>}
              {!loading && runs.length === 0 && (
                <p style={{ textAlign: 'center', opacity: 0.6 }}>No runs found for this task</p>
              )}
              {!loading && runs.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {runs.map((run) => (
                    <div
                      key={run.run_id}
                      onClick={() => setSelectedRun(run.run_id)}
                      style={{
                        padding: '1rem',
                        backgroundColor: '#f3f4f6',
                        border: '2px solid #d1d5db',
                        borderRadius: '0.5rem',
                        cursor: 'pointer',
                        transition: 'all 0.3s',
                        userSelect: 'none'
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.borderColor = '#3b82f6';
                        e.currentTarget.style.backgroundColor = '#eff6ff';
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.borderColor = '#d1d5db';
                        e.currentTarget.style.backgroundColor = '#f3f4f6';
                      }}
                    >
                      <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Run: {run.run_id}</div>
                      <div style={{ fontSize: '0.9rem', color: '#6b7280', marginBottom: '0.3rem' }}>
                        Type: {run.task_type}
                      </div>
                      <div style={{ fontSize: '0.9rem', color: '#6b7280', marginBottom: '0.3rem' }}>
                        Duration: {run.total_duration_ms?.toFixed(2) || '—'} ms
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
                        {new Date(run.start_time).toLocaleString()}
                      </div>
                      {run.num_samples && (
                        <div style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.3rem' }}>
                          📊 Samples: {run.num_samples}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Run Details & Metrics */}
      {selectedRun && runDetails && (
        <>
          <button
            onClick={() => {
              setSelectedRun(null);
              setRunDetails(null);
            }}
            style={{
              padding: '0.5rem 1rem',
              marginBottom: '1.5rem',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '0.375rem',
              cursor: 'pointer'
            }}
          >
            ← Back to Runs
          </button>

          {/* Summary Statistics */}
          <div className="section">
            <h2 className="section-title">📊 Summary Statistics</h2>
            <div className="section-content">
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '1.5rem'
                }}
              >
                <MetricCard
                  title="Avg Response Time"
                  value={runDetails.summary_statistics.avg_response_ms.toFixed(2)}
                  unit="ms"
                  icon="⏱️"
                />
                <MetricCard
                  title="Peak Memory"
                  value={runDetails.summary_statistics.peak_memory_mb.toFixed(2)}
                  unit="MB"
                  icon="💾"
                />
                <MetricCard
                  title="Avg CPU"
                  value={runDetails.summary_statistics.avg_cpu_percent.toFixed(2)}
                  unit="%"
                  icon="🖥️"
                />
                <MetricCard
                  title="Total Requests"
                  value={runDetails.total_requests}
                  icon="📋"
                />
                <MetricCard
                  title="Cached Requests"
                  value={runDetails.summary_statistics.total_cached}
                  icon="⚡"
                />
                <MetricCard
                  title="Min Response"
                  value={runDetails.summary_statistics.min_response_ms.toFixed(2)}
                  unit="ms"
                  icon="🎯"
                />
              </div>
            </div>
          </div>

          {/* Response Time Trend */}
          {runDetails.request_metrics.length > 0 && (
            <div className="section">
              <h2 className="section-title">📈 Response Time Trend</h2>
              <div className="section-content">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={runDetails.request_metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="request_number" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="response_time_ms"
                      stroke="#3b82f6"
                      dot={false}
                      name="Response Time (ms)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Task Breakdown (Time Components) */}
          {runDetails.request_metrics.some((m) => m.llm_inference_ms || m.rag_retrieval_ms) && (
            <div className="section">
              <h2 className="section-title">⏱️ Task Component Breakdown</h2>
              <div className="section-content">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={runDetails.request_metrics.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="request_number" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="rag_retrieval_ms" stackId="a" fill="#10b981" name="RAG Retrieval" />
                    <Bar dataKey="llm_inference_ms" stackId="a" fill="#f59e0b" name="LLM Inference" />
                    <Bar dataKey="post_processing_ms" stackId="a" fill="#8b5cf6" name="Post-Processing" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Memory & CPU Usage */}
          <div className="section">
            <h2 className="section-title">🖥️ System Resource Usage</h2>
            <div className="section-content">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={runDetails.request_metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="request_number" />
                  <YAxis yAxisId="left" label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: 'CPU (%)', angle: 90, position: 'insideRight' }} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="memory_usage_mb" stroke="#06b6d4" name="Memory (MB)" />
                  <Line yAxisId="right" type="monotone" dataKey="cpu_percent" stroke="#ef4444" name="CPU (%)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Input/Output Size Trend */}
          <div className="section">
            <h2 className="section-title">📤 Input/Output Data Size</h2>
            <div className="section-content">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={runDetails.request_metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="request_number" />
                  <YAxis label={{ value: 'Bytes', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="input_size_bytes" fill="#3b82f6" name="Input Size (bytes)" />
                  <Bar dataKey="output_size_bytes" fill="#10b981" name="Output Size (bytes)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Request Table */}
          <div className="section">
            <h2 className="section-title">📋 Detailed Request Metrics</h2>
            <div className="section-content">
              <div style={{ overflowX: 'auto' }}>
                <table
                  style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '0.9rem'
                  }}
                >
                  <thead>
                    <tr style={{ backgroundColor: '#f3f4f6', borderBottom: '2px solid #d1d5db' }}>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>Req</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>Response (ms)</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>Memory (MB)</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>CPU (%)</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>Cache</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runDetails.request_metrics.map((metric, idx) => (
                      <tr
                        key={idx}
                        style={{
                          backgroundColor: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                          borderBottom: '1px solid #e5e7eb'
                        }}
                      >
                        <td style={{ padding: '0.75rem' }}>{metric.request_number}</td>
                        <td style={{ padding: '0.75rem', color: '#0891b2' }}>
                          {metric.response_time_ms.toFixed(2)}
                        </td>
                        <td style={{ padding: '0.75rem' }}>
                          {metric.memory_usage_mb.toFixed(2)}
                        </td>
                        <td style={{ padding: '0.75rem' }}>{metric.cpu_percent.toFixed(2)}</td>
                        <td style={{ padding: '0.75rem' }}>
                          {metric.cached ? '✅' : '❌'}
                        </td>
                        <td style={{ padding: '0.75rem', color: metric.error ? '#dc2626' : '#059669' }}>
                          {metric.error ? '❌ ' + metric.error : '✓'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Stream Metrics */}
          {runDetails.stream_metrics && (
            <div className="section">
              <h2 className="section-title">🌊 Stream Execution Metrics</h2>
              <div className="section-content">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                  <MetricCard
                    title="Total Points"
                    value={runDetails.stream_metrics.total_points}
                    icon="📊"
                  />
                  <MetricCard
                    title="Processed Points"
                    value={runDetails.stream_metrics.processed_points}
                    icon="✅"
                  />
                  <MetricCard
                    title="Avg Point Duration"
                    value={runDetails.stream_metrics.average_point_duration_ms?.toFixed(2)}
                    unit="ms"
                    icon="⏱️"
                  />
                  <MetricCard
                    title="Stream Total Duration"
                    value={(runDetails.stream_metrics.total_stream_duration_ms / 1000).toFixed(2)}
                    unit="s"
                    icon="⏱️"
                  />
                  <MetricCard
                    title="Total Tokens"
                    value={runDetails.stream_metrics.total_input_tokens + runDetails.stream_metrics.total_output_tokens}
                    icon="🎫"
                  />
                  <MetricCard
                    title="Peak Memory"
                    value={runDetails.stream_metrics.total_memory_peak_mb?.toFixed(2)}
                    unit="MB"
                    icon="💾"
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

// Reusable Metric Card Component
const MetricCard = ({ title, value, unit = '', icon = '' }) => (
  <div
    style={{
      padding: '1.5rem',
      backgroundColor: '#f3f4f6',
      border: '1px solid #d1d5db',
      borderRadius: '0.5rem'
    }}
  >
    <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{icon}</div>
    <div style={{ fontWeight: '600', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#6b7280' }}>
      {title}
    </div>
    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1f2937' }}>
      {value}
      {unit && <span style={{ fontSize: '0.9rem', marginLeft: '0.25rem' }}>{unit}</span>}
    </div>
  </div>
);

export default MLOpsDashboard;
