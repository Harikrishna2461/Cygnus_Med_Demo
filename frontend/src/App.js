import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ClinicalReasoning from './pages/ClinicalReasoning';
import ProbeGuidance from './pages/ProbeGuidance';
import MLOpsDashboard from './pages/MLOpsDashboard';
import VisionAnalyzer from './pages/VisionAnalyzer';
import VeinClassification from './pages/VeinClassification';
import VideoAnalysis from './pages/VideoAnalysis';
import './App.css';

function App() {
  const [appInfo, setAppInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch app info on startup
    fetch('/api/info')
      .then(res => res.json())
      .then(data => {
        setAppInfo(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch app info:', err);
        setError('Backend not available');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="app-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Connecting to Clinical Support System...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-container">
        <div className="error-banner">
          <h2>⚠️ Connection Error</h2>
          <p>{error}</p>
          <p>Make sure the Flask backend is running on http://localhost:5000</p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="header-logo-left">
            <img src="/logo.png" alt="Cygnus Medical Logo" className="logo-image" />
          </div>
          <div className="header-content">
            <h1>Cygnus Medical</h1>
            <p className="subtitle">AI-Assisted CHIVA Shunt Assessment &amp; Clinical Decision Support</p>
          </div>
        </header>

        {/* Navigation */}
        <nav className="app-nav">
          <Link to="/" className="nav-link">
            📋 Clinical Reasoning (RAG)
          </Link>
          <Link to="/probe" className="nav-link">
            🎯 Probe Guidance
          </Link>
          <Link to="/vision" className="nav-link">
            🩺 Vein Detection
          </Link>
          <Link to="/classify-veins" className="nav-link">
            🔬 Vein Classification (VLM)
          </Link>
          <Link to="/video-analysis" className="nav-link">
            🎬 Video Analysis
          </Link>
          <Link to="/mlops" className="nav-link">
            📊 MLOps & LLMOps Monitoring
          </Link>
        </nav>

        {/* Main Content */}
        <main className="app-main">
          <Routes>
            <Route path="/" element={<ClinicalReasoning />} />
            <Route path="/probe" element={<ProbeGuidance />} />
            <Route path="/vision" element={<VisionAnalyzer />} />
            <Route path="/classify-veins" element={<VeinClassification />} />
            <Route path="/video-analysis" element={<VideoAnalysis />} />
            <Route path="/mlops" element={<MLOpsDashboard />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <p>Cygnus Medical &copy; {new Date().getFullYear()} &nbsp;|&nbsp; AI-Assisted CHIVA Assessment &nbsp;|&nbsp; For clinical review only — validated by a qualified vascular surgeon</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
