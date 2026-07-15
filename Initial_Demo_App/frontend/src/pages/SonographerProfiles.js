import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SonographerProfiles = () => {
  const navigate = useNavigate();
  const [sonographers, setSonographers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('/api/sonographers')
      .then(res => { setSonographers(res.data); setLoading(false); })
      .catch(err => { setError(err.message); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '4rem' }}>
        <div className="spinner" />
        <p style={{ marginTop: '1rem', color: '#6b7280' }}>Loading sonographer profiles...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
          <p className="text-error">Failed to load profiles: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div className="section" style={{ marginBottom: '0.5rem' }}>
        <h2 className="section-title" style={{ fontSize: '1.5rem' }}>
          🎯 Probe Guidance — Select Sonographer
        </h2>
        <p style={{ color: '#6b7280', marginTop: '0.5rem', lineHeight: '1.6' }}>
          Each sonographer has a personalised digital twin. The AI learns from their scanning style and
          past sessions to provide tailored probe guidance.
        </p>
      </div>

      {/* Profile Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem', marginTop: '1.5rem' }}>
        {sonographers.map(s => (
          <div
            key={s.id}
            onClick={() => navigate(`/probe/${s.id}`)}
            style={{
              background: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '0.75rem',
              padding: '1.5rem',
              cursor: 'pointer',
              transition: 'box-shadow 0.15s, transform 0.15s',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.08)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            {/* Avatar + Name */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{
                width: '52px', height: '52px',
                borderRadius: '50%',
                backgroundColor: s.avatar_color || '#3b82f6',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.4rem', color: 'white', fontWeight: '700', flexShrink: 0,
              }}>
                {s.name.split(' ').map(n => n[0]).slice(0, 2).join('')}
              </div>
              <div>
                <div style={{ fontWeight: '700', fontSize: '1.05rem', color: '#111827' }}>{s.name}</div>
                <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>{s.title}</div>
              </div>
            </div>

            {/* Specialty */}
            <div style={{
              background: '#f9fafb', borderRadius: '0.4rem',
              padding: '0.5rem 0.75rem', marginBottom: '0.75rem',
              fontSize: '0.875rem', color: '#374151',
            }}>
              <span style={{ fontWeight: '600' }}>Specialty: </span>{s.specialty}
            </div>

            {/* Experience */}
            <div style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.75rem' }}>
              🏥 {s.experience_years} years experience
            </div>

            {/* Session stats */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{
                flex: 1, textAlign: 'center',
                background: '#eff6ff', borderRadius: '0.4rem', padding: '0.5rem',
              }}>
                <div style={{ fontSize: '1.3rem', fontWeight: '700', color: '#1d4ed8' }}>
                  {s.session_count || 0}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Sessions</div>
              </div>
              <div style={{
                flex: 2, textAlign: 'center',
                background: '#f0fdf4', borderRadius: '0.4rem', padding: '0.5rem',
              }}>
                <div style={{ fontSize: '0.8rem', fontWeight: '600', color: '#166534' }}>
                  {s.last_session_date
                    ? new Date(s.last_session_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
                    : 'No sessions yet'}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Last session</div>
              </div>
            </div>

            {/* CTA */}
            <div style={{
              textAlign: 'center',
              background: s.avatar_color || '#3b82f6',
              color: 'white',
              borderRadius: '0.4rem',
              padding: '0.6rem',
              fontWeight: '600',
              fontSize: '0.9rem',
            }}>
              Start Session →
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SonographerProfiles;
