# Frontend Integration Guide: Vision Vein Detection

This guide explains how to integrate the vision vein detection endpoints into the frontend application.

## 📡 API Endpoints

### Endpoint 1: Detect Veins from Video

**Endpoint**: `POST /api/vision/detect-veins`

**Purpose**: Process ultrasound video file to detect and classify veins

#### Request Format

```javascript
const formData = new FormData();
formData.append('file', videoFile);           // File input from user
formData.append('enable_llm', 'true');        // Optional: use Vision LLM
formData.append('llm_provider', 'openai');    // Optional: 'openai' or 'anthropic'
formData.append('llm_api_key', 'sk-...');     // Optional: API key
formData.append('max_frames', '30');          // Optional: max frames to process

const response = await fetch('/api/vision/detect-veins', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

#### Response Format

```javascript
{
  "status": "success",
  "video_file": "ultrasound.mp4",
  "total_frames_processed": 30,
  "summary": {
    "total_veins_detected": 15,
    "vein_type_summary": {
      "deep_vein": 4,
      "superficial_vein": 8,
      "perforator_vein": 2,
      "gsv": 1
    },
    "nlevel_summary": {
      "N1": 4,
      "N2": 8,
      "N3": 3
    },
    "gsv_frames": [5, 12, 18]
  },
  "frame_results": [
    {
      "frame_index": 0,
      "timestamp": "2024-01-15T10:30:45.123Z",
      "veins": [
        {
          "vein_id": "V0",
          "classification": {
            "primary_classification": "superficial_vein",
            "n_level": "N3",
            "confidence": 0.92
          },
          "spatial_analysis": {
            "vein_centroid": [256.5, 128.3],
            "distance_to_fascia_mm": 3.2,
            "relative_position": "above"
          }
        }
      ],
      "summary_statistics": {
        "total_veins": 2,
        "by_type": {...}
      }
    }
  ]
}
```

### Endpoint 2: Analyze Single Frame

**Endpoint**: `POST /api/vision/analyze-frame`

**Purpose**: Analyze a single ultrasound image

#### Request Format

```javascript
const formData = new FormData();
formData.append('file', imageFile);      // Image file input
formData.append('enable_llm', 'false');  // Optional: use Vision LLM

const response = await fetch('/api/vision/analyze-frame', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

#### Response Format

```javascript
{
  "status": "success",
  "image_file": "frame.jpg",
  "veins": [
    {
      "vein_id": "V0",
      "classification": {
        "primary_classification": "superficial_vein",
        "n_level": "N3",
        "confidence": 0.92
      },
      "spatial_analysis": {...}
    }
  ],
  "summary_statistics": {
    "total_veins": 1,
    "by_type": {...}
  },
  "visualization": {
    "format": "base64",
    "data": "iVBORw0KGgoAAAAN..."
  }
}
```

### Endpoint 3: Vision Module Health

**Endpoint**: `GET /api/vision/health`

**Purpose**: Check if vision module is available

#### Response Format

```javascript
{
  "status": "healthy",
  "module": "vision_vein_detection",
  "capabilities": [
    "video_processing",
    "frame_segmentation",
    "vein_classification",
    "spatial_analysis",
    "optional_llm_integration"
  ]
}
```

## 🎨 Sample React Component

### Video Analysis Component

```javascript
import React, { useState } from 'react';
import './VisionAnalysis.css';

const VisionAnalysis = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [enableLLM, setEnableLLM] = useState(false);

  const handleVideoUpload = (event) => {
    setVideoFile(event.target.files[0]);
    setError(null);
  };

  const analyzeVideo = async () => {
    if (!videoFile) {
      setError('Please select a video file');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', videoFile);
      formData.append('enable_llm', enableLLM.toString());
      formData.append('max_frames', '30');

      const response = await fetch('/api/vision/detect-veins', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="vision-analysis-container">
      <h2>🔍 Ultrasound Vein Detection</h2>
      
      <div className="upload-section">
        <input
          type="file"
          accept="video/*"
          onChange={handleVideoUpload}
          disabled={analyzing}
        />
        
        <label>
          <input
            type="checkbox"
            checked={enableLLM}
            onChange={(e) => setEnableLLM(e.target.checked)}
            disabled={analyzing}
          />
          Use AI Confirmation (Vision LLM)
        </label>

        <button
          onClick={analyzeVideo}
          disabled={analyzing || !videoFile}
        >
          {analyzing ? 'Analyzing...' : 'Analyze Video'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="results-section">
          <h3>Analysis Results</h3>
          
          <div className="summary">
            <p>📊 Total Veins: {result.summary.total_veins_detected}</p>
            <p>🎬 Frames Processed: {result.total_frames_processed}</p>
            {result.summary.gsv_frames.length > 0 && (
              <p>🏆 GSV Found in Frames: {result.summary.gsv_frames.join(', ')}</p>
            )}
          </div>

          <div className="vein-types">
            <h4>Vein Distribution by Type:</h4>
            <ul>
              {Object.entries(result.summary.vein_type_summary).map(([type, count]) => (
                <li key={type}>
                  {type.replace(/_/g, ' ')}: {count}
                </li>
              ))}
            </ul>
          </div>

          <div className="nlevel-distribution">
            <h4>Distribution by Depth Level:</h4>
            <ul>
              {Object.entries(result.summary.nlevel_summary).map(([level, count]) => (
                <li key={level}>
                  {level}: {count}
                </li>
              ))}
            </ul>
          </div>

          <VeinDetailsTable frameResults={result.frame_results} />
        </div>
      )}
    </div>
  );
};

// Sub-component to display vein details
const VeinDetailsTable = ({ frameResults }) => {
  const [expandedFrame, setExpandedFrame] = useState(null);

  return (
    <div className="detail-table">
      <h4>Vein Details by Frame</h4>
      {frameResults.map((frame, idx) => (
        <div key={idx} className="frame-section">
          <button
            className="frame-toggle"
            onClick={() => setExpandedFrame(expandedFrame === idx ? null : idx)}
          >
            Frame {frame.frame_index} ({frame.veins.length} veins)
          </button>

          {expandedFrame === idx && (
            <table className="veins-table">
              <thead>
                <tr>
                  <th>Vein ID</th>
                  <th>Type</th>
                  <th>Depth (N-Level)</th>
                  <th>Confidence</th>
                  <th>Distance to Fascia</th>
                </tr>
              </thead>
              <tbody>
                {frame.veins.map((vein) => (
                  <tr key={vein.vein_id}>
                    <td>{vein.vein_id}</td>
                    <td>{vein.classification.primary_classification}</td>
                    <td>{vein.classification.n_level}</td>
                    <td>{(vein.classification.confidence * 100).toFixed(0)}%</td>
                    <td>{vein.spatial_analysis.distance_to_fascia_mm.toFixed(1)} mm</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </div>
  );
};

export default VisionAnalysis;
```

### CSS Styling

```css
/* VisionAnalysis.css */

.vision-analysis-container {
  padding: 20px;
  margin: 20px 0;
  background: #f5f5f5;
  border-radius: 8px;
}

.vision-analysis-container h2 {
  color: #333;
  margin-bottom: 20px;
}

.upload-section {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  align-items: center;
  flex-wrap: wrap;
}

.upload-section input[type="file"],
.upload-section button {
  padding: 10px 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.upload-section button {
  background: #007bff;
  color: white;
  cursor: pointer;
  border: none;
}

.upload-section button:hover {
  background: #0056b3;
}

.upload-section button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.upload-section label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.error-message {
  color: #d32f2f;
  padding: 10px;
  background: #ffcdd2;
  border-radius: 4px;
  margin-bottom: 15px;
}

.results-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin-top: 20px;
}

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.summary p {
  padding: 10px;
  background: #e8f5e9;
  border-left: 4px solid #4caf50;
  margin: 0;
}

.vein-types,
.nlevel-distribution {
  margin-bottom: 20px;
}

.vein-types ul,
.nlevel-distribution ul {
  list-style: none;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.vein-types li,
.nlevel-distribution li {
  padding: 10px;
  background: #f0f0f0;
  border-radius: 4px;
  border-left: 3px solid #2196f3;
}

.detail-table {
  margin-top: 20px;
}

.frame-toggle {
  width: 100%;
  padding: 12px;
  background: #f9f9f9;
  border: 1px solid #ddd;
  cursor: pointer;
  text-align: left;
  margin-bottom: 5px;
  border-radius: 4px;
}

.frame-toggle:hover {
  background: #f0f0f0;
}

.veins-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  margin-bottom: 15px;
}

.veins-table th,
.veins-table td {
  padding: 10px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.veins-table th {
  background: #f5f5f5;
  font-weight: bold;
}

.veins-table tbody tr:hover {
  background: #f9f9f9;
}
```

### Image Analysis Component

```javascript
const ImageAnalysis = () => {
  const [imageFile, setImageFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleImageUpload = (event) => {
    setImageFile(event.target.files[0]);
  };

  const analyzeImage = async () => {
    if (!imageFile) return;

    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('enable_llm', 'false');

      const response = await fetch('/api/vision/analyze-frame', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div>
      <h3>Single Frame Analysis</h3>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <button onClick={analyzeImage} disabled={analyzing}>
        {analyzing ? 'Analyzing...' : 'Analyze Image'}
      </button>

      {result && (
        <div>
          <h4>Results: {result.veins.length} veins detected</h4>
          
          {result.visualization?.data && (
            <img
              src={`data:image/png;base64,${result.visualization.data}`}
              alt="Analyzed ultrasound"
              style={{ maxWidth: '100%', marginTop: '10px' }}
            />
          )}

          <pre>{JSON.stringify(result.summary_statistics, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};
```

## 🔑 Environment Variables

For LLM integration, set your API key:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # If using Anthropic
```

Or pass directly in the API call:
```javascript
formData.append('llm_api_key', 'sk-...');
```

## 🔄 Workflow

1. User selects ultrasound video file
2. Frontend sends POST to `/api/vision/detect-veins`
3. Backend processes video frame by frame
4. Each frame is:
   - Segmented (fascia + veins detected)
   - Analyzed spatially
   - Classified using rules
   - Optionally enhanced with LLM
5. Results returned with summary and frame-by-frame details
6. Frontend displays results, visualizations, and statistics

## 📊 Data Visualization Ideas

### 1. Vein Type Distribution Pie Chart
```javascript
<PieChart
  data={result.summary.vein_type_summary}
  title="Vein Types"
/>
```

### 2. Depth Distribution Bar Chart
```javascript
<BarChart
  data={result.summary.nlevel_summary}
  title="Veins by Depth Level"
/>
```

### 3. Timeline of GSV Presence
```javascript
<Timeline
  frames={result.total_frames_processed}
  gsv_frames={result.summary.gsv_frames}
/>
```

### 4. Interactive Frame Viewer
```javascript
<FrameViewer
  frames={result.frame_results}
  onFrameSelect={handleFrameSelect}
/>
```

## ⚠️ Error Handling

```javascript
async function analyzeWithErrorHandling(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/vision/detect-veins', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      if (response.status === 503) {
        throw new Error('Vision module not available. Install CV dependencies.');
      } else if (response.status === 400) {
        throw new Error('Invalid file format or input');
      } else {
        throw new Error(`Server error: ${response.status}`);
      }
    }

    const result = await response.json();
    
    if (result.status === 'error') {
      throw new Error(result.error);
    }

    return result;
  } catch (error) {
    console.error('Analysis failed:', error);
    // Display user-friendly error message
    return null;
  }
}
```

## 🚀 Performance Tips

1. **Reduce max_frames** for faster processing (default 30)
2. **Disable LLM** if not needed (saves API calls and time)
3. **Use compressed videos** for better upload speed
4. **Cache results** on frontend to avoid re-processing
5. **Stream progress** updates for better UX

## 📱 Mobile Considerations

- Use smaller max_frames on mobile devices
- Show progress indicators during processing
- Display results in responsive layout
- consider data usage (videos are large)

## 🔐 Security Notes

- Never expose API keys in frontend code
- Use environment variables or backend proxies
- Validate file types and sizes on frontend
- Sanitize user input

---

For more details, see [Vision Module README](./README.md)
