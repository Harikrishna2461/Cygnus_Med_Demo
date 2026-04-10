# Clinical Medical Decision Support System

A full-stack demo application for clinical reasoning and ultrasound probe guidance using **RAG + Local LLM**.

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   React Frontend                         │
│        (Two pages: Clinical Reasoning + Probe Guidance)  │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/JSON
┌─────────────────────────────────────────────────────────┐
│                  Flask Backend                           │
│  - RAG Pipeline (FAISS + Ollama)                        │
│  - Probe Guidance (Vector Math + LLM)                   │
│  - Response Caching & Logging                           │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    ┌───────────┐      ┌─────────────┐
    │   Ollama  │      │  FAISS DB   │
    │   LLM     │      │ (Persistent)│
    │(Mistral)  │      │             │
    └───────────┘      └─────────────┘
```

## 📋 Features

### Page 1: Clinical Reasoning (RAG)
- Input ultrasound JSON data
- Retrieve top-3 relevant chunks from FAISS medical index
- Generate structured clinical analysis:
  - Shunt Type Assessment
  - Hemodynamic Reasoning (N1/N2/N3, Entry Point, Re-entry pathways)
  - Treatment Plan (CHIVA principles)
- **Response caching** for repeated inputs
- Latency tracking

### Page 2: Probe Guidance
- Interactive canvas for probe positioning
- Calculate displacement vector (Δx, Δy, distance)
- **LLM-generated** human-readable instructions
- Real-time position tracking
- Movement magnitude categorization

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 18 + React Router |
| Backend | Flask + Flask-CORS |
| LLM | Ollama (Mistral or Phi-3) |
| Vector DB | FAISS (CPU) |
| Embeddings | Ollama embeddings API |
| Caching | In-memory Python dict |

## 📦 Prerequisites

1. **Python 3.8+**
2. **Node.js 16+** (for frontend)
3. **Ollama** (installed and running)
   - Download: https://ollama.ai
   - Model: `mistral` or `neural-chat`

### Verify Ollama Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# Pull model (in another terminal)
ollama pull mistral
```

## 🚀 Installation & Setup

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Medical Data

This creates FAISS index from medical text:

```bash
cd backend
source venv/bin/activate
python ingest.py
```

**Expected output:**
```
================
MEDICAL TEXT INGESTION
================
[1/4] Loading medical text...
[2/4] Chunking text...
✓ Created 45 chunks

[3/4] Generating embeddings with Ollama...
✓ Generated 45 embeddings in 12.3s

[4/4] Creating FAISS index...
✓ Saved FAISS index: backend/faiss_index/medical_index.faiss
✓ Saved metadata: backend/faiss_index/metadata.pkl

Index Statistics:
  - Total chunks: 45
  - Embedding dimension: 4096
```

### 3. Start Backend Server

```bash
cd backend
source venv/bin/activate
python app.py
```

**Expected output:**
```
════════════════════════════════════════
Clinical Medical Decision Support Backend
════════════════════════════════════════
✓ Loaded FAISS index with 45 chunks
Starting Flask server on http://localhost:5000
```

### 4. Frontend Setup (in NEW terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Browser will open at `http://localhost:3000`

## 📝 Usage

### Clinical Reasoning Page

1. Input ultrasound data (provided as JSON example)
2. Click **"Analyze"**
3. See structured output:
   - Shunt Type Assessment
   - Hemodynamic Reasoning
   - Treatment Plan

**Example Input:**
```json
{
  "vein_diameter": 8.5,
  "flow_velocity": 0.45,
  "reflux_present": true,
  "reflux_duration": 2.3,
  "location": "GSV",
  "valve_competence": "incompetent"
}
```

### Probe Guidance Page

1. **Click on canvas** to position:
   - Red dot = Ultrasound probe
   - Blue dot = Target vein
2. Click **"Get Guidance"**
3. See:
   - Movement calculations (Δx, Δy, distance)
   - LLM-generated instruction
   - Magnitude categorization

## 🔍 Key API Endpoints

### Health & Info
```
GET /api/health          → Check backend status
GET /api/info           → App metadata
```

### Clinical Reasoning
```
POST /api/analyze
Request:  { "ultrasound_data": {...} }
Response: {
  "shunt_type_assessment": "...",
  "reasoning": "...",
  "treatment_plan": "...",
  "raw_response": "..."
}
```

### Probe Guidance
```
POST /api/probe-guidance
Request:  { "probe": [x, y], "target": [x, y] }
Response: {
  "dx": number,
  "dy": number,
  "distance": number,
  "base_direction": "left|right|up|down",
  "magnitude": "minimal|slight|moderate|large",
  "instruction": "string"
}
```

## ⚙️ Configuration

Edit `backend/config.py` to customize:

```python
# LLM Model
OLLAMA_MODEL = "mistral"              # or "neural-chat", "phi3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 400                      # tokens (approximate)
CHUNK_OVERLAP = 50                    # tokens

# Cache
CACHE_TTL = 3600                      # seconds
CACHE_ENABLED = True
```

## 📊 Performance Considerations

### Optimizations Implemented

1. **FAISS Index Caching**: Loaded once at startup
2. **Response Caching**: In-memory cache with TTL
3. **Async-Ready**: Backend supports concurrent requests
4. **Prompt Optimization**: Concise LLM prompts
5. **Lazy Loading**: Medical data only loaded during analysis

### Typical Latencies

| Operation | Time |
|-----------|------|
| FAISS Retrieval | 0.5-1.0s |
| Ollama Embedding | 1-2s |
| LLM Inference | 3-8s |
| **Total End-to-End** | **5-12s** |

*Times vary by model and hardware*

## 🐛 Troubleshooting

### Ollama Not Found
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### FAISS Index Not Found
```bash
cd backend
source venv/bin/activate
python ingest.py
```

### CORS Issues
Ensure Flask-CORS is installed:
```bash
pip install Flask-CORS
```

### Model Not Found
```bash
ollama pull mistral
# or
ollama pull neural-chat
```

### Port Already in Use
```bash
# Change port in app.py (line ~500)
app.run(host='0.0.0.0', port=5001)  # Use 5001 instead
```

## 📁 Project Structure

```
cmed_demo/
├── backend/
│   ├── app.py                 # Flask application
│   ├── ingest.py             # PDF ingestion + FAISS indexing
│   ├── config.py             # Configuration
│   ├── requirements.txt       # Python dependencies
│   └── faiss_index/          # Persisted FAISS index
│       ├── medical_index.faiss
│       └── metadata.pkl
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js            # Main app component
│   │   ├── App.css           # Global styles
│   │   ├── index.js          # React entry point
│   │   └── pages/
│   │       ├── ClinicalReasoning.js
│   │       └── ProbeGuidance.js
│   ├── package.json
│   └── .gitignore
│
└── README.md
```

## 🚧 Development Notes

### Adding New Features

1. **New LLM Prompt**: Update `backend/app.py` function
2. **New Endpoint**: Add route in `app.py`
3. **New Page**: Create component in `frontend/src/pages/`
4. **Styling**: Update `frontend/src/App.css`

### Testing Backend

```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Test analysis
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ultrasound_data": {"vein_diameter": 8.5}}'

# Test probe guidance
curl -X POST http://localhost:5000/api/probe-guidance \
  -H "Content-Type: application/json" \
  -d '{"probe": [100, 100], "target": [200, 150]}'
```

## 🎓 Learning Resources

- **RAG Pattern**: https://python.langchain.com/docs/modules/data_connection/
- **Ollama**: https://github.com/ollama/ollama
- **FAISS**: https://github.com/facebookresearch/faiss
- **Medical Domain**: Adapted from vascular surgery best practices (CHIVA method)

## 📢 Important Notes

1. **Medical Disclaimer**: This is a **demonstration system** only. Not for clinical use.
2. **Data Privacy**: Medical data is processed locally (no cloud transmission)
3. **Ollama Models**: Download models on first use (large files)
4. **CPU vs GPU**: FAISS CPU is sufficient for demo; use GPU variant for production

## 🤝 Contributing

To extend this demo:

1. Add more medical text to `backend/ingest.py`
2. Implement specialized prompts for different conditions
3. Add persistent database (PostgreSQL) instead of in-memory cache
4. Deploy with production WSGI server (Gunicorn)

## 📄 License

MIT License - Educational use

## 👨‍💻 Questions?

For issues or questions:
1. Check backend logs: `backend/app.log`
2. Verify Ollama connectivity
3. Ensure FAISS index exists: `backend/faiss_index/`

---

**Built for clinical decision support research** | **RAG-powered LLM inference** | **Local-first privacy**
