# CHIVA Clinical Assistant — Quick Start Guide

## Prerequisites

**Minimum Requirements:**
- Windows OS
- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Node.js 16+ ([Download](https://nodejs.org/)) — *optional, only needed for frontend development*

## Getting Started

### Option 1: Backend Only (Recommended for Testing)

The fastest way to start the backend API:

1. **Double-click `start.bat`** in the project root
   - This will automatically:
     - Check if Python is installed
     - Install all required libraries
     - Start the Flask backend on `http://localhost:5000`

2. **Test the API** in your browser:
   - Status: `http://localhost:5000/api/status`
   - Web UI: `http://localhost:5000/` (served from built frontend)

### Option 2: Full Stack (Backend + Frontend Dev Server)

For frontend development with hot-reload:

**Terminal 1 — Backend:**
```batch
start.bat
```

**Terminal 2 — Frontend:**
```batch
start_frontend.bat
```

Then open `http://localhost:3000` in your browser (React dev server with hot-reload)

## Troubleshooting

### "Python is not installed or not in PATH"
- Install Python from https://www.python.org/downloads/
- **Important:** During installation, check the box "Add Python to PATH"
- Restart your terminal after installation

### "pip install failed"
- Check your internet connection
- Try running manually: `cd cmed_demo\backend && pip install -r requirements.txt`
- If using a proxy, configure pip accordingly

### "Qdrant vector database not found"
- This is expected on first run
- The app will create it when knowledge base is ingested
- Or run the parent backend's ingestion script to populate it

### "Cannot find module" errors
- Make sure you completed the `pip install` step
- Try deleting `__pycache__` folders and running again
- Restart your terminal

## File Structure

```
Cygnus_Med_Demo/
├── start.bat                  ← Run this to start backend
├── start_frontend.bat         ← Run this to start frontend dev server
├── cmed_demo/
│   ├── backend/
│   │   ├── app.py            (Flask app entry point)
│   │   ├── requirements.txt   (Python dependencies)
│   │   ├── config.py          (Configuration)
│   │   └── qdrant_storage/    (Vector database - created on first use)
│   ├── frontend/
│   │   ├── src/               (React source code)
│   │   └── package.json       (Node.js dependencies)
│   └── ...
├── cross_encoder_finetuning/  (Fine-tuned model - research only)
├── backend/                   (Parent backend - for ingestion, etc.)
└── ...
```

## Ports

- **Backend API:** http://localhost:5000
- **Frontend Dev Server:** http://localhost:3000 (if running start_frontend.bat)
- **Qdrant Vector DB:** localhost:6333 (internal, no need to access directly)

## Next Steps

1. Once the app is running, open http://localhost:5000 in your browser
2. Try the clinical assistant by describing a patient's venous condition
3. The app will classify the shunt type and suggest ligation approaches

## For Development

See the individual README files in each module:
- `cmed_demo/backend/README.md` — Backend API documentation
- `cmed_demo/frontend/README.md` — Frontend component documentation
- `backend/README.md` — Parent backend for knowledge base ingestion

