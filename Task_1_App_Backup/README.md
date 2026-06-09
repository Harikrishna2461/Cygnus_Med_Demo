# CHIVA Clinical Assistant

A professional offline-shippable chat application for CHIVA venous shunt classification and ligation planning. Clinicians describe patient conditions in natural language and receive structured guidance on shunt type, reasoning, and ligation strategy.

---

## What it does

| Input | Output |
|-------|--------|
| Natural language description of venous blood flow | **Shunt Classification** (Type 1/2A/2B/2C/3/1+2/…) + confidence |
| e.g. *"Reflux at SFJ, flows down GSV to mid-thigh, escapes to tributaries"* | **Classification Reasoning** — step-by-step CHIVA decision logic |
| Works with formal clip notation too | **Ligation Point** — exact anatomical target |
| | **Ligation Strategy** — ordered steps |
| | **CHIVA Hemodynamic Reasoning** |
| | **Clinical Rationale, Follow-up, Complications** |

All conversations are stored and viewable in the **Feedback Log**.

---

## Architecture

```
cmed_demo/
├── backend/
│   ├── app.py                    Flask API server (port 7860)
│   ├── config.py                 Configuration
│   ├── chat_db.py                SQLite — sessions, messages, feedback
│   ├── rag_engine.py             Improved RAG: BM25 + vector + cross-encoder rerank
│   ├── nl_interpreter.py         Natural language → CHIVA clip notation
│   ├── ingest.py                 Knowledge base ingestion pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html                Single-file SPA (no build step)
├── start.bat                     Windows launcher
├── start.sh                      Linux/macOS launcher
└── README.md
```

**RAG pipeline** (per validated retrieval architecture):
1. Vector search — top-50 candidates by cosine similarity (Qdrant + Ollama embeddings)
2. BM25 keyword search — merged and deduplicated with vector results
3. Cross-encoder reranking — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks all candidates
4. Top-5 returned to the ligation planning prompt

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Must be in PATH |
| **Ollama** | Local embedding server — `http://localhost:11434` |
| `llama3.2:1b` model | Pulled in Ollama — used for text embeddings only |
| **Internet access** | For Groq API (LLM inference via `llama-3.3-70b-versatile`) |

Install Ollama: https://ollama.com  
Pull the embedding model:
```bash
ollama pull llama3.2:1b
```

---

## Running locally (development)

```bash
# 1. From the project root (Cygnus_Med_Demo/)
cd cmed_demo

# Windows
start.bat

# Linux / macOS
chmod +x start.sh
./start.sh
```

The script will:
1. Check Python and Ollama
2. Install all Python dependencies
3. Copy `shunt_classification_and_ligation_llm.py` from the parent backend (if needed)
4. Auto-ingest the knowledge base on first run
5. Start the Flask server on `http://127.0.0.1:7860`

Open `http://127.0.0.1:7860` in your browser.

---

## First-time setup details

### Knowledge base ingestion

On first startup, the app automatically runs ingestion. It reads all PDF and DOCX files from `../backend/Knowledgebases/` (the parent project's knowledge base folder) and indexes them in a local Qdrant vector database.

To re-ingest manually:
```bash
cd cmed_demo/backend
python ingest.py --force
```

### Cross-encoder model download

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB) is downloaded from HuggingFace automatically on first use. It is cached in `~/.cache/huggingface/`.

For fully offline deployment, pre-download it once on a connected machine, then ship the HuggingFace cache folder with the app.

---

## Shipping to a clinician's device

### Step 1 — Prepare the zip (on your machine)

```bash
# From Cygnus_Med_Demo/ root:
cd cmed_demo

# Run ingest once so the knowledge base is pre-built
cd backend && python ingest.py && cd ..

# Pre-download the cross-encoder model
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

Now zip the `cmed_demo/` folder along with the pre-built assets:

```
cmed_demo.zip
├── backend/
│   ├── app.py
│   ├── chat_db.py
│   ├── rag_engine.py
│   ├── nl_interpreter.py
│   ├── ingest.py
│   ├── config.py
│   ├── requirements.txt
│   ├── qdrant_storage/          ← pre-built, include this
│   ├── chunks_cache.json        ← pre-built, include this
│   └── shunt_classification_and_ligation_llm.py  ← copy from parent backend
├── frontend/
│   └── index.html
├── start.bat
├── start.sh
└── README.md
```

### Step 2 — Clinician setup (their machine)

1. Install **Python 3.10+**: https://www.python.org/downloads/  
   ✓ Check "Add Python to PATH" during installation.

2. Install and start **Ollama**: https://ollama.com  
   Pull the embedding model:
   ```
   ollama pull llama3.2:1b
   ```

3. Unzip `cmed_demo.zip` anywhere (e.g. Desktop).

4. Copy the HuggingFace cache (if doing fully offline):
   ```
   ~/.cache/huggingface/   →   copy to clinician's machine same path
   ```

5. Double-click `start.bat` (Windows) or run `./start.sh` (Linux/macOS).

6. Open browser at `http://127.0.0.1:7860`.

> **Groq API** — the app uses `llama-3.3-70b-versatile` via Groq for LLM inference. This requires an internet connection. The Groq API key is bundled in `config.py`. To use a different key, set the `GROQ_API_KEY` environment variable before starting.

---

## Usage guide

### Starting a consultation

Type a plain-language description of the patient's venous condition:

> *"There is reflux at the saphenofemoral junction. Blood flows backward down the GSV to mid-thigh, then escapes into a tributary branch."*

> *"There's a perforator entry at the Hunterian level. The GSV trunk appears competent at the SFJ. Tributaries show significant reflux."*

> *"The patient has incompetent SFJ. GSV reflux all the way to the knee. Tributaries fed by GSV are also refluxing. Elimination test showed continued reflux."*

The assistant also accepts formal CHIVA notation if you prefer.

### Conversational follow-up

After the initial analysis, ask follow-up questions:

> *"What does Type 3 mean clinically?"*  
> *"Why do we ligate the tributary first and not the SFJ?"*  
> *"What are the risks if the patient is on anticoagulation?"*

### Feedback log

Click **View Feedback Log** in the sidebar to see all past clinical inputs with their classifications and ligation decisions. This serves as a session audit trail.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (bundled) | Override the Groq API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CMED_PORT` | `7860` | Port for the Flask server |
| `QDRANT_HOST` | (none) | Set for remote Qdrant instance |

---

## Ports

| Service | Port | Notes |
|---------|------|-------|
| CHIVA Clinical Assistant | 7860 | This app |
| Main Cygnus Med app | 5000 | Parent app — no conflict |
| Ollama | 11434 | Embedding server |

---

## Troubleshooting

**"Knowledge base not indexed"** in the status bar  
→ Run `python ingest.py` from `cmed_demo/backend/`

**"Core classification module not found"**  
→ Copy `shunt_classification_and_ligation_llm.py` from `../backend/` into `cmed_demo/backend/`

**Ollama not running**  
→ Start with `ollama serve` in a separate terminal  
→ Pull model: `ollama pull llama3.2:1b`

**Cross-encoder fails to load**  
→ The app falls back to vector-only retrieval automatically  
→ To fix: ensure internet access and re-run the app (model downloads on first use)

**Port 7860 in use**  
→ Set `CMED_PORT=7861` (or any free port) before starting
