import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'sample_medical_text.txt')

# Legacy FAISS paths (read-only, used only during migration)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index', 'medical_index.faiss')
FAISS_METADATA_PATH = os.path.join(BASE_DIR, 'faiss_index', 'metadata.pkl')

# Qdrant (local file-based — no Docker required; swap host/port/api_key for cloud)
QDRANT_PATH = os.path.join(BASE_DIR, 'qdrant_storage')  # local persistent storage
QDRANT_HOST = os.getenv("QDRANT_HOST", None)            # set for remote/cloud
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)      # set for Qdrant Cloud
QDRANT_COLLECTION = "medical_knowledge"

# LLM Configuration — Ollama (local, used for embeddings only)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"           # kept for embeddings
OLLAMA_EMBEDDING_MODEL = "llama3.2:1b"
OLLAMA_KEEP_ALIVE = "20m"

# Groq API — used for ALL LLM inference (70B model, high quality)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "REDACTED")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunking parameters
CHUNK_SIZE = 400   # approximate word count per chunk
CHUNK_OVERLAP = 50

# Embedding / retrieval
EMBEDDING_DIMENSION = 2048  # llama3.2:1b output dimension
MAX_RETRIEVAL_RESULTS = 3

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, 'app.log')