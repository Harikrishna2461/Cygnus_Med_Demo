import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index', 'medical_index.faiss')
FAISS_METADATA_PATH = os.path.join(BASE_DIR, 'faiss_index', 'metadata.pkl')
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'sample_medical_text.txt')

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"  # Ultra-fast 1B model with MLX (7x faster)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_KEEP_ALIVE = "20m"  # Keep model in GPU memory for faster responses

# Chunking parameters
CHUNK_SIZE = 400  # tokens (approximate)
CHUNK_OVERLAP = 50  # tokens

# FAISS parameters
EMBEDDING_DIMENSION = 768  # Nomic-embed-text embeddings dimension
MAX_RETRIEVAL_RESULTS = 3

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, 'app.log')