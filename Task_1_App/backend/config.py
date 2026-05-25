import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent

# Parent backend — for importing shared prompts/rules (dev environment)
PARENT_BACKEND_DIR = BASE_DIR.parent.parent / "backend"

# Qdrant — local copy at cmed_demo/backend/qdrant_storage
# ligation_knowledgebase_db_v2 (nomic-embed-text, 768-dim, Apr 23) is newer than
# ligation_knowledgebase_db (llama3.2:1b, 2048-dim, Apr 21) so we use v2.
QDRANT_PATH = str(BASE_DIR / "qdrant_storage")
QDRANT_HOST = os.getenv("QDRANT_HOST", None)
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = "ligation_knowledgebase_db_v2"

# Ollama — local embeddings only.
# nomic-embed-text is required to query ligation_knowledgebase_db_v2 (768-dim).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

# Groq LLM — remote inference
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Improved RAG retrieval (cross-encoder reranking pipeline)
VECTOR_TOP_K = 50        # Pull top-50 by vector similarity
RERANK_TOP_N = 15        # Return top-15 after cross-encoder reranking (more context for subtypes)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_ENABLED = True

# App server
PORT = int(os.getenv("CMED_PORT", 7860))
HOST = "127.0.0.1"
DEBUG = False

# SQLite for chat history and feedback
DB_PATH = str(BASE_DIR / "chat_history.db")

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "cmed_chat.log")
