# Dual-Mode Medical Assistant Setup

## Overview
The application now supports two distinct chat modes:

### 1. **Clinical Decision Support Mode** (`/clinical`)
- Specialized for venous shunt classification and ligation planning
- Uses existing `ligation_knowledgebase_db_v2` collection
- Includes clinical interpretation pipeline
- Features: Classification results, ligation strategies, clinical reasoning

### 2. **General Medical Chat Mode** (`/general`)
- Allows users to ask any medical/surgical questions
- Uses new `final_structured_rag` collection (2,653 medical text chunks)
- Implements hybrid retrieval: BM25 + Vector search + Cross-encoder reranking
- Features: Browse knowledge base, research topics, get general medical knowledge
- **Includes domain guardrails**: Politely refuses off-topic questions

## Components

### New Files Created

1. **`backend/general_chat_engine.py`**
   - RAG engine for general medical questions
   - Hybrid search with BM25 and vector embeddings
   - Cross-encoder reranking for better relevance
   - Domain guardrails with keyword-based filtering
   - Functions:
     - `retrieve_general_context()`: Two-stage retrieval
     - `is_domain_relevant()`: Guardrail check
     - `load_bm25_from_collection()`: Build BM25 index

2. **`backend/ingest_final_collection_fast.py`** (helper script)
   - Fast ingestion of final folder JSONL files
   - Creates and populates `final_structured_rag` collection
   - Status: ✓ All 2,653 records fully ingested

### Modified Files

1. **`backend/app.py`**
   - Landing page route (`/`) with mode selection UI
   - Clinical mode route (`/clinical`)
   - General mode route (`/general`)
   - New API endpoint: `POST /api/general-chat`
   - Startup improvements: Load both RAG collections

2. **`frontend/index.html`**
   - Mode detection from URL pathname
   - Dynamic mode indicator in sidebar
   - Mode-aware session prompts
   - Mode-specific API routing
   - Back button to landing page

## Data Structure

### final_structured_rag Collection
Each record contains:
```json
{
  "text": "Medical reference content...",
  "token_count": 843,
  "source_book": "author-year-title",
  "chapter": "Chapter Name",
  "section": "Section Name",
  "position": 0,
  "high_value": true/false,
  "vector": [768-dimensional embedding]
}
```

**Total**: 2,653 points
- Training data: 2,637 chunks
- Evaluation data: 16 chunks

## Domain Guardrails

### Allowed Topics
Medical and surgical topics including:
- Anatomy, pathology, physiology
- Diseases, symptoms, conditions
- Diagnosis and treatment procedures
- Medications and pharmacology
- Ultrasound, imaging, scans
- Venous insufficiency, thrombosis, embolism
- Clinical guidelines and protocols

### Rejected Topics
The system politely refuses:
- Jokes, poems, creative writing
- Code, programming languages
- Sports, politics, entertainment
- Recipes, weather, general knowledge unrelated to medicine

### How It Works
1. **Keyword-based pre-filter**: Checks query against medical keywords
2. **Polite refusal**: If off-topic, responds with educational message
3. **User guidance**: Explains what topics are available
4. **Graceful handling**: Allows user to rephrase with medical context

## API Endpoints

### Clinical Mode
```
POST /api/chat
{
  "session_id": "string",
  "message": "string"
}
```

### General Mode
```
POST /api/general-chat
{
  "session_id": "string",
  "message": "string",
  "mode": "general"
}
```

### Shared Endpoints
- `GET /api/status` - System status
- `GET /api/sessions` - List sessions
- `POST /api/session` - Create new session
- `GET /api/session/{id}/messages` - Get chat history
- `POST /api/feedback` - Submit feedback

## Retrieval Pipeline (General Mode)

### Stage 1a: Vector Search
- Embed query using nomic-embed-text (768-dim)
- Search `final_structured_rag` with cosine similarity
- Retrieve top-50 candidates

### Stage 1b: BM25 Keyword Search
- Tokenize and score documents with BM25
- Retrieve top-50 candidates
- Merge with vector results (deduplicate)

### Stage 2: Cross-Encoder Reranking
- Use `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Score all candidates for relevance to query
- Return top-5 most relevant results

## Workflow

### Landing Page (`/`)
1. User sees two mode options
2. Clicks "Enter Clinical Mode" or "Enter General Mode"
3. Directed to appropriate interface

### General Chat Workflow
1. User enters query in chat
2. Guardrail check: Is it medically relevant?
   - ❌ No → Polite refusal + guidance
   - ✓ Yes → Continue
3. Retrieve top-5 contexts using hybrid search
4. Call LLM (llama-3.3-70b-versatile) with context
5. Generate response based on knowledge base
6. Display answer with context count
7. Save to chat history

### Clinical Mode (Unchanged)
- Uses existing pipeline
- Classification + ligation planning
- Specialized clinical analysis

## Configuration

Key settings in `backend/config.py`:
```python
QDRANT_COLLECTION = "ligation_knowledgebase_db_v2"  # Clinical
# (General mode uses "final_structured_rag")

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

VECTOR_TOP_K = 50        # Pull top-50 candidates
RERANK_TOP_N = 5         # Return top-5 after reranking

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_ENABLED = True

GROQ_MODEL = "llama-3.3-70b-versatile"
```

## Testing the Setup

### 1. Verify Collections
```bash
cd backend && python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(path="qdrant_storage")
cols = [c.name for c in client.get_collections().collections]
print("Collections:", cols)
EOF
```

Expected: Both `ligation_knowledgebase_db_v2` and `final_structured_rag`

### 2. Start Backend
```bash
cd backend && python app.py
```

### 3. Test Landing Page
- Navigate to `http://localhost:7860`
- See two buttons (Clinical / General)
- Click each to test mode routing

### 4. Test General Chat
- Click "General Medical Chat"
- Try medical question: "What is Doppler ultrasound used for?"
- Try off-topic: "Tell me a joke"
  - Should be refused politely

### 5. Test Clinical Mode
- Click "Clinical Decision Support"
- Describe a venous condition
- Should get classification + ligation plan

## Performance Notes

- **Ingestion**: All 2,653 records ingested in ~8 seconds
- **BM25 Building**: ~0.5-1s per collection (built at startup)
- **Query Time**: ~1-2s (vector search + reranking)
- **LLM Response**: ~2-3s (llama-3.3-70b)

## Future Improvements

1. Persistent BM25 index (skip rebuilding)
2. Caching for frequent queries
3. User feedback loop for guardrail improvement
4. Multi-language support
5. Citation tracking (source documents)
6. Advanced filtering by source_book/chapter
