# RAG Retrieval Analysis - CHIVA Knowledge Base

## Summary
The RAG system has inconsistent retrieval quality for CHIVA queries. Some queries return relevant information while others return fallback responses.

## Observed Behavior

### Query 1: "What is CHIVA?"
- **Cross-Encoder Score**: -3.360 (NEGATIVE - not relevant)
- **Response Type**: Fallback/Generic  
- **Issue**: Query too generic, embeddings may not match well

### Query 2: "What is CHIVA Shunt Type 1?"
- **Cross-Encoder Score**: -9.129 (HIGHLY NEGATIVE - not relevant)
- **Response Type**: Fallback response saying "knowledge base does not contain..."
- **Issue**: Despite CHIVA data existing in collection, retrieval fails

### Query 3: "CHIVA technique"
- **Cross-Encoder Score**: 4.954 (POSITIVE - relevant)
- **Response Type**: Detailed CHIVA information with citations
- **Issue**: Works well, provides specific references

---

## Root Causes Identified

### 1. **Vector Embedding Sensitivity**
The `nomic-embed-text` model is sensitive to query phrasing:
- "CHIVA?" vs "CHIVA technique" → Different embeddings → Different retrieval
- Question format vs noun phrase → Different semantic space

### 2. **Cross-Encoder Score Threshold Issue**
- **Threshold seems to be around 0.0** (positive = relevant, negative = not)
- Query 1: -3.360 → rejected (but CHIVA data exists!)
- Query 3: 4.954 → accepted
- **Problem**: Some CHIVA chunks are being scored as irrelevant

### 3. **Hybrid Search Not Compensating**
Even though hybrid search combines:
- **Vector search** (semantic): Finding chunks similar to "What is CHIVA shunt type 1?"
- **BM25 search** (keyword): Finding chunks containing "CHIVA", "shunt", "type"

**Both are returning low-scoring candidates** that the cross-encoder rejects.

---

## Retrieval Pipeline Flow

```
Query: "What is CHIVA Shunt Type 1?"
    ↓
1A. Vector Search (Semantic Embedding)
    - Finds 50 chunks semantically similar to query
    - Uses nomic-embed-text embeddings
    - Problem: May not find CHIVA-specific chunks if embedding is off
    ↓
1B. BM25 Search (Keyword Matching)
    - Finds 50 chunks containing keywords: "CHIVA", "shunt", "type"
    - Merges with vector results (deduplication)
    - Total: ~50 candidates
    ↓
2. Cross-Encoder Reranking
    - Scores [query, chunk] pairs with ms-marco-MiniLM-L-6-v2
    - Expected: CHIVA chunks get positive scores (>0)
    - Actual: All chunks get negative scores (<0)
    - Return: Top-5 (which are still negative = not relevant)
    ↓
3. LLM Response
    - Receives 5 low-scoring chunks
    - Interprets as "knowledge base doesn't have info"
    - Returns fallback response
```

---

## Why "CHIVA technique" Works But "CHIVA Shunt Type 1?" Doesn't

### Query: "CHIVA technique"
- **Embedding match**: Very good (exact phrase in training data context)
- **BM25 match**: Excellent (both "CHIVA" and "technique" match chunks)
- **Cross-encoder score**: 4.954 (POSITIVE - accepts chunks)
- **Result**: Returns detailed information ✓

### Query: "What is CHIVA Shunt Type 1?"
- **Embedding match**: May be offset due to question format + additional complexity
- **BM25 match**: Good (contains "CHIVA", "shunt", "type") but chunks may be less relevant
- **Cross-encoder score**: -9.129 (NEGATIVE - rejects all chunks)
- **Result**: Returns "knowledge base doesn't contain..." ✗

---

## Evidence: CHIVA Data Exists in Collection

From earlier confirmation:
- **Total CHIVA references in training data**: 297 occurrences
- **Collection size**: 2,653 points (all ingested successfully)
- **Proof it works**: Query 3 ("CHIVA technique") retrieves and returns specific information

---

## Recommendations

### 1. **Lower Cross-Encoder Score Threshold**
- Current: Seems to require score > 0
- Proposed: Accept scores > -5.0 for medical queries
- Rationale: Medical domain has different relevance patterns than general text

### 2. **Adjust Query Preprocessing**
- Convert questions to statement form for embedding
- "What is CHIVA Shunt Type 1?" → "CHIVA shunt type 1" for embedding
- Improves semantic matching

### 3. **Strengthen BM25 Component**
- Add "question words" to stopword removal
- Increase BM25 weight in hybrid search
- Ensure keyword matches boost candidates more

### 4. **Fine-tune Cross-Encoder**
- Current model (ms-marco-MiniLM-L-6-v2) trained on general web data
- Possible: Use medical domain specific cross-encoder
- Or: Adjust scoring function for medical context

---

## Current System State

✅ **Working**:
- Real Ollama embeddings in collection
- Hybrid search (vector + BM25) functioning
- Cross-encoder installed and running
- Some queries retrieve CHIVA info correctly

❌ **Not Working**:
- Question-format queries to CHIVA don't retrieve info
- Cross-encoder score threshold filtering out valid medical content
- Inconsistent behavior based on query phrasing

⚠️ **Needs Investigation**:
- Why different query formats produce vastly different scores
- Whether cross-encoder needs retraining or recalibration for medical domain
