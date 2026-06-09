"""
General Medical Chat with Domain Guardrails and RAG
----------------------------------------------------
Allows users to ask general medical/surgical questions from the medical knowledge base.
Uses final_structured_rag collection with hybrid search + cross-encoder reranking.
Includes guardrails to keep conversation domain-specific.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
import numpy as np
import requests

from config import (
    QDRANT_PATH, EMBEDDING_DIMENSION,
    OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL,
    VECTOR_TOP_K, RERANK_TOP_N,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_ENABLED,
)

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_cross_encoder = None
_bm25_index = None
_bm25_corpus: list[str] = []

GENERAL_COLLECTION = "final_structured_rag"


def set_qdrant_client(client: QdrantClient):
    """Set the shared Qdrant client (called by app.py during startup)."""
    global _qdrant_client
    _qdrant_client = client

# Domain keywords for guardrails
MEDICAL_KEYWORDS = {
    "medical", "surgical", "surgery", "treatment", "diagnosis", "disease", "condition",
    "symptom", "vein", "venous", "vessel", "artery", "ultrasound", "imaging", "scan",
    "biopsy", "procedure", "operation", "patient", "clinician", "doctor", "physician",
    "anatomy", "pathology", "physiology", "pharmacology", "drug", "medication",
    "ligation", "shunt", "insufficiency", "thrombosis", "embolism", "reflux",
    "classification", "evaluation", "assessment", "protocol", "guideline", "evidence",
    "valve", "flow", "blood", "circulation", "perfusion", "ischemia", "hemorrhage",
    "chiva", "saphenous", "femoral", "popliteal", "varicose", "venography", "doppler",
}

REJECT_KEYWORDS = {
    "joke", "poem", "song", "story", "code", "programming", "python", "javascript",
    "politics", "sports", "weather", "recipe", "movie", "game", "music", "video",
}


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def collection_exists() -> bool:
    try:
        cols = [c.name for c in _get_qdrant().get_collections().collections]
        return GENERAL_COLLECTION in cols
    except Exception as e:
        logger.error(f"Error checking collections: {e}")
        return False


def get_collection_size() -> int:
    try:
        info = _get_qdrant().get_collection(GENERAL_COLLECTION)
        return info.points_count or 0
    except Exception:
        return 0


def get_embedding(text: str) -> np.ndarray:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"][0], dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None and CROSS_ENCODER_ENABLED:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info("Cross-encoder ready.")
        except Exception as e:
            logger.warning(
                f"Cross-encoder not available ({e}). Falling back to vector-only retrieval."
            )
    return _cross_encoder


def build_bm25_index(chunks: list[str]):
    global _bm25_index, _bm25_corpus
    if not chunks:
        return
    try:
        from rank_bm25 import BM25Okapi
        _bm25_corpus = chunks
        tokenized = [chunk.lower().split() for chunk in chunks]
        _bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built ({len(chunks)} chunks).")
    except Exception as e:
        logger.warning(f"BM25 unavailable ({e}). Using vector-only.")


def load_bm25_from_collection():
    """Build BM25 index from the final_structured_rag collection."""
    if not collection_exists():
        logger.warning(f"Collection '{GENERAL_COLLECTION}' not found — BM25 index skipped.")
        return
    try:
        client = _get_qdrant()
        chunks: list[str] = []
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=GENERAL_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                text = (point.payload or {}).get("text", "")
                if text:
                    chunks.append(text)
            if next_offset is None:
                break
            offset = next_offset

        if chunks:
            build_bm25_index(chunks)
            logger.info(f"BM25 index loaded from '{GENERAL_COLLECTION}' ({len(chunks)} chunks).")
        else:
            logger.warning("No text payloads found in collection — BM25 index empty.")
    except Exception as e:
        logger.warning(f"BM25 from Qdrant failed: {e}. Continuing without BM25.")


def is_domain_relevant(query: str) -> bool:
    """Check if query is about medical/surgical topics."""
    query_lower = query.lower()

    # Check for explicit rejection keywords
    if any(keyword in query_lower for keyword in REJECT_KEYWORDS):
        return False

    # Check for medical keywords
    has_medical_keyword = any(keyword in query_lower for keyword in MEDICAL_KEYWORDS)

    return has_medical_keyword


def retrieve_general_context(query: str, k: int = RERANK_TOP_N) -> list[str]:
    """
    Hybrid retrieval with reranking:
    Stage 1A — Vector search: semantic similarity
    Stage 1B — BM25 search: keyword matching (merged, deduplicated)
    Stage 2  — Cross-encoder: rerank all candidates, return top-k
    """
    client = _get_qdrant()

    if not collection_exists():
        logger.warning(f"Collection '{GENERAL_COLLECTION}' not found.")
        return []

    # Stage 1A: Vector search with semantic embeddings
    q_vec = get_embedding(query)
    candidates: list[str] = []
    try:
        vector_hits = client.query_points(
            collection_name=GENERAL_COLLECTION,
            query=q_vec.tolist(),
            limit=VECTOR_TOP_K,
            with_payload=True,
        )
        candidates = [h.payload.get("text", "") for h in vector_hits.points]
        logger.info(f"Vector search found {len(candidates)} candidates")
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")

    # Stage 1B: BM25 merge (keyword matching)
    if _bm25_index and _bm25_corpus:
        try:
            scores = _bm25_index.get_scores(query.lower().split())
            top_idx = np.argsort(scores)[::-1][:VECTOR_TOP_K]
            bm25_chunks = [_bm25_corpus[i] for i in top_idx if scores[i] > 0]

            logger.info(f"BM25 found {len(bm25_chunks)} matching chunks")

            seen: set[int] = {hash(c) for c in candidates}
            for chunk in bm25_chunks:
                h = hash(chunk)
                if h not in seen:
                    seen.add(h)
                    candidates.append(chunk)

            candidates = candidates[:VECTOR_TOP_K]
            logger.info(f"After merge: {len(candidates)} total candidates")
        except Exception as e:
            logger.warning(f"BM25 merge failed: {e}")

    if not candidates:
        logger.warning("No candidates from vector or BM25 search")
        return []

    # Stage 2: Cross-encoder reranking with relevance threshold
    cross_enc = _get_cross_encoder()
    if cross_enc:
        try:
            pairs = [[query, chunk] for chunk in candidates]
            scores = cross_enc.predict(pairs)

            # Filter by threshold - only keep relevant chunks
            threshold = -2.0  # Medical domain threshold
            ranked = [(score, chunk) for score, chunk in zip(scores, candidates) if score >= threshold]
            ranked.sort(key=lambda x: x[0], reverse=True)

            logger.info(
                f"Cross-encoder filtered {len(candidates)} → {len(ranked)} candidates "
                f"(threshold: {threshold}, best: {ranked[0][0]:.2f})" if ranked else "no candidates above threshold"
            )

            if ranked:
                return [chunk for _, chunk in ranked[:k]]
            else:
                # Fallback: return top unfiltered if nothing passes threshold
                logger.warning(f"No candidates above threshold {threshold}, returning top-{k} unfiltered")
                return candidates[:k]
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}. Using vector-BM25 order.")
            return candidates[:k]

    return candidates[:k]
