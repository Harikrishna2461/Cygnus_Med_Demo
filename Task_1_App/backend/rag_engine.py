"""
Improved RAG Engine — Hybrid BM25 + Vector Search with Cross-Encoder Reranking
-------------------------------------------------------------------------------
Architecture (per the validated retrieval pattern):

  Stage 1 — Recall:  BM25 (keyword) + Vector (semantic) → top-50 merged candidates
  Stage 2 — Rerank:  Cross-encoder (ms-marco-MiniLM-L-6-v2) → top-5 returned

Uses the pre-built ligation_knowledgebase_db_v2 collection (nomic-embed-text,
768-dim) from backend/qdrant_storage — no ingestion step needed.
"""

import logging
import os
import numpy as np
import requests

from qdrant_client import QdrantClient

from config import (
    QDRANT_PATH, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    QDRANT_COLLECTION, EMBEDDING_DIMENSION,
    OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL,
    VECTOR_TOP_K, RERANK_TOP_N,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_ENABLED,
)

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_cross_encoder = None
_bm25_index = None
_bm25_corpus: list[str] = []


# ── Qdrant client ─────────────────────────────────────────────────────────────

def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_HOST:
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY
            )
        else:
            # Read-only open of existing qdrant_storage — do NOT makedirs
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def collection_exists() -> bool:
    try:
        cols = [c.name for c in _get_qdrant().get_collections().collections]
        return QDRANT_COLLECTION in cols
    except Exception:
        return False


def get_collection_size() -> int:
    try:
        info = _get_qdrant().get_collection(QDRANT_COLLECTION)
        return info.points_count or 0
    except Exception:
        return 0


# ── Embedding ─────────────────────────────────────────────────────────────────

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


# ── Cross-encoder ─────────────────────────────────────────────────────────────

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


# ── BM25 index ────────────────────────────────────────────────────────────────

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


def load_bm25_from_qdrant():
    """
    Scroll all text payloads from the existing Qdrant collection and
    build a BM25 index in memory. Called once on app startup.
    No ingest step required — the collection is pre-built.
    """
    if not collection_exists():
        logger.warning(f"Collection '{QDRANT_COLLECTION}' not found — BM25 index skipped.")
        return
    try:
        client = _get_qdrant()
        chunks: list[str] = []
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
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
            logger.info(f"BM25 index loaded from '{QDRANT_COLLECTION}' ({len(chunks)} chunks).")
        else:
            logger.warning("No text payloads found in collection — BM25 index empty.")
    except Exception as e:
        logger.warning(f"BM25 from Qdrant failed: {e}. Continuing without BM25.")


# ── Two-stage retrieval ────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = RERANK_TOP_N) -> list[str]:
    """
    Stage 1a — Vector search:  top-VECTOR_TOP_K by cosine similarity (nomic-embed-text)
    Stage 1b — BM25 search:    top-VECTOR_TOP_K by keyword match (merged, deduplicated)
    Stage 2  — Cross-encoder:  rerank all candidates, return top-k
    """
    client = _get_qdrant()

    if not collection_exists():
        logger.warning(f"Collection '{QDRANT_COLLECTION}' not found.")
        return []

    # Stage 1a: Vector search
    q_vec = get_embedding(query)
    try:
        vector_hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec.tolist(),
            limit=VECTOR_TOP_K,
            with_payload=True,
        )
        candidates: list[str] = [h.payload.get("text", "") for h in vector_hits]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

    # Stage 1b: BM25 merge (if index available)
    if _bm25_index and _bm25_corpus:
        try:
            scores = _bm25_index.get_scores(query.lower().split())
            top_idx = np.argsort(scores)[::-1][:VECTOR_TOP_K]
            bm25_chunks = [_bm25_corpus[i] for i in top_idx if scores[i] > 0]

            seen: set[int] = {hash(c) for c in candidates}
            for chunk in bm25_chunks:
                h = hash(chunk)
                if h not in seen:
                    seen.add(h)
                    candidates.append(chunk)

            candidates = candidates[:VECTOR_TOP_K]
        except Exception as e:
            logger.warning(f"BM25 merge failed: {e}")

    if not candidates:
        return []

    # Stage 2: Cross-encoder reranking
    cross_enc = _get_cross_encoder()
    if cross_enc and len(candidates) > k:
        try:
            pairs = [[query, chunk] for chunk in candidates]
            scores = cross_enc.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            logger.info(
                f"Reranked {len(candidates)} candidates → top-{k} "
                f"(best score: {ranked[0][0]:.3f})"
            )
            return [chunk for _, chunk in ranked[:k]]
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}. Using vector order.")

    return candidates[:k]
