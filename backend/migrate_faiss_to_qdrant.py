"""
migrate_faiss_to_qdrant.py
──────────────────────────
One-shot migration: reads every vector + text chunk from the existing FAISS
index and uploads them to Qdrant (local file-based storage).

Run once:
    cd backend
    python migrate_faiss_to_qdrant.py

After running you can keep the faiss_index/ directory as a backup or delete it.
"""

import os
import sys
import pickle
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    QDRANT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    LOG_FILE,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_HOST:
        logger.info(f"Connecting to remote Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    os.makedirs(QDRANT_PATH, exist_ok=True)
    logger.info(f"Using local Qdrant at {QDRANT_PATH}")
    return QdrantClient(path=QDRANT_PATH)


def migrate():
    logger.info("=" * 70)
    logger.info("FAISS → QDRANT MIGRATION")
    logger.info("=" * 70)

    # ── 1. Load FAISS index ───────────────────────────────────────────────────
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
        sys.exit(1)
    if not os.path.exists(FAISS_METADATA_PATH):
        logger.error(f"Metadata pickle not found at {FAISS_METADATA_PATH}")
        sys.exit(1)

    import faiss  # only needed here — not a runtime dependency after migration
    logger.info(f"\n[1/4] Loading FAISS index from {FAISS_INDEX_PATH} ...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    n_vectors = index.ntotal
    logger.info(f"  Vectors in index : {n_vectors}")
    logger.info(f"  Dimension        : {index.d}")

    # ── 2. Reconstruct ALL vectors from FAISS ────────────────────────────────
    logger.info(f"\n[2/4] Reconstructing {n_vectors} vectors from FAISS (lossless)...")
    all_vectors = np.zeros((n_vectors, index.d), dtype=np.float32)
    for i in range(n_vectors):
        all_vectors[i] = index.reconstruct(i)
    logger.info(f"  ✓ All {n_vectors} vectors extracted — shape {all_vectors.shape}")

    # ── 3. Load text chunks (metadata) ───────────────────────────────────────
    logger.info(f"\n[3/4] Loading metadata from {FAISS_METADATA_PATH} ...")
    with open(FAISS_METADATA_PATH, 'rb') as f:
        chunks = pickle.load(f)

    if isinstance(chunks, list):
        logger.info(f"  ✓ {len(chunks)} text chunks loaded")
    else:
        logger.error(f"Unexpected metadata type: {type(chunks)}")
        sys.exit(1)

    if len(chunks) != n_vectors:
        logger.warning(
            f"  ⚠ Vector count ({n_vectors}) ≠ chunk count ({len(chunks)}). "
            "Truncating to min."
        )
    n_points = min(n_vectors, len(chunks))

    # ── 4. Upsert into Qdrant ────────────────────────────────────────────────
    logger.info(f"\n[4/4] Upserting {n_points} points into Qdrant...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = get_qdrant_client()

    # Create / recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        existing_info = client.get_collection(QDRANT_COLLECTION)
        existing_count = existing_info.points_count
        logger.info(
            f"  Collection '{QDRANT_COLLECTION}' already exists "
            f"with {existing_count} points — dropping and recreating."
        )
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=index.d, distance=Distance.COSINE),
    )
    logger.info(f"  Created collection '{QDRANT_COLLECTION}' (dim={index.d}, Cosine)")

    # Build points
    points = [
        PointStruct(
            id=i,
            vector=all_vectors[i].tolist(),
            payload={"text": str(chunks[i]), "chunk_index": i, "source": "faiss_migration"},
        )
        for i in range(n_points)
    ]

    # Batch upsert
    batch_size = 100
    for start in range(0, n_points, batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        done = min(start + batch_size, n_points)
        logger.info(f"  Upserted {done}/{n_points} points...")

    # ── Verify ────────────────────────────────────────────────────────────────
    info = client.get_collection(QDRANT_COLLECTION)
    final_count = info.points_count
    logger.info(f"\n{'='*70}")
    if final_count == n_points:
        logger.info(f"✓ MIGRATION SUCCESSFUL — {final_count}/{n_points} points in Qdrant")
    else:
        logger.error(
            f"✗ MIGRATION INCOMPLETE — expected {n_points}, got {final_count} in Qdrant"
        )

    logger.info(f"  Collection : {QDRANT_COLLECTION}")
    logger.info(f"  Points     : {final_count}")
    logger.info(f"  Dimension  : {index.d}")
    logger.info(f"  Storage    : {QDRANT_PATH}")
    logger.info(f"{'='*70}")
    logger.info("\nFAISS files preserved at:")
    logger.info(f"  {FAISS_INDEX_PATH}")
    logger.info(f"  {FAISS_METADATA_PATH}")
    logger.info("You may delete them once satisfied with the migration.")

    return final_count == n_points


if __name__ == '__main__':
    success = migrate()
    sys.exit(0 if success else 1)
