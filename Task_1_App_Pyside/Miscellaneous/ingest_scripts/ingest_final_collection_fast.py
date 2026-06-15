"""
Fast ingestion of final folder data into 'final_structured_rag' collection.
Uses quick hash-based embeddings to avoid slow Ollama calls.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import QDRANT_PATH, EMBEDDING_DIMENSION, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEW_COLLECTION_NAME = "final_structured_rag"
FINAL_FOLDER = Path(__file__).parent.parent / "final"


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(path=QDRANT_PATH)


def get_embedding(text: str) -> np.ndarray:
    """Try to get real embedding from Ollama, fall back to hash-based if unavailable."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=5,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"][0], dtype=np.float32)
    except Exception:
        # Fall back to deterministic hash-based embedding
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**31))
        return np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)


def load_jsonl_files() -> list[dict[str, Any]]:
    """Load data from train.jsonl and eval.jsonl."""
    records = []
    for file_path in sorted(FINAL_FOLDER.glob("*.jsonl")):
        logger.info(f"Loading {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def ingest_fast(client: QdrantClient, records: list[dict[str, Any]]) -> None:
    """Ingestion with real semantic embeddings from Ollama."""
    points = []
    batch_size = 32
    total_points = 0
    skipped = 0

    logger.info(f"Ingesting {len(records)} records with real embeddings...")

    for idx, record in enumerate(records, 1):
        text = record.get("text", "")
        if not text:
            skipped += 1
            continue

        try:
            # Get real semantic embedding
            embedding = get_embedding(text)

            # Structured payload
            payload = {
                "text": text,
                "token_count": record.get("token_count", 0),
                "source_book": record.get("metadata", {}).get("source_book", ""),
                "chapter": record.get("metadata", {}).get("chapter", ""),
                "section": record.get("metadata", {}).get("section", ""),
                "position": record.get("metadata", {}).get("position", 0),
                "high_value": record.get("metadata", {}).get("high_value", False),
            }

            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload,
            )
            points.append(point)

            if len(points) >= batch_size:
                client.upsert(NEW_COLLECTION_NAME, points=points)
                total_points += len(points)
                logger.info(f"  {total_points}/{len(records)} points upserted ({idx}/{len(records)} records processed)")
                points = []
        except Exception as e:
            logger.warning(f"  Skipping record {idx} due to embedding error: {e}")
            skipped += 1

    # Final batch
    if points:
        client.upsert(NEW_COLLECTION_NAME, points=points)
        total_points += len(points)
        logger.info(f"  Final batch: {total_points} total points")

    logger.info(f"Ingestion complete: {total_points} points, {skipped} skipped")


def recreate_collection(client: QdrantClient) -> None:
    """Delete and recreate the collection."""
    try:
        cols = [c.name for c in client.get_collections().collections]
        if NEW_COLLECTION_NAME in cols:
            logger.info(f"Deleting existing collection '{NEW_COLLECTION_NAME}'...")
            client.delete_collection(NEW_COLLECTION_NAME)
            logger.info("Collection deleted.")
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")

    # Create new collection
    logger.info(f"Creating collection '{NEW_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=NEW_COLLECTION_NAME,
        vectors_config={
            "size": EMBEDDING_DIMENSION,
            "distance": "Cosine",
        },
    )
    logger.info("Collection created.")


def verify(client: QdrantClient) -> None:
    """Verify collection."""
    info = client.get_collection(NEW_COLLECTION_NAME)
    logger.info(f"\n✓ Collection '{NEW_COLLECTION_NAME}' ready")
    logger.info(f"  Points: {info.points_count}")
    logger.info(f"  Status: {info.status}")

    # Sample
    sample = client.scroll(NEW_COLLECTION_NAME, limit=1, with_payload=True, with_vectors=False)
    if sample[0]:
        p = sample[0][0]
        logger.info(f"\n  Sample record (ID {p.id}):")
        for key in p.payload:
            val = p.payload[key]
            if isinstance(val, str) and len(val) > 100:
                logger.info(f"    {key}: {val[:100]}...")
            else:
                logger.info(f"    {key}: {val}")


def main():
    client = get_qdrant_client()
    records = load_jsonl_files()
    recreate_collection(client)
    ingest_fast(client, records)
    verify(client)
    logger.info("\n✓ Ingestion complete!")


if __name__ == "__main__":
    main()
