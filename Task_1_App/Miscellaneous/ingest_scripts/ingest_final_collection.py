"""
Ingest final folder data into a new structured Qdrant collection.
Creates a new collection: 'final_structured_rag' with rich metadata.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    QDRANT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEW_COLLECTION_NAME = "final_structured_rag"
FINAL_FOLDER = Path(__file__).parent.parent / "final"


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    if QDRANT_HOST:
        return QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY
        )
    else:
        return QdrantClient(path=QDRANT_PATH)


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama, fallback to hash-based deterministic embedding."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"][0], dtype=np.float32)
    except Exception as e:
        logger.warning(
            f"Ollama unavailable, using deterministic hash-based embeddings: {e}"
        )
        # Generate deterministic embedding from text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**31))
        return np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)


def create_collection(client: QdrantClient) -> None:
    """Create new collection with vector configuration."""
    try:
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if NEW_COLLECTION_NAME in collections:
            logger.info(
                f"Collection '{NEW_COLLECTION_NAME}' already exists. Skipping creation."
            )
            return

        logger.info(
            f"Creating collection '{NEW_COLLECTION_NAME}' with {EMBEDDING_DIMENSION}-dim vectors..."
        )
        client.create_collection(
            collection_name=NEW_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )
        logger.info(f"Collection '{NEW_COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise


def load_jsonl_files() -> list[dict[str, Any]]:
    """Load and combine data from train.jsonl and eval.jsonl."""
    records = []

    for file_path in sorted(FINAL_FOLDER.glob("*.jsonl")):
        logger.info(f"Loading {file_path.name}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        record = json.loads(line)
                        record["_source_file"] = file_path.name
                        record["_line_number"] = line_num
                        records.append(record)
            logger.info(f"  Loaded {len(records)} records total")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    return records


def ingest_records(client: QdrantClient, records: list[dict[str, Any]]) -> None:
    """Ingest records into Qdrant collection."""
    points = []
    batch_size = 64
    skipped = 0

    logger.info(f"Generating embeddings and ingesting {len(records)} records...")

    for idx, record in enumerate(records, 1):
        text = record.get("text", "")
        if not text:
            skipped += 1
            continue

        # Get embedding
        embedding = get_embedding(text)

        # Prepare structured payload
        payload = {
            "text": text,
            "token_count": record.get("token_count", 0),
            "source_book": record.get("metadata", {}).get("source_book", ""),
            "chapter": record.get("metadata", {}).get("chapter", ""),
            "section": record.get("metadata", {}).get("section", ""),
            "position": record.get("metadata", {}).get("position", 0),
            "high_value": record.get("metadata", {}).get("high_value", False),
            "source_file": record.get("_source_file", ""),
            "line_number": record.get("_line_number", 0),
        }

        # Create point
        point = PointStruct(
            id=idx - skipped,
            vector=embedding.tolist(),
            payload=payload,
        )
        points.append(point)

        # Batch upsert
        if len(points) >= batch_size:
            try:
                client.upsert(
                    collection_name=NEW_COLLECTION_NAME,
                    points=points,
                )
                logger.info(f"  Upserted batch ({idx}/{len(records)} processed, {len(points)} points)")
                points = []
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")

    # Final batch
    if points:
        try:
            client.upsert(
                collection_name=NEW_COLLECTION_NAME,
                points=points,
            )
            logger.info(f"  Final upsert: {len(points)} points")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")

    logger.info(f"Ingestion complete. Total records: {len(records) - skipped} (skipped: {skipped})")


def verify_collection(client: QdrantClient) -> None:
    """Verify the collection and print stats."""
    try:
        info = client.get_collection(NEW_COLLECTION_NAME)
        logger.info(f"\n{'='*60}")
        logger.info(f"Collection: {NEW_COLLECTION_NAME}")
        logger.info(f"Points count: {info.points_count}")
        logger.info(f"Vector size: {EMBEDDING_DIMENSION}")
        logger.info(f"Vector count: {len(info.vectors_count) if hasattr(info, 'vectors_count') else 'N/A'}")
        logger.info(f"{'='*60}\n")

        # Sample a few records
        sample = client.scroll(
            collection_name=NEW_COLLECTION_NAME,
            limit=2,
            with_payload=True,
            with_vectors=False,
        )
        if sample[0]:
            logger.info("Sample record structure:")
            for point in sample[0]:
                logger.info(f"  ID: {point.id}")
                logger.info(f"  Payload keys: {list(point.payload.keys())}")
                break

    except Exception as e:
        logger.error(f"Verification failed: {e}")


def main():
    """Main ingestion workflow."""
    logger.info("Starting collection ingestion...")

    # Initialize client
    client = get_qdrant_client()

    # Create collection
    create_collection(client)

    # Load data
    records = load_jsonl_files()
    if not records:
        logger.error("No records loaded. Exiting.")
        return

    # Ingest
    ingest_records(client, records)

    # Verify
    verify_collection(client)

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
