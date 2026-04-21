"""
Shunt Classification Knowledge Base Ingestion
==============================================
Ingests:
  1. Task_1_Shunt_Classification_Knowledgebase.pdf  (all pages)
  2. chiva_rules.txt

Into a dedicated Qdrant collection: shunt_classification_db

Run:
    cd backend
    python ingest_shunt_classification.py
"""

import os
import time
import base64
import logging
import numpy as np
import requests
from pathlib import Path
from groq import Groq
import fitz  # pymupdf — renders image-based PDF pages
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_SHUNT_COLLECTION,
    EMBEDDING_DIMENSION,
    GROQ_API_KEY,
    LOG_FILE,
    LOG_LEVEL,
)

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Task_1_Shunt_Classification_Knowledgebase.pdf"
RULES_PATH = BASE_DIR / "chiva_rules.txt"
SHUNT_BOOK_PATH = BASE_DIR.parent / "Shunt_Book_8.pdf"


# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    if QDRANT_HOST:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    os.makedirs(QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


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


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using Groq vision OCR (handles image-based/scanned PDFs)."""
    logger.info(f"Extracting PDF (vision OCR): {pdf_path.name}")
    groq_client = Groq(api_key=GROQ_API_KEY)
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    logger.info(f"  {total} pages — sending to Groq vision model")

    all_text = []
    for i, page in enumerate(doc):
        if (i + 1) % 5 == 0 or (i + 1) == total:
            logger.info(f"  OCR page {i + 1}/{total}...")
        # Render page at 2x zoom for better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe all text visible on this page exactly as written. "
                                    "Preserve headings, lists, tables, and medical terminology. "
                                    "Output only the transcribed text — no commentary."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=2048,
            )
            page_text = resp.choices[0].message.content or ""
            all_text.append(page_text)
        except Exception as e:
            logger.warning(f"  OCR failed for page {i + 1}: {e}")
            all_text.append("")

        time.sleep(0.3)  # rate-limit buffer

    doc.close()
    full_text = "\n".join(all_text)
    logger.info(f"  Extracted {len(full_text):,} characters via OCR")
    return full_text


def load_rules_text(rules_path: Path) -> str:
    logger.info(f"Loading rules: {rules_path.name}")
    with open(rules_path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"  {len(text):,} characters")
    return text


# ── Ingestion pipeline ────────────────────────────────────────────────────────

def ingest():
    logger.info("=" * 70)
    logger.info(f"SHUNT CLASSIFICATION INGESTION → {QDRANT_SHUNT_COLLECTION}")
    logger.info("=" * 70)

    # 1. Load sources
    sources: list[tuple[str, str]] = []

    if PDF_PATH.exists():
        pdf_text = extract_pdf_text(PDF_PATH)
        if pdf_text.strip():
            sources.append((PDF_PATH.name, pdf_text))
    else:
        logger.error(f"PDF not found: {PDF_PATH}")

    if RULES_PATH.exists():
        rules_text = load_rules_text(RULES_PATH)
        if rules_text.strip():
            sources.append((RULES_PATH.name, rules_text))
    else:
        logger.error(f"Rules file not found: {RULES_PATH}")

    if SHUNT_BOOK_PATH.exists():
        book_text = extract_pdf_text(SHUNT_BOOK_PATH)
        if book_text.strip():
            sources.append((SHUNT_BOOK_PATH.name, book_text))
    else:
        logger.warning(f"Shunt_Book_8.pdf not found at {SHUNT_BOOK_PATH} — skipping")

    if not sources:
        raise RuntimeError("No source documents found — aborting.")

    # 2. Combine and chunk per source (preserves source metadata)
    # chiva_rules.txt uses fine-grained chunks (80 words) so each Case / shunt
    # type definition lands in its own retrievable chunk.  PDF uses default size.
    all_chunks: list[dict] = []
    for source_name, text in sources:
        if source_name == RULES_PATH.name:
            chunks = split_into_chunks(text, chunk_size=80, overlap=10)
        else:
            chunks = split_into_chunks(text)
        logger.info(f"  {source_name}: {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": source_name,
                "chunk_index": idx,
            })

    logger.info(f"\nTotal chunks: {len(all_chunks)}")

    # 3. Embed
    logger.info(f"\nEmbedding {len(all_chunks)} chunks with {OLLAMA_EMBEDDING_MODEL}...")
    embeddings: list[np.ndarray] = []
    t0 = time.time()
    for i, item in enumerate(all_chunks):
        if (i + 1) % 10 == 0 or (i + 1) == len(all_chunks):
            logger.info(f"  {i + 1}/{len(all_chunks)} embedded...")
        embeddings.append(get_embedding(item["text"]))
        time.sleep(0.05)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # 4. Upsert into Qdrant
    logger.info(f"\nUpserting into Qdrant collection '{QDRANT_SHUNT_COLLECTION}'...")
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_SHUNT_COLLECTION in existing:
        client.delete_collection(QDRANT_SHUNT_COLLECTION)
        logger.info(f"  Dropped existing collection '{QDRANT_SHUNT_COLLECTION}'")

    client.create_collection(
        collection_name=QDRANT_SHUNT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
    logger.info(f"  Created collection (dim={EMBEDDING_DIMENSION}, Cosine)")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": all_chunks[i]["text"],
                "source": all_chunks[i]["source"],
                "chunk_index": all_chunks[i]["chunk_index"],
            },
        )
        for i in range(len(all_chunks))
    ]

    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=QDRANT_SHUNT_COLLECTION, points=batch)
        logger.info(f"  Upserted {min(start + batch_size, len(points))}/{len(points)} points")

    info = client.get_collection(QDRANT_SHUNT_COLLECTION)
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Collection : {QDRANT_SHUNT_COLLECTION}")
    logger.info(f"  Points     : {info.points_count}")
    logger.info(f"  Sources    : {[s[0] for s in sources]}")
    logger.info(f"  Storage    : {QDRANT_PATH}")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
