"""
CHIVA Knowledge Base Ingestion
--------------------------------
Reads PDFs and DOCX files from the parent project's Knowledgebases folder,
chunks them, embeds via Ollama, and stores in a local Qdrant collection.

Also writes chunks_cache.json so the BM25 index can be rebuilt on subsequent
startups without re-querying Qdrant.

Usage:
    python ingest.py           # Skip if collection already populated
    python ingest.py --force   # Force re-ingestion
"""

import json
import logging
import os
import sys
import time
import numpy as np
import requests
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    QDRANT_PATH, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    QDRANT_COLLECTION, EMBEDDING_DIMENSION,
    OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    KB_DIR, CHUNKS_CACHE_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    if QDRANT_HOST:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    os.makedirs(QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


def get_embedding(text: str) -> np.ndarray:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"][0], dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def split_into_chunks(text: str) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _read_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        logger.error(f"PDF read failed ({path}): {e}")
        return ""


def _read_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX read failed ({path}): {e}")
        return ""


def _read_txt(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"TXT read failed ({path}): {e}")
        return ""


def load_knowledge_base() -> str:
    kb = Path(KB_DIR)
    if not kb.exists():
        logger.warning(f"Knowledgebases directory not found: {kb}. Using built-in fallback.")
        return _builtin_kb()

    all_text = ""
    readers = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}

    for suffix, reader_fn in readers.items():
        for fpath in sorted(kb.glob(f"*{suffix}")):
            if fpath.name.startswith("~"):
                continue
            logger.info(f"  Loading: {fpath.name}")
            text = reader_fn(str(fpath))
            if text.strip():
                header = f"\n\n{'='*60}\nSOURCE: {fpath.name}\n{'='*60}\n\n"
                all_text += header + text

    if not all_text.strip():
        logger.warning("No files loaded from KB directory. Using built-in fallback.")
        return _builtin_kb()

    logger.info(f"Loaded {len(all_text):,} characters from {kb}")
    return all_text


def _builtin_kb() -> str:
    return """
CHIVA VENOUS SHUNT CLASSIFICATION AND LIGATION — KNOWLEDGE BASE

=== TYPE 1 SHUNT ===
Definition: SFJ or Hunterian perforator incompetent (EP N1->N2). Circular reflux
N1->N2->N1. No tributary escape (no EP N2->N3). RP N2->N1 is the hallmark.
Ligation: High ligation at SFJ (posYRatio <= 0.098) or Hunterian perforator
(0.098 < y <= 0.353). If multiple RP N2->N1 sites: ligate below each except the
most distal one. Outpatient surgery possible. Preserve GSV if calibre acceptable.

=== TYPE 2A SHUNT ===
Definition: No SFJ entry (EP N1->N2 absent). EP N2->N3 is the key: GSV feeds a
tributary without deep-system origin. SFJ competent. May have RP N3->N2 or N3->N1.
Ligation: Ligate the highest EP N2->N3 junction. If multiple branching tributaries
(ask_branching=true): choose based on calibre and drainage path.
Follow-up: Duplex at 6-12 months.

=== TYPE 2B SHUNT ===
Definition: Perforator entry EP N2->N2 (SFJ competent). RP at N3 (N3->N2 or N3->N1).
No RP N2->N1. Perforator is the culprit, NOT the SFJ.
Ligation: Ligate the highest EP N2->N2 perforator entry point.

=== TYPE 2C SHUNT ===
Definition: Perforator entry EP N2->N2 PLUS secondary GSV reflux RP N2->N1.
SFJ remains competent (EP N1->N2 absent). Both the perforator and GSV trunk involved.
Ligation: Ligate perforator entry (EP N2->N2) AND all RP N2->N1 sites along GSV.

=== TYPE 3 SHUNT ===
Definition: SFJ incompetent (EP N1->N2) AND tributary escape (EP N2->N3).
RP only at N3, no RP N2->N1. The tributary is the dominant reflux path.
Ligation (CHIVA 2 staged approach):
  Step 1: Ligate EP N2->N3 tributary junction(s) first.
  Step 2: Follow-up duplex at 6-12 months.
  Step 3: If GSV (N2) reflux then develops, ligate SFJ/Hunterian.
Rationale: Removing the escape path often normalises GSV without
needing SFJ ligation, sparing the saphenous for future bypass grafting.

=== TYPE 1+2 SHUNT ===
Definition: SFJ incompetent (EP N1->N2) + tributary escape (EP N2->N3) + RP N3
+ RP N2->N1. Confirmed by elimination test = "Reflux".
Ligation depends on RP N2->N1 calibre:
  Small RP N2->N1: CHIVA 2 staged: ligate tributaries first, then SFJ.
  Large/multiple RP N2->N1: Simultaneous ligation of SFJ + all refluxing tributaries.
    Ligate below each RP N2->N1 site except the most distal.

=== TYPE 4 SHUNT ===
Definition: EP N1->N3 (direct deep-to-tributary entry, bypassing GSV) + RP N2->N1.
Ligation: Target the EP N1->N3 entry point and the return path.

=== TYPE 5 SHUNT ===
Definition: EP N1->N3 with looping return via RP N3->N2 or RP N3->N1.
Ligation: Target EP N1->N3 entry and all refluxing N3 return segments.

=== CHIVA PRINCIPLES ===
- Preserve the saphenous vein where possible (future bypass grafting value)
- Address the primary hemodynamic entry point first
- Staged approach preferred over saphenectomy
- Outpatient local/tumescent anaesthesia for isolated ligation
- Post-operative compression 3-4 weeks
- Follow-up duplex at 6-12 months to assess residual shunt

=== LIGATION TECHNIQUE ===
- Flush ligation at junction, no GSV stump > 1 cm (prevents recurrence)
- Ligate and divide, not merely compress
- Protect lymphatics (especially at SFJ in groin)
- Mark ligation site with duplex pre-operatively in erect position

=== COMPLICATIONS AND CONTRAINDICATIONS ===
Complications: Lymphocele, wound infection, nerve injury (saphenous nerve adjacent
to GSV below knee), haematoma, DVT (rare < 1%), recurrence if entry point not fully ligated.
Contraindications: Active DVT in target vein, severe arterial insufficiency (ABI < 0.6),
active skin infection overlying site, pregnancy (defer elective surgery).

=== FOLLOW-UP SCHEDULE ===
Standard: Duplex ultrasound at 6-12 months post-ligation.
High-risk (Type 1+2, multiple sites): 3 months, 6 months, 12 months.
Criteria for re-intervention: residual RP N2->N1 > 1 second OR new EP.
"""


def ingest(force_reingest: bool = False) -> int:
    logger.info("=" * 60)
    logger.info("CMED DEMO — Knowledge Base Ingestion")
    logger.info("=" * 60)

    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION in existing and not force_reingest:
        count = client.get_collection(QDRANT_COLLECTION).points_count or 0
        if count > 0:
            logger.info(
                f"Collection '{QDRANT_COLLECTION}' already has {count} points. "
                "Skipping. (Use --force to re-ingest.)"
            )
            return count

    # Load
    logger.info("[1/4] Loading knowledge base...")
    text = load_knowledge_base()
    logger.info(f"      {len(text):,} characters loaded")

    # Chunk
    logger.info(f"[2/4] Chunking (~{CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap)...")
    chunks = split_into_chunks(text)
    logger.info(f"      {len(chunks)} chunks created")

    # Embed
    logger.info(f"[3/4] Embedding via Ollama ({OLLAMA_EMBEDDING_MODEL})...")
    embeddings: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk))
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            logger.info(f"      {i + 1}/{len(chunks)} embedded")
        time.sleep(0.05)

    # Upsert
    logger.info("[4/4] Storing in Qdrant...")
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    for start in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[start : start + 100])
    logger.info(f"      {len(points)} points stored")

    # Write BM25 chunk cache
    with open(CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info(f"      BM25 cache written → {CHUNKS_CACHE_PATH}")

    final_count = client.get_collection(QDRANT_COLLECTION).points_count or len(chunks)
    logger.info("=" * 60)
    logger.info(f"Done. {final_count} points in '{QDRANT_COLLECTION}'.")
    logger.info("=" * 60)
    return final_count


if __name__ == "__main__":
    force = "--force" in sys.argv
    try:
        ingest(force_reingest=force)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)
