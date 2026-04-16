"""
Addendum: Add CHIVA Type 2A / 2B / 2C knowledge to existing Qdrant collection.
DOES NOT delete or recreate the collection — only upserts new points.
Safe to run against a live DB.
"""

import os
import time
import logging
import requests
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import (
    QDRANT_PATH, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    QDRANT_COLLECTION, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIMENSION,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── New knowledge chunks to add ───────────────────────────────────────────────

NEW_CHUNKS = [
    """CHIVA TYPE 2A SHUNT — Definition and Clip Pattern
Type 2A occurs when the SFJ (saphenofemoral junction) is COMPETENT — no EP N1→N2 at groin.
Blood enters the superficial system via the GSV (EP N2→N3: GSV feeds a tributary).
The tributary then refluxes back: RP N3→N2 or RP N3→N1.
Crucially, there is NO RP N2→N1 — the GSV trunk itself does NOT reflux.
Key signals: EP N2→N3 present + RP at N3 only + NO EP N1→N2 at SFJ + NO RP N2→N1.
Location: Mid-thigh or knee level along the GSV medial course.
Treatment: Ligate the HIGHEST EP at the N2→N3 junction. Follow up 6-12 months for secondary N2 reflux.""",

    """CHIVA TYPE 2B SHUNT — Definition and Clip Pattern
Type 2B occurs when the SFJ is COMPETENT and entry is via a CALF PERFORATOR or SPJ (not SFJ).
The entry clip shows EP N2→N2 at the SPJ or calf level (step = SPJ, SPJ-Ankle, Knee-Ankle).
A tributary then refluxes: RP N3→N2 or RP N3→N1 at posterior or lateral calf.
NO EP N1→N2 at SFJ. NO RP N2→N1 along the GSV trunk.
Key signals: EP N2→N2 at SPJ/calf + RP N3 only + NO SFJ entry + NO N2→N1 reflux.
Location: Posterior calf, SPJ region, popliteal fossa. Small Saphenous Vein (SSV) often involved.
Treatment: Ligate the perforator entry at SPJ or highest calf perforator. No groin surgery needed.""",

    """CHIVA TYPE 2C SHUNT — Definition and Clip Pattern
Type 2C is a Type 2A or 2B pattern where SECONDARY GSV REFLUX has also developed.
Both RP N3 (tributary reflux) AND RP N2→N1 (GSV reflux) are present.
The SFJ is still COMPETENT — NO EP N1→N2 at the groin.
This distinguishes 2C from Type 1+2, where the SFJ IS incompetent (EP N1→N2 present).
Key signals: EP N2→N3 (or perforator entry) + RP N3 + RP N2→N1 + NO EP N1→N2 at SFJ.
Treatment: Ligate highest EP at N2→N3 AND all RP N2→N1 sites along the GSV.""",

    """CHIVA TYPE 2 SUBTYPES — Quick Distinction Table
All Type 2 subtypes share: SFJ COMPETENT (no EP N1→N2 at groin).

TYPE 2A:
  Entry: EP N2→N3 (GSV into tributary) at mid-thigh
  Reflux: RP N3 only (N3→N2 or N3→N1)
  GSV trunk (N2→N1) reflux: ABSENT
  Distinguishing: GSV-to-tributary entry; mid-thigh location

TYPE 2B:
  Entry: EP N2→N2 via SPJ or calf perforator (step = SPJ, SPJ-Ankle, Knee-Ankle)
  Reflux: RP N3 only (N3→N2 or N3→N1)
  GSV trunk (N2→N1) reflux: ABSENT
  Distinguishing: Perforator/SPJ entry at posterior calf; SSV often involved

TYPE 2C:
  Entry: EP N2→N3 or perforator (same as 2A or 2B)
  Reflux: BOTH RP N3 AND RP N2→N1 present
  GSV trunk (N2→N1) reflux: PRESENT (secondary)
  Distinguishing: N2→N1 reflux added on top of 2A/2B pattern; SFJ still competent

Difference from Type 1+2: In Type 2C the SFJ is competent; in Type 1+2 the SFJ is incompetent (EP N1→N2 present).""",

    """CHIVA TYPE 2 vs TYPE 1 and TYPE 1+2 — Critical Distinctions
TYPE 1: SFJ INCOMPETENT (EP N1→N2 present). RP N2→N1 along GSV. No N3 involvement.
TYPE 2A: SFJ competent. EP N2→N3. RP N3 only. No N2→N1. No SFJ entry.
TYPE 2B: SFJ competent. EP via SPJ/perforator (N2→N2). RP N3 only. No N2→N1.
TYPE 2C: SFJ competent. EP N2→N3 or perforator. RP N3 AND RP N2→N1. No SFJ entry.
TYPE 3:  SFJ INCOMPETENT (EP N1→N2 present). EP N2→N3 also present. RP N3. eliminationTest=No Reflux.
TYPE 1+2: SFJ INCOMPETENT. EP N1→N2 AND EP N2→N3. RP N3 AND RP N2→N1. eliminationTest=Reflux.

The single most important discriminator for Type 2 subtypes (2A, 2B, 2C):
  Is EP N1→N2 at SFJ present? YES → NOT Type 2 (go to Type 1, 3, or 1+2).
                              NO  → Type 2 subtype. Then check RP N2→N1 presence for 2C vs 2A/2B.""",
]


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


def main():
    # Connect to existing Qdrant
    if QDRANT_HOST:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(path=QDRANT_PATH)

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        logger.error(f"Collection '{QDRANT_COLLECTION}' not found. Run ingest.py first.")
        return

    info = client.get_collection(QDRANT_COLLECTION)
    existing_count = info.points_count
    logger.info(f"Existing collection '{QDRANT_COLLECTION}' has {existing_count} points.")

    # Embed new chunks
    logger.info(f"Embedding {len(NEW_CHUNKS)} new Type 2A/2B/2C chunks...")
    points = []
    for i, chunk in enumerate(NEW_CHUNKS):
        logger.info(f"  Embedding chunk {i+1}/{len(NEW_CHUNKS)}...")
        vec = get_embedding(chunk)
        points.append(PointStruct(
            id=existing_count + i,
            vector=vec.tolist(),
            payload={"text": chunk, "chunk_index": existing_count + i, "source": "chiva_type2_addendum"},
        ))
        time.sleep(0.1)

    # Upsert only — existing data untouched
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    new_count = client.get_collection(QDRANT_COLLECTION).points_count
    logger.info(f"Done. Collection now has {new_count} points (+{new_count - existing_count} added).")


if __name__ == "__main__":
    main()
