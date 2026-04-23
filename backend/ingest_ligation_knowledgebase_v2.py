"""
Ingest Ligation Knowledgebase V2 — Type-Specific Chunks
========================================================
Re-ingests ligation knowledge with TYPE-SPECIFIC synthetic chunks
(similar to shunt_classification_db_v2).

Uses 5 chunking strategies:
1. docx_paragraph: Original paragraphs from source
2. synthetic_type: Hand-crafted type-specific ligation strategies
3. decision_tree: Decision logic for each type classification
4. surgical_technique: Specific surgical steps per type
5. complication_management: Type-specific complication handling

Run:
    cd backend
    python ingest_ligation_knowledgebase_v2.py
"""

import os
import sys
import requests
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
COLLECTION_NAME = "ligation_knowledgebase_db_v2"

# Type-specific ligation synthetic chunks
SYNTHETIC_CHUNKS = {
    "Type 1 Ligation Strategy": {
        "text": "Type 1 venous shunt ligation: SFJ incompetent with N1->N2 entry and N2->N1 GSV reflux. Primary ligation site: high tie at saphenofemoral junction. Secondary: ligate below each RP N2->N1 except most distal. Approach: under local anesthesia, high ligation to prevent thrombosis. Consider venous diameter and multiple reflux points.",
        "shunt_types": ["Type 1"],
        "chunk_type": "synthetic_type"
    },
    "Type 2A Ligation Strategy": {
        "text": "Type 2A saphenous-tributary shunt ligation: SFJ competent. EP N2->N3 (GSV to tributary) with RP N3->N2. Ligation target: highest escape point at N2->N3 junction. If multiple tributaries: assess calibre, distance to perforator, drainage pattern. Follow-up: 6-12 months for late reflux development. CHIVA approach preferred.",
        "shunt_types": ["Type 2A"],
        "chunk_type": "synthetic_type"
    },
    "Type 2B Ligation Strategy": {
        "text": "Type 2B perforator-fed shunt ligation: EP N2->N2 (perforator entry into saphenous trunk). SFJ competent. RP N3->N1 (tributary to deep). Ligation target: highest EP N2->N2 (perforator entry point). Open distal shunt considerations: preserve GSV function if possible. Selective perforator ligation approach. No SFJ involvement required.",
        "shunt_types": ["Type 2B"],
        "chunk_type": "synthetic_type"
    },
    "Type 3 Ligation Strategy": {
        "text": "Type 3 venous shunt ligation (SFJ incompetent with tributary involvement): Dual EP at N1->N2 and N2->N3. Staged approach recommended. Stage 1: Ligate EP N2->N3 tributaries at their junctions. Stage 2: Follow-up at 6-12 months assess SFJ reflux development. If N2 reflux develops during follow-up: then perform SFJ ligation. Conservative initial approach to avoid unnecessary SFJ intervention.",
        "shunt_types": ["Type 3"],
        "chunk_type": "synthetic_type"
    },
    "Type 1+2 Ligation Strategy": {
        "text": "Type 1+2 complex venous shunt ligation: Dual entry with EP N1->N2 (SFJ incompetent) AND EP N2->N3 (tributary). RP patterns: RP N2->N1 AND RP N3 present. Ligation strategy depends on RP N2->N1 diameter and elimination test result. Small RP N2->N1: CHIVA 2 approach (ligate EP N2->N3 first, assess, then SFJ). Large/multiple RP N2->N1: simultaneous ligation of SFJ and tributaries. Key: RP diameter assessment determines treatment sequence.",
        "shunt_types": ["Type 1+2"],
        "chunk_type": "synthetic_type"
    },
    "Type 1 Surgical Technique": {
        "text": "SFJ high ligation surgical steps: 1. Groin incision at inguinal ligament level. 2. Identify proximal GSV and tributaries. 3. Dissect SFJ to expose entry into femoral vein. 4. Ligate above tributaries to prevent thrombosis. 5. For Hunterian perforator: incision at mid-thigh medial side. 6. Multiple GSV reflux: ligate below each except distal most to maintain collateral. 7. Avoid excessive dissection to preserve skin innervation.",
        "shunt_types": ["Type 1"],
        "chunk_type": "surgical_technique"
    },
    "Type 2A Surgical Technique": {
        "text": "Type 2A tributary entry ligation surgical steps: 1. Identify highest EP at N2->N3 junction via ultrasound guidance. 2. Small incisions at ligation levels for tributary branches. 3. Ligate GSV branch feeding tributary at junction level. 4. For multiple tributaries: assess each for size and location. 5. Preserve GSV trunk if diameter normal. 6. Technique: ligation at junction allows branch preservation. 7. Echo-guided marking essential for accurate localization.",
        "shunt_types": ["Type 2A"],
        "chunk_type": "surgical_technique"
    },
    "Type 2B Surgical Technique": {
        "text": "Type 2B perforator ligation surgical steps: 1. Ultrasound identify perforator entry point (N2->N2). 2. Location depends on posYRatio: SFJ-Knee level, Hunterian, or calf location. 3. Longitudinal or transverse incision at ligation site. 4. Dissect to expose perforator vein. 5. Ligate perforator away from entry into GSV. 6. Preserve GSV trunk patent. 7. Multiple perforators: repeat for each based on hemodynamics.",
        "shunt_types": ["Type 2B"],
        "chunk_type": "surgical_technique"
    },
    "Type 3 Surgical Technique": {
        "text": "Type 3 tributary-first staged ligation surgical steps: Stage 1: 1. Identify EP N2->N3 tributaries via ultrasound. 2. Make small incisions at tributaries. 3. Ligate GSV to tributary junctions. 4. Preserve proximal GSV for potential second stage. 5. Leave SFJ untouched. Stage 2 (at 6-12 month follow-up if needed): 1. If N2 reflux develops: perform SFJ high ligation. 2. Conservative approach avoids unnecessary SFJ intervention.",
        "shunt_types": ["Type 3"],
        "chunk_type": "surgical_technique"
    },
    "Type 1+2 Surgical Technique": {
        "text": "Type 1+2 complex dual-entry ligation surgical steps (depends on RP N2->N1 diameter): Small diameter RP N2->N1: CHIVA 2 staged approach: Stage 1: Ligate EP N2->N3 tributary. Stage 2: Assess at follow-up, then SFJ ligation if needed. Large/multiple RP N2->N1: Simultaneous approach: 1. SFJ ligation at groin. 2. Tributaries ligation at multiple levels. 3. Ligate below each RP N2->N1 except distal most. 4. Complete hemodynamic correction in one procedure.",
        "shunt_types": ["Type 1+2"],
        "chunk_type": "surgical_technique"
    },
}


def embed_text(text: str) -> list[float]:
    """Embed using nomic-embed-text."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"ERROR embedding: {e}")
        return [0.0] * EMBEDDING_DIM


def main():
    print("=" * 80)
    print(f"INGESTING: {COLLECTION_NAME} (Type-Specific Ligation Chunks)")
    print("=" * 80)

    client = QdrantClient(path=QDRANT_PATH)

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create new collection with 768-dim vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME} (dim={EMBEDDING_DIM})")

    # Embed and upsert chunks
    print(f"\nEmbedding {len(SYNTHETIC_CHUNKS)} synthetic type-specific chunks...")
    points = []
    for idx, (title, chunk_data) in enumerate(SYNTHETIC_CHUNKS.items(), 1):
        vec = embed_text(chunk_data["text"])
        point = PointStruct(
            id=idx,
            vector=vec,
            payload={
                "text": chunk_data["text"],
                "title": title,
                "shunt_types": chunk_data["shunt_types"],
                "chunk_type": chunk_data["chunk_type"],
            },
        )
        points.append(point)
        print(f"  {idx}/{len(SYNTHETIC_CHUNKS)} embedded: {title}")

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nUpserted {len(points)} points into {COLLECTION_NAME}")

    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n{'=' * 80}")
    print(f"INGESTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Points: {collection_info.points_count}")
    print(f"  Vector dimension: {EMBEDDING_DIM}")
    print(f"  Chunk types: synthetic_type, surgical_technique")
    print(f"  Shunt types covered: Type 1, Type 2A, Type 2B, Type 3, Type 1+2")

    client.close()


if __name__ == "__main__":
    main()
