"""
Ligation Chunk Differentiation Test
====================================
Tests whether type-specific ligation queries retrieve DIFFERENT chunks
for each shunt type. Shows retrieved chunk IDs to verify differentiation.

Run:
    cd backend
    python test_ligation_chunk_differentiation.py
"""

import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
QDRANT_COLLECTION = "ligation_knowledgebase_db_v2"  # NEW v2 with type-specific chunks
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Matches v2 indexing
EMBEDDING_DIM = 768

# Highly discriminative, type-specific ligation queries
QUERIES = {
    "Type 1 (SFJ)": "SFJ incompetent with circular reflux N1->N2->N1. High ligation tie at saphenofemoral junction. Multiple GSV reflux points management strategy.",
    "Type 2A (Tributary)": "Tributary entry from GSV trunk N2->N3 without SFJ involvement. Ligate highest EP at tributary junction. Branching anatomy considerations.",
    "Type 2B (Perforator)": "Perforator-fed shunt via N2->N2 entry into saphenous trunk. Open distal shunt with tributary reflux N3->N1. Selective perforator ligation.",
    "Type 3 (Staged)": "SFJ incompetent with dual entries: EP N1->N2 and EP N2->N3. Staged approach: tributary ligation first, then follow-up for SFJ. Six to twelve month reassessment.",
    "Type 1+2 (Complex)": "Complex dual entry shunt with SFJ incompetence and tributary involvement. RP N2->N1 diameter determines strategy. CHIVA 2 staged vs simultaneous ligation.",
}


def embed(text: str) -> list[float]:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"ERROR: {e}")
        return [0.0] * EMBEDDING_DIM


def retrieve_chunks(client: QdrantClient, query: str, k: int = 5) -> list[tuple]:
    """Returns (chunk_id, score, text_preview)"""
    try:
        vec = embed(query)
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=k,
            with_payload=True,
        )
        return [(h.id, h.score, h.payload.get("text", "")[:100]) for h in hits.points]
    except Exception as e:
        print(f"ERROR: {e}")
        raise


def main():
    print("=" * 90)
    print("LIGATION CHUNK DIFFERENTIATION TEST")
    print("=" * 90)
    print("\nTesting if TYPE-SPECIFIC queries retrieve DIFFERENT chunks for each type\n")

    client = QdrantClient(path=QDRANT_PATH)

    all_retrieved = {}
    all_ids = {}

    for type_name, query in QUERIES.items():
        print(f"\n{type_name}")
        print("-" * 90)
        print(f"Query: {query[:80]}...")
        print()

        chunks = retrieve_chunks(client, query, k=5)
        all_retrieved[type_name] = chunks
        chunk_ids = [ch[0] for ch in chunks]
        all_ids[type_name] = set(chunk_ids)

        for i, (chunk_id, score, text) in enumerate(chunks, 1):
            print(f"  {i}. ID={chunk_id:3d} (score: {score:.4f}) | {text}...")

    client.close()

    # Analysis
    print(f"\n{'=' * 90}")
    print("DIFFERENTIATION ANALYSIS")
    print(f"{'=' * 90}\n")

    # Show chunk ID sets
    print("Retrieved Chunk IDs per Type:")
    for type_name, ids in all_ids.items():
        print(f"  {type_name:25s}: {sorted(ids)}")

    print()

    # Calculate overlap
    all_chunk_ids = set()
    for ids in all_ids.values():
        all_chunk_ids.update(ids)

    print(f"Total unique chunks retrieved: {len(all_chunk_ids)}")

    # Find which chunks appear in multiple types
    overlapping = {}
    for chunk_id in all_chunk_ids:
        types = [t for t, ids in all_ids.items() if chunk_id in ids]
        if len(types) > 1:
            overlapping[chunk_id] = types

    if overlapping:
        print(f"\nChunks appearing in MULTIPLE types (BAD):")
        for chunk_id, types in sorted(overlapping.items()):
            print(f"  ID={chunk_id}: appears in {types}")
    else:
        print(f"\n✓ ALL chunks are TYPE-SPECIFIC (perfect differentiation!)")

    # Percentage
    unique_count = sum(1 for types_list in overlapping.values() if len(types_list) == 1)
    total_chunks_across_all = sum(len(ids) for ids in all_ids.values())
    overlap_ratio = 1.0 - (len(overlapping) / total_chunks_across_all) if total_chunks_across_all > 0 else 1.0

    print(f"\nDifferentiation Quality: {overlap_ratio*100:.1f}%")
    print(f"  (Percentage of chunks that are NOT shared across types)")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
