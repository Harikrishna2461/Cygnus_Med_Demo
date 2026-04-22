"""
Quick test: short, focused queries vs long V4 queries
=======================================================
Tests whether short, type-specific queries retrieve different chunks
and whether the LLM can classify correctly without CHIVA rules.

Run:
    cd backend
    python test_short_queries.py
"""

import os
import sys
import requests
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from groq import Groq as GroqClient
from config import GROQ_API_KEY, GROQ_MODEL

QDRANT_PATH = "qdrant_storage"
QDRANT_COLLECTION = "shunt_classification_db_v2"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# ── Short, focused queries (no type enumeration, no boilerplate) ─────────────

SHORT_QUERIES = {
    "Type 1": (
        "CHIVA shunt. EP N1->N2 SFJ incompetent. RP N2->N1 saphenous trunk reflux. "
        "Flow N1->N2->N1. No EP N2->N3. No RP at N3."
    ),
    "Type 2A": (
        "CHIVA shunt. EP N2->N3 GSV to tributary. SFJ competent. RP N3->N2. "
        "Flow N2->N3->N2. No EP N1->N2."
    ),
    "Type 2B": (
        "CHIVA shunt. EP N2->N2 perforator entry. SFJ competent. RP N3->N1. "
        "Flow N2->N3->N1. No RP N2->N1. No EP N1->N2."
    ),
}

# Test clips for each case (minimal, just flow information)
TEST_CLIPS = {
    "Type 1": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posXRatio": 0.25, "posYRatio": 0.06},
        {"flow": "RP", "fromType": "N2", "toType": "N1", "posXRatio": 0.20, "posYRatio": 0.70},
    ],
    "Type 2A": [
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posXRatio": 0.70, "posYRatio": 0.30},
        {"flow": "RP", "fromType": "N3", "toType": "N2", "posXRatio": 0.72, "posYRatio": 0.65},
    ],
    "Type 2B": [
        {"flow": "EP", "fromType": "N2", "toType": "N2", "posXRatio": 0.20, "posYRatio": 0.65},
        {"flow": "RP", "fromType": "N3", "toType": "N1", "posXRatio": 0.35, "posYRatio": 0.82},
    ],
}


def embed(text: str) -> list[float]:
    """Embed text using nomic-embed-text."""
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


def retrieve_chunks(client: QdrantClient, query: str, k: int = 3) -> list[dict]:
    """Retrieve top-k chunks from Qdrant."""
    vec = embed(query)
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "score": h.score,
            "chunk_type": h.payload.get("chunk_type", "unknown"),
            "shunt_types": h.payload.get("shunt_types", []),
            "text": h.payload.get("text", "")[:200],
        }
        for h in hits.points
    ]


def classify_without_rules(clips: list[dict], short_query: str, leg: str = "Right") -> dict:
    """Classify using LLM WITHOUT CHIVA rules — only the short query context."""
    from shunt_llm_classifier import _summarise_clips, _repair_and_parse

    clips_str = _summarise_clips(clips)

    json_schema = """{
    "shunt_type": "<Type 1 / Type 2A / Type 2B / Type 2C / Type 3 / Type 1+2 / No shunt detected>",
    "confidence": <0.0-1.0>,
    "reasoning": ["<step 1>", "<step 2>"],
    "summary": "<1 sentence>"
}"""

    prompt = f"""=== QUERY CONTEXT (NO CHIVA RULES) ===
{short_query}

=== ASSESSMENT: {leg} leg ===
{clips_str}

=== TASK ===
Based ONLY on the query context above (no embedded rules, no knowledge base),
classify the venous shunt type for the {leg} leg.
Output ONLY the JSON below — no other text.

{json_schema}"""

    try:
        client = GroqClient(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        result = _repair_and_parse(raw) or {"shunt_type": "PARSE_ERROR", "confidence": 0.0}
        return result
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return {"shunt_type": "LLM_ERROR", "confidence": 0.0, "error": str(e)}


def main():
    print("=" * 80)
    print("SHORT QUERY RETRIEVAL TEST")
    print("=" * 80)

    client = QdrantClient(path=QDRANT_PATH)

    retrieved_chunk_ids = {}  # Track which chunks were retrieved for each case

    for case_name, query in SHORT_QUERIES.items():
        print(f"\n{'=' * 80}")
        print(f"CASE: {case_name}")
        print(f"{'=' * 80}")
        print(f"\nQuery ({len(query)} chars):\n  {query}")

        # Retrieve chunks
        chunks = retrieve_chunks(client, query, k=3)
        print(f"\nRetrieved chunks (top-3):")
        chunk_ids = []
        for i, ch in enumerate(chunks, 1):
            print(f"\n  Chunk {i}:")
            print(f"    Score: {ch['score']:.4f}")
            print(f"    Type: {ch['chunk_type']}")
            print(f"    Shunt types: {ch['shunt_types']}")
            print(f"    Text: {ch['text']}")
            chunk_ids.append((ch['chunk_type'], tuple(ch['shunt_types'])))

        retrieved_chunk_ids[case_name] = chunk_ids

        # Classify without CHIVA rules
        clips = TEST_CLIPS[case_name]
        result = classify_without_rules(clips, query)
        print(f"\nLLM classification (WITHOUT CHIVA RULES):")
        print(f"  Shunt type: {result.get('shunt_type', '?')}")
        print(f"  Confidence: {result.get('confidence', '?')}")
        print(f"  Reasoning: {result.get('reasoning', [])}")
        print(f"  Summary: {result.get('summary', '')}")

    # Comparison: are the retrieved chunks DIFFERENT for each case?
    print(f"\n{'=' * 80}")
    print("CHUNK DIFFERENTIATION ANALYSIS")
    print(f"{'=' * 80}")

    # Check if any chunks appear in multiple cases
    all_chunks = []
    for case_name, chunk_ids in retrieved_chunk_ids.items():
        for chunk_id in chunk_ids:
            all_chunks.append((case_name, chunk_id))

    chunk_counts = {}
    for case_name, chunk_id in all_chunks:
        if chunk_id not in chunk_counts:
            chunk_counts[chunk_id] = []
        chunk_counts[chunk_id].append(case_name)

    print("\nChunk appearances across cases:")
    duplicates = 0
    for chunk_id, cases in sorted(chunk_counts.items()):
        if len(cases) > 1:
            duplicates += 1
            print(f"  [DUP] {chunk_id} appears in: {cases}")
        else:
            print(f"  [OK] {chunk_id} unique to: {cases[0]}")

    total_chunks = len(set(chunk_id for _, chunk_id in all_chunks))
    unique_chunks = sum(1 for cases in chunk_counts.values() if len(cases) == 1)

    print(f"\nSummary:")
    print(f"  Total unique chunks retrieved: {total_chunks}")
    print(f"  Chunks unique to one case: {unique_chunks}")
    print(f"  Chunks appearing in multiple cases: {duplicates}")
    print(f"  Differentiation ratio: {unique_chunks}/{total_chunks} ({100*unique_chunks//total_chunks}%)")

    if duplicates == 0:
        print("\n[SUCCESS] All cases retrieved completely different chunks!")
    else:
        print(f"\n[PROBLEM] {duplicates} chunks appeared in multiple cases (not fully differentiated)")

    client.close()


if __name__ == "__main__":
    main()
