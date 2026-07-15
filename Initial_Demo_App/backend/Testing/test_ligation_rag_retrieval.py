"""
Ligation RAG Retrieval Test
============================
Tests retrieval and LLM ligation planning for different CHIVA shunt types.
Logs retrieved chunks and generates ligation recommendations with reasoning.

Run:
    cd backend
    python test_ligation_rag_retrieval.py
"""

import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from groq import Groq as GroqClient
from config import GROQ_API_KEY, GROQ_MODEL

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
QDRANT_COLLECTION = "ligation_knowledgebase_db"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "llama3.2:1b"
EMBEDDING_DIM = 2048

# Optimized ligation queries — short, focused on ligation approach for each type
LIGATION_QUERIES = {
    "Type 1": (
        "Type 1 venous shunt: SFJ incompetent with N1->N2->N1 circular flow. "
        "Ligation strategy: Where to ligate at SFJ or Hunterian? Multiple RP N2->N1 handling?"
    ),
    "Type 2A": (
        "Type 2A venous shunt: SFJ competent, EP N2->N3 to tributary, RP N3->N2. "
        "Ligation strategy: Highest EP N2->N3 junction. Multiple tributary branches? Branching considerations?"
    ),
    "Type 2B": (
        "Type 2B venous shunt: Perforator entry EP N2->N2, RP N3->N1, no RP N2->N1. "
        "Ligation strategy: Highest EP N2->N2 (perforator entry point). Open distal shunt considerations?"
    ),
    "Type 3": (
        "Type 3 venous shunt: SFJ incompetent with EP N1->N2 AND EP N2->N3, RP only at N3. "
        "Ligation strategy: Ligate EP N2->N3 tributaries. Follow-up 6-12 months for SFJ reflux?"
    ),
}

# Test clip patterns for each type
TEST_CLIPS = {
    "Type 1": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.06},
        {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.70},
    ],
    "Type 2A": [
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.30},
        {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.65},
    ],
    "Type 2B": [
        {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.65},
        {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.82},
    ],
    "Type 3": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.06},
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.40},
        {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.50},
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
    """Retrieve top-k chunks from Qdrant ligation KB."""
    try:
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
                "text": h.payload.get("text", "")[:250],
            }
            for h in hits.points
        ]
    except ValueError as e:
        if "not found" in str(e):
            print(f"\nERROR: Collection '{QDRANT_COLLECTION}' not found at {QDRANT_PATH}")
            print(f"Available: {[c.name for c in client.get_collections().collections]}")
        raise


def summarise_clips(clips: list[dict]) -> str:
    """Summarize clips for LLM context."""
    lines = []
    for i, clip in enumerate(clips, 1):
        flow = clip.get("flow", "?")
        from_t = clip.get("fromType", "?")
        to_t = clip.get("toType", "?")
        y = clip.get("posYRatio", 0.0)
        lines.append(f"  Clip {i}: {flow} {from_t}->{to_t} (y={y:.2f})")
    return "\n".join(lines)


def generate_ligation_plan(clips: list[dict], query: str, shunt_type: str) -> dict:
    """Generate ligation plan using LLM with RAG context."""
    clips_str = summarise_clips(clips)

    json_schema = """{
    "shunt_type": "<Type 1 / Type 2A / Type 2B / Type 2C / Type 3 / Type 1+2>",
    "ligation_sites": ["<site 1>", "<site 2>"],
    "approach": "<primary ligation strategy>",
    "reasoning": ["<reason 1>", "<reason 2>"],
    "follow_up": "<post-ligation follow-up plan>",
    "confidence": 0.85
}"""

    prompt = f"""=== LIGATION TASK: {shunt_type} ===

Shunt Context:
{query}

Clips:
{clips_str}

=== TASK ===
Based on the shunt type and clips above, generate a ligation plan.
Specify: where to ligate, approach strategy, clinical reasoning, and follow-up.
Output ONLY the JSON below — no other text, no markdown.

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
        try:
            import json as json_lib
            result = json_lib.loads(raw)
            return result
        except json_lib.JSONDecodeError:
            return {"shunt_type": shunt_type, "ligation_sites": [], "approach": "PARSE_ERROR", "confidence": 0.0, "reasoning": []}
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return {"shunt_type": shunt_type, "ligation_sites": [], "error": str(e), "confidence": 0.0}


def main():
    print("=" * 80)
    print("LIGATION RAG RETRIEVAL TEST")
    print("=" * 80)

    client = QdrantClient(path=QDRANT_PATH)

    for shunt_type in ["Type 1", "Type 2A", "Type 2B", "Type 3"]:
        query = LIGATION_QUERIES[shunt_type]
        clips = TEST_CLIPS[shunt_type]

        print(f"\n{'=' * 80}")
        print(f"SHUNT TYPE: {shunt_type}")
        print(f"{'=' * 80}")
        print(f"\nQuery ({len(query)} chars):\n  {query}")

        # Retrieve chunks
        print(f"\nRetrieved chunks (top-3):")
        chunks = retrieve_chunks(client, query, k=3)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  Chunk {i} (score: {chunk['score']:.4f}):")
            print(f"    {chunk['text']}")

        # Generate ligation plan
        print(f"\nLLM Ligation Plan:")
        plan = generate_ligation_plan(clips, query, shunt_type)
        print(f"  Shunt Type: {plan.get('shunt_type', '?')}")
        print(f"  Ligation Sites: {plan.get('ligation_sites', [])}")
        print(f"  Approach: {plan.get('approach', '?')}")

        confidence = plan.get('confidence', 0.0)
        if isinstance(confidence, (int, float)):
            print(f"  Confidence: {confidence:.2f}")
        else:
            print(f"  Confidence: {confidence}")

        reasoning = plan.get('reasoning', [])
        if reasoning:
            print(f"  Reasoning (steps): {len(reasoning)}")

    client.close()

    print(f"\n{'=' * 80}")
    print("RETRIEVAL TEST COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
