"""
Comprehensive Ligation Testing Framework
==========================================
Unified script combining:
- K-divergence analysis (finding optimal k for each shunt type)
- Chunk retrieval logging (detailed chunk IDs, scores, text)
- LLM ligation output logging (sites, reasoning, confidence)
- Quality metrics (comparing against baseline CHIVA plans)

Outputs single Word document with all analysis.

Run:
    cd backend
    python test_ligation_comprehensive.py
"""

import os
import sys
import json
import requests
import glob
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from groq import Groq as GroqClient
from config import GROQ_API_KEY, GROQ_MODEL

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor
except ImportError:
    print("ERROR: python-docx not installed")
    sys.exit(1)

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
QDRANT_COLLECTION = "ligation_knowledgebase_db_v2"  # Type-specific chunks
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Matches v2 indexing
EMBEDDING_DIM = 768

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "json samples"))

# Baseline CHIVA ligation plans for quality comparison
BASELINE_PLANS = {
    "Type 1": {
        "primary_site": "SFJ",
        "ligation_targets": ["saphenofemoral junction", "SFJ", "high tie"],
        "approach": "high ligation at SFJ",
        "key_points": ["N1->N2 entry", "N2->N1 reflux", "circular flow"]
    },
    "Type 2A": {
        "primary_site": "N2->N3",
        "ligation_targets": ["tributary junction", "N2->N3", "EP N2->N3"],
        "approach": "ligate highest EP at tributary junction",
        "key_points": ["SFJ competent", "N2->N3 entry", "N3 reflux"]
    },
    "Type 2B": {
        "primary_site": "Perforator",
        "ligation_targets": ["perforator", "N2->N2", "EP N2->N2"],
        "approach": "selective perforator ligation",
        "key_points": ["SFJ competent", "N2->N2 entry", "open distal shunt"]
    },
    "Type 3": {
        "primary_site": "Tributary (staged)",
        "ligation_targets": ["tributary", "EP N2->N3", "staged"],
        "approach": "staged approach: ligate tributaries first, then SFJ if needed",
        "key_points": ["dual EP", "SFJ + tributary", "staged", "follow-up"]
    },
    "Type 1+2": {
        "primary_site": "SFJ + Tributary",
        "ligation_targets": ["SFJ", "tributary", "N2->N3", "RP N2->N1"],
        "approach": "depends on RP diameter: CHIVA 2 or simultaneous",
        "key_points": ["dual entry", "RP diameter", "CHIVA 2", "complex"]
    }
}

# Semantic query pairs for K-divergence testing
QUERY_PAIRS = {
    "Type 1": {
        "A": "SFJ incompetent with circular reflux N1->N2->N1. High ligation tie at saphenofemoral junction. Multiple GSV reflux points management.",
        "B": "Type 1 shunt: SFJ entry (N1->N2) with saphenous reflux back (N2->N1). Where to ligate for circular flow?"
    },
    "Type 2A": {
        "A": "Tributary entry from GSV trunk N2->N3 without SFJ involvement. Ligate highest EP at tributary junction. Branching anatomy.",
        "B": "Type 2A: SFJ competent, tributary entry (N2->N3) with reflux (N3->N2). Primary ligation site?"
    },
    "Type 2B": {
        "A": "Perforator-fed shunt via N2->N2 entry into saphenous trunk. Open distal shunt with tributary reflux N3->N1. Selective perforator ligation.",
        "B": "Type 2B: Perforator entry (N2->N2), SFJ intact, with distal reflux (N3->N1). Where is the entry point?"
    },
    "Type 3": {
        "A": "SFJ incompetent with dual entries: EP N1->N2 and EP N2->N3. Staged approach: tributary ligation first, then follow-up SFJ.",
        "B": "Type 3: Both SFJ and tributary entries (N1->N2 and N2->N3). What is the recommended treatment sequence?"
    },
    "Type 1+2": {
        "A": "Complex dual entry shunt with SFJ incompetence and tributary involvement. RP N2->N1 diameter determines strategy. CHIVA 2 vs simultaneous.",
        "B": "Type 1+2: SFJ + tributary entries (N1->N2, N2->N3) with RP at N2->N1. How does RP diameter affect ligation choice?"
    }
}


def embed(text: str) -> list:
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


def retrieve_chunks_at_k(client: QdrantClient, query: str, k_max: int = 10) -> dict:
    """
    Retrieve chunks at multiple k values (1 to k_max).
    Returns dict: {k: [(chunk_id, score, text), ...]}
    """
    vec = embed(query)
    results = {}
    try:
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=k_max,
            with_payload=True,
        )
        for k in range(1, k_max + 1):
            results[k] = [
                (h.id, h.score, h.payload.get("text", "")[:100])
                for h in hits.points[:k]
            ]
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    return results


def find_divergence_point(qa_results: dict, qb_results: dict) -> int:
    """Find k value where retrieved chunk IDs diverge between two queries."""
    for k in range(1, max(qa_results.keys()) + 1):
        qa_ids = {item[0] for item in qa_results[k]}
        qb_ids = {item[0] for item in qb_results[k]}
        if qa_ids != qb_ids:
            return k
    return 0


def calculate_stability(qa_results: dict, qb_results: dict, k: int) -> float:
    """Calculate stability at specific k: Jaccard similarity of chunk IDs."""
    qa_ids = {item[0] for item in qa_results[k]}
    qb_ids = {item[0] for item in qb_results[k]}
    if not qa_ids and not qb_ids:
        return 1.0
    intersection = len(qa_ids & qb_ids)
    union = len(qa_ids | qb_ids)
    return intersection / union if union > 0 else 0.0


def generate_ligation_plan(clips: list, shunt_type: str) -> dict:
    """Generate ligation plan using LLM with RAG context."""
    clips_str = "\n".join([
        f"  Clip {i+1}: {c.get('flow', '?')} {c.get('fromType', '?')}->"
        f"{c.get('toType', '?')} (y={c.get('posYRatio', 0.0):.3f})"
        for i, c in enumerate(clips)
    ])

    prompt = f"""=== LIGATION PLANNING TASK ===

Shunt Type: {shunt_type}

Clips Analysis:
{clips_str}

Based on CHIVA ligation principles, generate a specific ligation plan.
Output ONLY valid JSON — no markdown, no explanation.

{{
    "ligation_sites": ["<site 1>", "<site 2>"],
    "primary_approach": "<main strategy>",
    "reasoning": ["<reason 1>", "<reason 2>", "<reason 3>"],
    "confidence": 0.85,
    "chiva_alignment": "high"
}}"""

    try:
        client = GroqClient(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ligation_sites": [],
            "primary_approach": "PARSE_ERROR",
            "reasoning": ["JSON parsing failed"],
            "confidence": 0.0,
            "chiva_alignment": "unknown"
        }
    except Exception as e:
        return {
            "ligation_sites": [],
            "error": str(e),
            "confidence": 0.0,
            "chiva_alignment": "unknown"
        }


def score_ligation_quality(llm_plan: dict, baseline: dict, shunt_type: str) -> dict:
    """
    Compare LLM-generated ligation against baseline CHIVA plan.
    Metrics:
    - site_match: Does LLM recommend correct primary ligation site?
    - approach_alignment: Does approach match CHIVA guideline?
    - confidence_calibration: Is stated confidence appropriate?
    """
    scores = {
        "site_match": 0.0,
        "approach_alignment": 0.0,
        "confidence_calibration": 0.0,
        "overall_quality": 0.0,
        "details": []
    }

    # 1. Site Match: Check if LLM sites overlap with baseline targets
    llm_sites = set(str(s).lower() for s in llm_plan.get("ligation_sites", []))
    baseline_targets = set(s.lower() for s in baseline["ligation_targets"])

    if llm_sites and baseline_targets:
        matches = len(llm_sites & baseline_targets)
        scores["site_match"] = matches / max(len(llm_sites), len(baseline_targets))
        scores["details"].append(
            f"Site match: {matches}/{max(len(llm_sites), len(baseline_targets))} "
            f"(LLM: {llm_sites}, Baseline: {baseline_targets})"
        )
    else:
        scores["details"].append("No ligation sites provided by LLM")

    # 2. Approach Alignment: Check if approach keywords match baseline
    llm_approach = str(llm_plan.get("primary_approach", "")).lower()
    baseline_approach = baseline["approach"].lower()

    baseline_keywords = set(baseline_approach.split())
    approach_matches = sum(1 for kw in baseline_keywords if kw in llm_approach)
    scores["approach_alignment"] = approach_matches / len(baseline_keywords) if baseline_keywords else 0.0
    scores["details"].append(
        f"Approach alignment: {approach_matches}/{len(baseline_keywords)} keywords matched"
    )

    # 3. Confidence Calibration: Is confidence appropriate for correctness?
    confidence = llm_plan.get("confidence", 0.0)
    if isinstance(confidence, (int, float)):
        site_match = scores["site_match"]
        approach_match = scores["approach_alignment"]
        actual_quality = (site_match + approach_match) / 2

        # Penalize if confidence doesn't correlate with actual quality
        confidence_error = abs(confidence - actual_quality)
        scores["confidence_calibration"] = max(0.0, 1.0 - confidence_error)
        scores["details"].append(
            f"Confidence calibration: stated={confidence:.2f}, actual={actual_quality:.2f}, "
            f"calibration_score={scores['confidence_calibration']:.2f}"
        )

    # 4. Overall quality score
    scores["overall_quality"] = (
        scores["site_match"] * 0.4 +
        scores["approach_alignment"] * 0.4 +
        scores["confidence_calibration"] * 0.2
    )

    return scores


def add_kdivergence_to_doc(doc: Document, shunt_type: str, qa_results: dict, qb_results: dict):
    """Add K-divergence analysis to document."""
    doc.add_heading(f"{shunt_type} — K-Divergence Analysis", level=2)

    divergence_k = find_divergence_point(qa_results, qb_results)
    doc.add_paragraph(f"Divergence Point: k={divergence_k}", style="List Bullet")

    doc.add_heading("Stability at Each K", level=3)
    for k in range(1, 8):
        stability = calculate_stability(qa_results, qb_results, k)
        qa_ids = [item[0] for item in qa_results[k]]
        qb_ids = [item[0] for item in qb_results[k]]

        status = "STABLE" if stability == 1.0 else "DIVERGING"
        doc.add_paragraph(
            f"k={k}: stability={stability:.2f} ({status}) | "
            f"QA_IDs: {qa_ids}, QB_IDs: {qb_ids}",
            style="List Bullet"
        )


def add_retrieval_to_doc(doc: Document, query_label: str, results: dict, k_focus: int = 3):
    """Add chunk retrieval details to document."""
    doc.add_heading(f"Retrieval: {query_label}", level=3)

    chunks = results.get(k_focus, [])
    for i, (chunk_id, score, text) in enumerate(chunks, 1):
        doc.add_paragraph(
            f"Chunk {i}: ID={chunk_id}, Score={score:.4f}",
            style="List Bullet"
        )
        doc.add_paragraph(f"{text}...", style="List Bullet 2")


def add_llm_output_to_doc(doc: Document, llm_plan: dict, quality_scores: dict):
    """Add LLM output and quality metrics to document."""
    doc.add_heading("LLM Ligation Plan", level=3)

    doc.add_paragraph(
        f"Primary Approach: {llm_plan.get('primary_approach', '?')}",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"Ligation Sites: {llm_plan.get('ligation_sites', [])}",
        style="List Bullet"
    )

    confidence = llm_plan.get("confidence", 0.0)
    if isinstance(confidence, (int, float)):
        doc.add_paragraph(f"Confidence: {confidence:.2f}", style="List Bullet")

    reasoning = llm_plan.get("reasoning", [])
    if reasoning:
        doc.add_paragraph("Reasoning:", style="List Bullet")
        for reason in reasoning:
            doc.add_paragraph(reason, style="List Bullet 2")

    # Quality metrics
    doc.add_heading("Quality Assessment", level=3)
    doc.add_paragraph(
        f"Overall Quality Score: {quality_scores['overall_quality']:.2f}/1.0",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"Site Match: {quality_scores['site_match']:.2f}",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"Approach Alignment: {quality_scores['approach_alignment']:.2f}",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"Confidence Calibration: {quality_scores['confidence_calibration']:.2f}",
        style="List Bullet"
    )

    for detail in quality_scores["details"]:
        doc.add_paragraph(detail, style="List Bullet 2")


def main():
    print("=" * 90)
    print("COMPREHENSIVE LIGATION TESTING FRAMEWORK")
    print("=" * 90)

    client = QdrantClient(path=QDRANT_PATH)
    doc = Document()

    # Title
    title = doc.add_heading("Comprehensive Ligation Analysis Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Executive Summary
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "This comprehensive test combines K-divergence analysis, chunk retrieval logging, "
        "LLM output logging, and quality metrics for ligation planning. "
        "Tests are run against type-specific knowledge base (ligation_knowledgebase_db_v2) "
        "using semantic query pairs and actual patient data."
    )

    # Section 1: K-Divergence Analysis
    print("\n" + "=" * 90)
    print("SECTION 1: K-DIVERGENCE ANALYSIS")
    print("=" * 90)

    doc.add_heading("Section 1: K-Divergence Analysis", level=1)
    doc.add_paragraph(
        "Finding optimal k for each shunt type by detecting where retrieved chunks "
        "diverge between semantically similar queries."
    )

    kdivergence_results = {}
    for shunt_type, queries in QUERY_PAIRS.items():
        print(f"\n{shunt_type}:")
        qa_results = retrieve_chunks_at_k(client, queries["A"], k_max=10)
        qb_results = retrieve_chunks_at_k(client, queries["B"], k_max=10)
        divergence_k = find_divergence_point(qa_results, qb_results)
        kdivergence_results[shunt_type] = divergence_k
        print(f"  Divergence at k={divergence_k}")

        doc.add_heading(shunt_type, level=2)
        doc.add_paragraph(f"Query A: {queries['A'][:80]}...", style="List Bullet")
        doc.add_paragraph(f"Query B: {queries['B'][:80]}...", style="List Bullet")
        add_kdivergence_to_doc(doc, shunt_type, qa_results, qb_results)

    # Section 2: Detailed Retrieval & LLM Output with Actual Data
    print("\n" + "=" * 90)
    print("SECTION 2: DETAILED RETRIEVAL & LLM OUTPUT ANALYSIS")
    print("=" * 90)

    doc.add_heading("Section 2: Detailed Retrieval & LLM Output Analysis", level=1)
    doc.add_paragraph(
        "Testing with actual patient data from JSON samples. For each sample, "
        "retrieval chunks and LLM-generated ligation plans are logged with quality assessment."
    )

    json_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.json")))[:5]
    quality_summary = defaultdict(list)

    for json_file in json_files:
        sample_name = os.path.basename(json_file).replace(".json", "")
        print(f"\n{sample_name}:")

        try:
            with open(json_file) as f:
                data = json.load(f)

            clips = data.get("clips", [])
            if not clips:
                print("  (No clips, skipping)")
                continue

            # Infer shunt type from clips
            flows = {(c.get('flow'), c.get('fromType'), c.get('toType')) for c in clips if c.get('flow')}
            has_ep_n1_n2 = any(f[0] == 'EP' and f[1] == 'N1' and f[2] == 'N2' for f in flows)
            has_ep_n2_n3 = any(f[0] == 'EP' and f[1] == 'N2' and f[2] == 'N3' for f in flows)
            has_ep_n2_n2 = any(f[0] == 'EP' and f[1] == 'N2' and f[2] == 'N2' for f in flows)
            has_rp_n2_n1 = any(f[0] == 'RP' and f[1] == 'N2' and f[2] == 'N1' for f in flows)
            has_rp_n3 = any(f[0] == 'RP' and f[1] == 'N3' for f in flows)

            if has_ep_n1_n2 and not has_ep_n2_n3 and has_rp_n2_n1 and not has_rp_n3:
                inferred_type = "Type 1"
            elif not has_ep_n1_n2 and has_ep_n2_n3 and not has_ep_n2_n2:
                inferred_type = "Type 2A"
            elif has_ep_n2_n2 and not has_ep_n1_n2 and has_rp_n3 and not has_rp_n2_n1:
                inferred_type = "Type 2B"
            elif has_ep_n1_n2 and has_ep_n2_n3 and has_rp_n3 and not has_rp_n2_n1:
                inferred_type = "Type 3"
            elif has_ep_n1_n2 and has_ep_n2_n3 and has_rp_n2_n1 and has_rp_n3:
                inferred_type = "Type 1+2"
            else:
                inferred_type = "Unknown"

            print(f"  Inferred Type: {inferred_type}")

            # Retrieve chunks
            query = QUERY_PAIRS.get(inferred_type, {}).get("A", f"Ligation planning for {inferred_type}")
            qa_results = retrieve_chunks_at_k(client, query, k_max=3)
            chunks = qa_results[3]

            # Generate LLM plan
            llm_plan = generate_ligation_plan(clips, inferred_type)
            print(f"  LLM sites: {llm_plan.get('ligation_sites', [])}")

            # Score quality
            baseline = BASELINE_PLANS.get(inferred_type, {})
            quality_scores = score_ligation_quality(llm_plan, baseline, inferred_type)
            quality_summary[inferred_type].append(quality_scores["overall_quality"])

            print(f"  Quality score: {quality_scores['overall_quality']:.2f}")

            # Add to document
            doc.add_heading(sample_name, level=2)
            doc.add_paragraph(f"Inferred Type: {inferred_type}", style="List Bullet")
            doc.add_paragraph(f"Clips count: {len(clips)}", style="List Bullet")

            doc.add_heading("Retrieved Chunks (k=3)", level=3)
            for i, (chunk_id, score, text) in enumerate(chunks, 1):
                doc.add_paragraph(
                    f"Chunk {i}: ID={chunk_id}, Score={score:.4f}",
                    style="List Bullet"
                )
                doc.add_paragraph(f"{text}...", style="List Bullet 2")

            add_llm_output_to_doc(doc, llm_plan, quality_scores)
            doc.add_paragraph("")  # Spacing

        except Exception as e:
            print(f"  ERROR: {e}")

    client.close()

    # Section 3: Summary & Metrics
    print("\n" + "=" * 90)
    print("SECTION 3: SUMMARY & QUALITY METRICS")
    print("=" * 90)

    doc.add_heading("Section 3: Summary & Quality Metrics", level=1)

    # K-divergence summary
    doc.add_heading("K-Divergence Summary", level=2)
    for shunt_type, divergence_k in kdivergence_results.items():
        if divergence_k == 0:
            msg = "No divergence (highly stable retrieval)"
        else:
            msg = f"Diverges at k={divergence_k} (stable up to k={divergence_k-1})"
        doc.add_paragraph(f"{shunt_type}: {msg}", style="List Bullet")

    # Quality summary
    doc.add_heading("LLM Output Quality Summary", level=2)
    if quality_summary:
        for shunt_type in sorted(quality_summary.keys()):
            scores = quality_summary[shunt_type]
            if scores:
                avg_score = sum(scores) / len(scores)
                doc.add_paragraph(
                    f"{shunt_type}: avg quality={avg_score:.2f} "
                    f"(n={len(scores)} samples)",
                    style="List Bullet"
                )
    else:
        doc.add_paragraph("No quality samples evaluated", style="List Bullet")

    doc.add_heading("Interpretation", level=2)
    doc.add_paragraph(
        "K-Divergence: Lower values indicate less stable retrieval under query variations. "
        "Divergence at k=1 suggests very low stability; k>=4 is more stable.",
        style="List Bullet"
    )
    doc.add_paragraph(
        "Quality Score: Composite of site match (40%), approach alignment (40%), "
        "and confidence calibration (20%). Scores >= 0.7 indicate strong alignment with CHIVA guidelines.",
        style="List Bullet"
    )

    # Save report
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Ligation_Comprehensive_{timestamp_str}.docx"
    doc.save(filename)
    print(f"\n{'=' * 90}")
    print(f"Report saved to: {filename}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
