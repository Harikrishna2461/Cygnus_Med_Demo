"""
Shunt Classification Knowledge Base Ingestion — v2
====================================================
Source:  Shunt_Classification_Knowledgebase.docx  (primary, book excerpt pg 573-606)
         chiva_rules.txt                           (decision rules)

Advanced chunking strategy:
  1. Docx paragraph chunks      — each paragraph is a natural semantic unit
  2. Synthetic type chunks       — one per shunt type, hand-crafted from document reading,
                                   optimised for retrieval query patterns
  3. Flow-pattern lookup chunks  — minimal N→N notation + type label for pattern matching
  4. Context-window chunks       — 2-paragraph sliding windows for continuity
  5. Rules fine-grained chunks   — 80-word windows from chiva_rules.txt

Every chunk carries rich metadata:
  shunt_types    — list of shunt types covered (e.g. ["Type 1", "Type 2A"])
  flow_patterns  — N→N patterns present (e.g. ["N1->N2->N1"])
  chunk_type     — "docx_paragraph" | "synthetic_type" | "flow_pattern" |
                   "context_window" | "rules"
  source         — filename

Collection:  shunt_classification_db_v2

Run:
    cd backend
    python ingest_shunt_classification_v2.py
"""

import os
import re
import time
import shutil
import logging
import numpy as np
import requests
from pathlib import Path
from docx import Document as DocxDocument
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    OLLAMA_BASE_URL,
    QDRANT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    LOG_FILE,
    LOG_LEVEL,
)

QDRANT_SHUNT_V2_COLLECTION = "shunt_classification_db_v2"

# nomic-embed-text is a retrieval-specialized model — far better semantic
# discrimination than llama3.2:1b which collapses all CHIVA queries to the same region.
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION    = 768

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DOCX_PATH = BASE_DIR / "Shunt_Classification_Knowledgebase.docx"
RULES_PATH = BASE_DIR / "chiva_rules.txt"


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


# ── Docx extraction ───────────────────────────────────────────────────────────

def extract_docx_paragraphs(docx_path: Path) -> list[str]:
    """Return non-empty paragraph strings from the docx."""
    doc = DocxDocument(str(docx_path))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 20:   # skip trivial lines
            paragraphs.append(text)
    logger.info(f"  {docx_path.name}: {len(paragraphs)} paragraphs extracted")
    return paragraphs


# ── Rules loading ─────────────────────────────────────────────────────────────

def load_rules_text(rules_path: Path) -> str:
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


def split_rules_into_chunks(text: str, chunk_size: int = 80, overlap: int = 10) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ── Metadata helpers ──────────────────────────────────────────────────────────

_TYPE_PATTERNS = {
    "Type 1":   [r"\btype[\s\-]*1\b", r"\bshunt[\s\-]*1\b"],
    "Type 1+2": [r"\btype[\s\-]*1\+2\b", r"\bshunt[\s\-]*1\+2\b", r"\b1\+2\b"],
    "Type 2":   [r"\btype[\s\-]*2\b", r"\bshunt[\s\-]*2\b"],
    "Type 2A":  [r"\btype[\s\-]*2a\b", r"\bshunt[\s\-]*2a\b", r"\b2a\b"],
    "Type 2B":  [r"\btype[\s\-]*2b\b", r"\bshunt[\s\-]*2b\b", r"\b2b\b"],
    "Type 2C":  [r"\btype[\s\-]*2c\b", r"\bshunt[\s\-]*2c\b", r"\b2c\b"],
    "Type 3":   [r"\btype[\s\-]*3\b", r"\bshunt[\s\-]*3\b"],
    "Type 4":   [r"\btype[\s\-]*4\b", r"\bshunt[\s\-]*4\b"],
    "Type 5":   [r"\btype[\s\-]*5\b", r"\bshunt[\s\-]*5\b"],
    "Type 6":   [r"\btype[\s\-]*6\b", r"\bshunt[\s\-]*6\b"],
}

_FLOW_RE = re.compile(r"N[123456](?:->(?:N[123456]|B|P))+")


def detect_shunt_types(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for stype, patterns in _TYPE_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            found.append(stype)
    return found


def detect_flow_patterns(text: str) -> list[str]:
    return list(dict.fromkeys(_FLOW_RE.findall(text)))   # deduped, order-preserved


# ── Chunk builders ────────────────────────────────────────────────────────────

def build_docx_paragraph_chunks(paragraphs: list[str]) -> list[dict]:
    """Strategy 1: each paragraph = one chunk (natural semantic unit)."""
    chunks = []
    for idx, text in enumerate(paragraphs):
        chunks.append({
            "text": text,
            "source": DOCX_PATH.name,
            "chunk_type": "docx_paragraph",
            "shunt_types": detect_shunt_types(text),
            "flow_patterns": detect_flow_patterns(text),
            "chunk_index": idx,
        })
    return chunks


def build_context_window_chunks(paragraphs: list[str], window: int = 2) -> list[dict]:
    """Strategy 2: sliding 2-paragraph windows for cross-paragraph context."""
    chunks = []
    for i in range(len(paragraphs) - window + 1):
        text = " | ".join(paragraphs[i: i + window])
        chunks.append({
            "text": text,
            "source": DOCX_PATH.name,
            "chunk_type": "context_window",
            "shunt_types": detect_shunt_types(text),
            "flow_patterns": detect_flow_patterns(text),
            "chunk_index": i,
        })
    return chunks


def build_rules_chunks(rules_text: str) -> list[dict]:
    """Strategy 3: fine-grained chunks from chiva_rules.txt."""
    raw_chunks = split_rules_into_chunks(rules_text, chunk_size=80, overlap=10)
    chunks = []
    for idx, text in enumerate(raw_chunks):
        chunks.append({
            "text": text,
            "source": RULES_PATH.name,
            "chunk_type": "rules",
            "shunt_types": detect_shunt_types(text),
            "flow_patterns": detect_flow_patterns(text),
            "chunk_index": idx,
        })
    return chunks


# ── Synthetic type-specific chunks ────────────────────────────────────────────
# Each synthetic chunk is crafted to exactly mirror what a retrieval query looks
# like: EP/RP node pairs, flow notation, anatomical context, discriminators.
# Based on thorough reading of Shunt_Classification_Knowledgebase.docx (pg 573-606).

SYNTHETIC_TYPE_CHUNKS = [
    {
        "label": "Type 1 — comprehensive",
        "shunt_types": ["Type 1"],
        "flow_patterns": ["N1->N2->N1"],
        "text": (
            "CHIVA Shunt Type 1 — Closed Shunt (CS). "
            "Flow pattern: N1->N2->N1. "
            "Escape Point (EP): blood escapes from N1 (deep venous system) to N2 (saphenous trunk — GSV or SSV). "
            "EP is located at the saphenofemoral junction (SFJ), saphenopopliteal junction (SPJ), "
            "or N2 direct insufficient perforating vein. "
            "Re-entry Point (RP): focused on the N2 saphenous trunk itself (N2->N1), not on any tributary. "
            "SFJ is INCOMPETENT — clip has fromType=N1 AND toType=N2 (EP N1->N2). "
            "Key discriminator: RP is on N2 (retrograde GSV trunk reflux), no RP at N3 (tributary). "
            "Closed shunt: transmural pressure increases due to hydrostatic pressure. "
            "Blood refluxes from N1 to N2 and re-enters via N2->N1. "
            "Triggered by both Paranà/squeezing diastole and Valsalva systole. "
            "Classification signal: EP N1->N2 EXISTS, RP N2->N1 present, NO EP N2->N3, NO RP at N3. "
            "Anatomy: saphenofemoral junction incompetent, GSV reflux, saphenous trunk reflux. "
            "Ligation: ligate at SFJ (posYRatio ≤ 0.098) or Hunterian perforator (0.098 < posYRatio ≤ 0.353). "
        ),
    },
    {
        "label": "Type 1+2 — comprehensive",
        "shunt_types": ["Type 1+2"],
        "flow_patterns": ["N1->N2->N1", "N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 1+2 — Closed Shunt (CS). Combination of Shunt 1 and Shunt 2. "
            "Flow pattern: N1->N2->N1 AND N2->N3->N1 simultaneously. "
            "Two branches from N2: one goes directly N2->N1 (Shunt 1 re-entry), "
            "the other goes via N3 then N1 (Shunt 2 component). "
            "Escape Points: N1->N2 at SFJ or Hunterian (Shunt 1 EP) AND N2->N3 into tributary (Shunt 2 EP). "
            "Re-entry Points: RP N2->N1 along saphenous trunk AND RP N3->N1 from tributary. "
            "SFJ is INCOMPETENT (EP N1->N2 exists). "
            "N2->N3 reflux triggered by BOTH Paranà/squeezing diastole AND Valsalva systole "
            "(unlike Type 2 alone which is triggered only by diastole). "
            "eliminationTest clip with value 'Reflux' on RP N2->N1 confirms Type 1+2 "
            "(eliminationTest='No Reflux' would make it Type 3 instead). "
            "Classification signal: EP N1->N2 EXISTS + EP N2->N3 EXISTS + RP N3->N1 + RP N2->N1 "
            "+ eliminationTest='Reflux' on the RP N2->N1 clip. "
            "Ligation: depends on calibre of RP N2->N1 (ask_diameter=true). "
            "Small RP N2->N1: CHIVA 2 approach (ligate EP N2->N3 first, then SFJ). "
            "Large RP N2->N1: ligate SFJ/Hunterian + all tributaries simultaneously. "
        ),
    },
    {
        "label": "Type 2 general — N2 to N3 escape",
        "shunt_types": ["Type 2"],
        "flow_patterns": ["N2->N3->N2", "N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 2 — General. "
            "General flow: N2->N3->N2. "
            "Key characteristic: N2-N3 pathological compartment jump. "
            "Escape Point (EP): from N2 (saphenous trunk) to N3 (tributary/superficial branch). "
            "Re-entry perforator (RP) is focused on the same incompetent tributary, draining toward N1. "
            "CRITICAL: No reflux along the saphenous trunk itself (N2). No EP at N1->N2 (SFJ competent). "
            "SFJ is COMPETENT — no clip has fromType=N1 and toType=N2. "
            "Sub-types: 2A (ODS or CS), 2B (always ODS), 2C (always ODS). "
            "Triggered by Paranà/squeezing diastole but NOT by Valsalva systole. "
            "Classification signal: NO EP N1->N2 anywhere + EP N2->N3 or EP N2->N2 present. "
        ),
    },
    {
        "label": "Type 2A — GSV feeds tributary, SFJ competent",
        "shunt_types": ["Type 2A"],
        "flow_patterns": ["N2->N3->N2", "N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 2A — can be Open Deviated Shunt (ODS) or Closed Shunt (CS). "
            "The only exception among shunt type 2 sub-types that can be closed. "
            "Flow pattern: N2->N3->N2 AND N2->N3->N1 (two branches from the same N2 escape point). "
            "Branch 1: N2->N3->N2 (returns to saphenous trunk). "
            "Branch 2: N2->N3->N1 (drains to deep system). "
            "Escape Point (EP): N2->N3 (GSV or SSV trunk feeds a tributary, SFJ remains competent). "
            "Re-entry Points: RP N3->N2 back to saphenous AND RP N3->N1 to deep system. "
            "SFJ COMPETENT — no EP N1->N2 clip. "
            "In ODS variant: only hydrostatic pressure increases, no N2 segments with retrograde flow. "
            "Triggered by Paranà/squeezing diastole but NOT Valsalva systole. "
            "Classification signal: NO EP N1->N2 + EP N2->N3 + RP N3->N2 or RP N3->N1 + NO RP N2->N1. "
            "Ligation: ligate highest EP at N2->N3 junction. "
            "If multiple branching at N3: ask_branching=true. "
        ),
    },
    {
        "label": "Type 2B — perforator entry, no GSV reflux",
        "shunt_types": ["Type 2B"],
        "flow_patterns": ["N2->N3->N2", "N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 2B — always Open Deviated Shunt (ODS), never closed. "
            "Flow pattern: N2->N3->N2 AND N2->N3->N1 (same flow notation as 2A but always ODS). "
            "Escape Point: EP N2->N2 — perforator entry. fromType=N2, toType=N2. "
            "CRITICAL DISTINCTION: EP N2->N2 means blood circulates within the saphenous trunk via a perforator. "
            "SFJ REMAINS COMPETENT even if posYRatio is very low (e.g. 0.05). "
            "EP N2->N2 at ANY posYRatio = perforator entry, NOT SFJ entry. "
            "Re-entry at N3: RP N3->N2 or RP N3->N1. "
            "NO RP N2->N1 (no retrograde GSV trunk reflux — key distinction from Type 2C). "
            "Classification signal: EP N2->N2 EXISTS + RP N3 EXISTS + NO EP N1->N2 + NO RP N2->N1. "
            "Ligation: ligate the highest EP N2->N2 (perforator entry point). "
            "If multiple RP at N3: ask_branching=true. "
        ),
    },
    {
        "label": "Type 2C — perforator entry plus secondary GSV reflux",
        "shunt_types": ["Type 2C"],
        "flow_patterns": ["N2->N1", "N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 2C — always Open Deviated Shunt (ODS), never closed. "
            "Flow pattern: N2->N1 AND N2->N3->N1 (two branches from N2). "
            "Branch 1: N2->N1 directly (retrograde GSV trunk drains to deep). "
            "Branch 2: N2->N3->N1 (via tributary to deep system). "
            "Escape Point: EP N2->N2 (perforator entry, same as Type 2B). "
            "SFJ COMPETENT — no EP N1->N2 clip anywhere. "
            "Key distinction from Type 2B: Type 2C also has RP N2->N1 (secondary GSV trunk reflux). "
            "Key distinction from Type 1+2: Type 1+2 has EP N1->N2 (SFJ incompetent); "
            "Type 2C has only EP N2->N2 (SFJ competent). "
            "Classification signal: EP N2->N2 EXISTS + RP N3 EXISTS + RP N2->N1 EXISTS + NO EP N1->N2. "
            "Ligation: ligate perforator entry (highest EP N2->N2) AND all RP N2->N1 sites along GSV. "
        ),
    },
    {
        "label": "Type 3 — SFJ entry, no RP on N2 trunk",
        "shunt_types": ["Type 3"],
        "flow_patterns": ["N1->N2->N3->N1", "N1->N2->N3->N2->N1"],
        "text": (
            "CHIVA Shunt Type 3 — Closed Shunt (CS). "
            "Flow pattern: N1->N2->N3->N1 or N1->N2->N3->N2->N1. "
            "Extended flow: N1->N2->N3->N2->N3->N1 or N1->N3->N2->N3->N1. "
            "Escape Point (EP): N1->N2 (same as Type 1, at SFJ or Hunterian perforator). "
            "SFJ INCOMPETENT — EP N1->N2 exists. "
            "CRITICAL DISTINCTION from Type 1: NO re-entry point (RP) along the N2 saphenous trunk. "
            "Blood flow goes N1->N2->N3 then re-enters N1 via a tributary (N3), bypassing N2. "
            "An N3 or N4 tributary interposes itself between the EP and the RP, "
            "bypassing an N2-continent segment. "
            "N2 between EP and N3 branch is CONTINENT (competent). "
            "Also features EP N2->N3 (escape into tributary) with RP N3->N1 or RP N3->N2. "
            "N1->N2 reflux evoked by both Paranà/squeezing diastole AND Valsalva systole "
            "but N1 doesn't drain directly — flows through N3. "
            "Classification signal: EP N1->N2 EXISTS + EP N2->N3 EXISTS + RP N3->N1 or RP N3->N2 + NO RP N2->N1. "
            "OR: EP N1->N2 + EP N2->N3 + RP N3->N1 + RP N2->N1 with eliminationTest='No Reflux'. "
            "Ligation (single RP at N3): ligate EP at N2->N3, follow up 6-12 months. "
            "Ligation (multiple RP at N3): ligate every refluxing tributary at N2 junction (CHIVA 2). "
        ),
    },
    {
        "label": "Type 4 — N1 to N3 via perforator",
        "shunt_types": ["Type 4"],
        "flow_patterns": ["N1->N3->N2->N1"],
        "text": (
            "CHIVA Shunt Type 4 — Closed Shunt (CS). Also called '4 Perforator Shunt'. "
            "Flow pattern: N1->N3->N2->N1 or N1->B->N3->N2->N1 (B = Bone Perforator). "
            "Escape Point (EP): N1->N3 via a perforator or pelvic point (bypasses N2 initially). "
            "Blood enters N3 (tributary) directly from N1 (deep system), then drains into N1 via N2. "
            "Re-entry: from N2 saphenous trunk back to N1. "
            "Pelvic variant flow: P->N2->N1 or P->N3->N2->N1 (P = Pelvic Perforator). "
            "Key feature: N1->N3 escape (not N1->N2), so N2 is involved only in the return path. "
            "Classify as Shunt 4 (not '4 Perforator Shunt') in output. "
        ),
    },
    {
        "label": "Type 5 — N1 to N3 via perforator, complex return",
        "shunt_types": ["Type 5"],
        "flow_patterns": ["N1->N3->N2->N3->N1"],
        "text": (
            "CHIVA Shunt Type 5 — Closed Shunt (CS). Also called '5 Perforator Shunt'. "
            "Flow pattern: N1->N3->N2->N3->N1 or N1->N3->N2->N3->N2->N1 "
            "or N1->B->N3->N2->N3->N1 (B = Bone Perforator). "
            "Escape Point (EP): N1->N3 via pelvic or perforator point. "
            "More complex than Type 4: involves double N3 involvement (N3 appears twice in flow). "
            "Blood: N1->N3->N2->N3->N1 — enters N3, goes through N2, back to N3, then N1. "
            "Pelvic variant: P->N2->N3->N1 or P->N3->N2->N3->N1 (P = Pelvic Perforator). "
            "Classify as Shunt 5 (not '5 Perforator Shunt') in output. "
        ),
    },
    {
        "label": "Type 6 — bone perforator direct",
        "shunt_types": ["Type 6"],
        "flow_patterns": ["N1->B->N3->N1"],
        "text": (
            "CHIVA Shunt Type 6. "
            "Flow pattern: N1->B->N3->N1 (B = Bone Perforator). "
            "The simplest direct perforator shunt. "
            "Blood flows from N1 (deep) through a bone perforator B directly into N3 (tributary), "
            "then returns to N1. "
            "No N2 (saphenous trunk) involvement in the shunt pathway. "
        ),
    },
    {
        "label": "Closed vs Open Shunt definitions",
        "shunt_types": ["Type 1", "Type 2A", "Type 3", "Type 4", "Type 5"],
        "flow_patterns": [],
        "text": (
            "CHIVA Shunt Classification — Open vs Closed Shunt definitions. "
            "Closed Shunt (CS): a refluxing pattern constituting a vicious re-circulation. "
            "The blood follows a closed loop — it escapes at EP and re-enters the venous system at RP. "
            "All closed shunts: Type 1, Type 1+2, Type 3, Type 4, Type 5 (and Type 2A when CS). "
            "Open Deviated Shunt (ODS): blood escapes from the shunt without forming a closed loop. "
            "Always ODS: Type 2B, Type 2C. "
            "Type 2A is the ONLY Type 2 sub-type that can be either ODS or CS. "
            "Shunt 1 CS: EP at SFJ N1->N2, RP at saphenous trunk N2->N1, "
            "transmural pressure increases due to hydrostatic pressure and additional reflux flow. "
        ),
    },
    {
        "label": "Anatomical node definitions and EP/RP meaning",
        "shunt_types": [],
        "flow_patterns": [],
        "text": (
            "CHIVA Anatomy — Node definitions for shunt classification. "
            "N1 = Deep venous system (femoral vein, popliteal vein). "
            "N2 = Saphenous trunk: Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV). "
            "N3 = Tributaries / superficial branches of saphenous system. "
            "N4 = Sub-tributaries (branches of N3). "
            "B = Bone Perforator (direct perforator from deep to superficial bypassing saphenous). "
            "P = Pelvic Perforator (pelvic origin, connects pelvic veins to superficial system). "
            "EP (Escape Point): antegrade / physiological direction flow FROM deep/saphenous TO superficial. "
            "Pathological EP: blood escaping from deeper compartment to higher one. "
            "RP (Re-entry Point): retrograde flow — blood re-entering deep system. "
            "SFJ (Saphenofemoral Junction): where GSV meets femoral vein. "
            "posYRatio ≤ 0.098 = SFJ zone. 0.098 < posYRatio ≤ 0.353 = Hunterian perforator zone. "
            "SFJ incompetent if and only if clip has fromType=N1 AND toType=N2 (EP N1->N2). "
            "EP N2->N2 always = perforator entry, SFJ competent regardless of posYRatio. "
        ),
    },
    {
        "label": "Shunt flow pattern quick reference table",
        "shunt_types": ["Type 1", "Type 1+2", "Type 2A", "Type 2B", "Type 2C", "Type 3", "Type 4", "Type 5", "Type 6"],
        "flow_patterns": ["N1->N2->N1", "N1->N2->N1", "N2->N3->N2", "N2->N3->N1", "N1->N2->N3->N1", "N1->N3->N2->N1", "N1->N3->N2->N3->N1"],
        "text": (
            "CHIVA Shunt Classification — Flow Pattern Quick Reference. "
            "Shunt Type 1 Flow: N1->N2->N1. (EP N1->N2 at SFJ, RP N2->N1 on GSV trunk). "
            "Shunt Type 1+2 Flow: N1->N2->N1 AND N2->N3->N1 (two simultaneous branches). "
            "Shunt Type 2 General Flow: N2->N3->N2. "
            "Shunt Type 2A Flow: N2->N3->N2 AND N2->N3->N1 (ODS or CS). "
            "Shunt Type 2B Flow: N2->N3->N2 AND N2->N3->N1 (always ODS, EP N2->N2 perforator). "
            "Shunt Type 2C Flow: N2->N1 AND N2->N3->N1 (always ODS, EP N2->N2 perforator + RP N2->N1). "
            "Shunt Type 3 Flow: N1->N2->N3->N1 or N1->N2->N3->N2->N1. "
            "Shunt Type 4 Flow: N1->N3->N2->N1 or N1->B->N3->N2->N1. "
            "Shunt Type 4 Pelvic Flow: P->N2->N1 or P->N3->N2->N1. "
            "Shunt Type 5 Flow: N1->N3->N2->N3->N1 or N1->N3->N2->N3->N2->N1 or N1->B->N3->N2->N3->N1. "
            "Shunt Type 5 Pelvic Flow: P->N2->N3->N1 or P->N3->N2->N3->N1. "
            "Shunt Type 6 Flow: N1->B->N3->N1. "
            "ODS = Open Deviated Shunt (2B, 2C always). CS = Closed Shunt (1, 1+2, 3, 4, 5). "
            "2A can be ODS or CS (only exception). "
        ),
    },
    {
        "label": "Type 1 vs Type 3 discriminator",
        "shunt_types": ["Type 1", "Type 3"],
        "flow_patterns": ["N1->N2->N1", "N1->N2->N3->N1"],
        "text": (
            "CHIVA Discriminator: Type 1 versus Type 3. "
            "Both Type 1 and Type 3 have EP N1->N2 (SFJ or Hunterian perforator — incompetent). "
            "CRITICAL DIFFERENCE: "
            "Type 1: RP is on N2 saphenous trunk (RP N2->N1). No RP at N3 tributary. "
            "Type 3: NO RP along N2 saphenous trunk. RP is at N3 tributary (RP N3->N1 or RP N3->N2). "
            "Type 3 has EP N2->N3 (tributary escape) whereas Type 1 does not. "
            "Type 3 flow: N1->N2->N3->N1 — blood bypasses N2 re-entry via N3. "
            "N2 segment between EP and N3 branch is continent (competent) in Type 3. "
            "If elimination test on RP N2->N1 shows 'Reflux' → Type 1+2 (not Type 3). "
            "If elimination test shows 'No Reflux' → Type 3. "
        ),
    },
    {
        "label": "Type 2B vs Type 2C vs Type 1+2 discriminator",
        "shunt_types": ["Type 2B", "Type 2C", "Type 1+2"],
        "flow_patterns": ["N2->N3->N1", "N2->N1", "N1->N2->N1"],
        "text": (
            "CHIVA Discriminator: Type 2B vs Type 2C vs Type 1+2. "
            "Type 2B: EP N2->N2 (perforator) + RP N3 + NO RP N2->N1. SFJ competent. "
            "Type 2C: EP N2->N2 (perforator) + RP N3 + RP N2->N1. SFJ competent. "
            "Type 1+2: EP N1->N2 (SFJ entry) + EP N2->N3 + RP N3 + RP N2->N1. SFJ INCOMPETENT. "
            "Key test — is SFJ incompetent? "
            "YES (EP N1->N2 exists) → could be Type 1+2 (if also has RP N2->N1 + eliminationTest=Reflux). "
            "NO (only EP N2->N2 or EP N2->N3 without EP N1->N2) → could be Type 2B or 2C. "
            "With RP N2->N1 and NO EP N1->N2 → Type 2C. "
            "With NO RP N2->N1 and NO EP N1->N2 → Type 2B. "
        ),
    },
    {
        "label": "No shunt detected criteria",
        "shunt_types": [],
        "flow_patterns": [],
        "text": (
            "CHIVA No Shunt Detected criteria. "
            "No shunt is present when: No RP (re-entry point) clips anywhere in the assessment. "
            "EP-only pattern: only EP N2->N2 clips exist with NO RP clips of any kind → NO SHUNT. "
            "Case D: No RP in any clip → NO SHUNT DETECTED. No ligation needed. "
            "The EP alone (antegrade flow) without retrograde return does not constitute a shunt. "
            "A shunt requires both an EP and an RP to complete the pathological flow circuit. "
        ),
    },
]


def build_synthetic_type_chunks() -> list[dict]:
    """Strategy 4: pre-crafted chunks, one per shunt type + discriminators."""
    chunks = []
    for idx, entry in enumerate(SYNTHETIC_TYPE_CHUNKS):
        chunks.append({
            "text": entry["text"],
            "source": "synthetic_knowledgebase",
            "chunk_type": "synthetic_type",
            "shunt_types": entry["shunt_types"],
            "flow_patterns": entry["flow_patterns"],
            "chunk_index": idx,
        })
    return chunks


def build_flow_pattern_lookup_chunks() -> list[dict]:
    """Strategy 5: minimal lookup chunks — just flow notation + type label.
    Optimised for queries that contain only the N1->N2->N3 pattern string."""
    entries = [
        ("Type 1",   "N1->N2->N1",                "EP N1->N2 SFJ or Hunterian. RP N2->N1 saphenous trunk. Closed shunt."),
        ("Type 1+2", "N1->N2->N1 and N2->N3->N1", "EP N1->N2 and EP N2->N3. RP N2->N1 and RP N3->N1. eliminationTest=Reflux. Closed shunt."),
        ("Type 2A",  "N2->N3->N2 and N2->N3->N1", "EP N2->N3. No SFJ incompetence. RP N3->N2 or N3->N1. ODS or CS."),
        ("Type 2B",  "N2->N3->N2 and N2->N3->N1", "EP N2->N2 perforator. No N1->N2. RP N3. No RP N2->N1. Always ODS."),
        ("Type 2C",  "N2->N1 and N2->N3->N1",     "EP N2->N2 perforator. No N1->N2. RP N3 and RP N2->N1. Always ODS."),
        ("Type 3",   "N1->N2->N3->N1",             "EP N1->N2 SFJ. EP N2->N3. RP N3->N1 or N3->N2. No RP N2->N1. Closed shunt."),
        ("Type 4",   "N1->N3->N2->N1",             "EP N1->N3 perforator. RP via N2 back to N1. Closed shunt. Perforator shunt."),
        ("Type 5",   "N1->N3->N2->N3->N1",         "EP N1->N3 pelvic or perforator. Double N3 in flow. Closed shunt."),
        ("Type 6",   "N1->B->N3->N1",              "EP N1->B bone perforator. Direct N3->N1 return. No N2 involvement."),
    ]
    chunks = []
    for idx, (stype, flow, note) in enumerate(entries):
        text = (
            f"Shunt {stype}: flow pattern {flow}. {note} "
            f"CHIVA classification: {stype}. Flow: {flow}."
        )
        chunks.append({
            "text": text,
            "source": "flow_pattern_lookup",
            "chunk_type": "flow_pattern",
            "shunt_types": [stype],
            "flow_patterns": detect_flow_patterns(flow),
            "chunk_index": idx,
        })
    return chunks


# ── Ingestion pipeline ────────────────────────────────────────────────────────

def ingest():
    logger.info("=" * 70)
    logger.info(f"SHUNT CLASSIFICATION v2 INGESTION -> {QDRANT_SHUNT_V2_COLLECTION}")
    logger.info("Advanced chunking: docx-paragraph + context-window + synthetic + flow-lookup + rules")
    logger.info("=" * 70)

    all_chunks: list[dict] = []

    # 1a. Docx paragraphs
    if DOCX_PATH.exists():
        paragraphs = extract_docx_paragraphs(DOCX_PATH)
        docx_chunks = build_docx_paragraph_chunks(paragraphs)
        ctx_chunks = build_context_window_chunks(paragraphs, window=2)
        logger.info(f"  Docx paragraph chunks : {len(docx_chunks)}")
        logger.info(f"  Context-window chunks : {len(ctx_chunks)}")
        all_chunks.extend(docx_chunks)
        all_chunks.extend(ctx_chunks)
    else:
        logger.error(f"Docx not found: {DOCX_PATH}")

    # 1b. Synthetic type-specific chunks
    synth_chunks = build_synthetic_type_chunks()
    logger.info(f"  Synthetic type chunks : {len(synth_chunks)}")
    all_chunks.extend(synth_chunks)

    # 1c. Flow-pattern lookup chunks
    flow_chunks = build_flow_pattern_lookup_chunks()
    logger.info(f"  Flow-pattern chunks   : {len(flow_chunks)}")
    all_chunks.extend(flow_chunks)

    # 1d. CHIVA rules
    if RULES_PATH.exists():
        rules_text = load_rules_text(RULES_PATH)
        rules_chunks = build_rules_chunks(rules_text)
        logger.info(f"  Rules chunks          : {len(rules_chunks)}")
        all_chunks.extend(rules_chunks)
    else:
        logger.warning(f"Rules file not found: {RULES_PATH} — skipping")

    logger.info(f"\nTotal chunks: {len(all_chunks)}")

    # 2. Embed
    logger.info(f"\nEmbedding {len(all_chunks)} chunks with {OLLAMA_EMBEDDING_MODEL}...")
    embeddings: list[np.ndarray] = []
    t0 = time.time()
    for i, item in enumerate(all_chunks):
        if (i + 1) % 10 == 0 or (i + 1) == len(all_chunks):
            logger.info(f"  {i + 1}/{len(all_chunks)} embedded...")
        embeddings.append(get_embedding(item["text"]))
        time.sleep(0.05)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # 3. Upsert into Qdrant
    logger.info(f"\nUpserting into Qdrant collection '{QDRANT_SHUNT_V2_COLLECTION}'...")

    # Clean-rebuild strategy for changing vector dimension on local Qdrant:
    #  1. Remove from meta.json via API (delete_collection) — this is what prevents
    #     the fresh client from loading the old collection definition.
    #  2. close() the old client handle.
    #  3. Manually wipe the orphaned on-disk folder using shutil; if the SQLite file
    #     is locked by another process, fall back to os.system rd /s /q (Windows).
    #  4. Open a fresh client and create the collection at the new dimension.
    init_client = get_qdrant_client()
    existing = [c.name for c in init_client.get_collections().collections]
    if QDRANT_SHUNT_V2_COLLECTION in existing:
        init_client.delete_collection(QDRANT_SHUNT_V2_COLLECTION)
        logger.info(f"  Deleted collection from meta.json: '{QDRANT_SHUNT_V2_COLLECTION}'")
    init_client.close()
    del init_client

    collection_dir = Path(QDRANT_PATH) / "collection" / QDRANT_SHUNT_V2_COLLECTION
    if collection_dir.exists():
        try:
            shutil.rmtree(collection_dir)
            logger.info(f"  Removed stale on-disk folder: {collection_dir}")
        except PermissionError:
            # Windows: SQLite WAL lock not released yet — forceful OS-level delete.
            import subprocess
            subprocess.run(["cmd", "/c", f"rd /s /q \"{collection_dir}\""], check=False)
            logger.info(f"  Force-removed folder (PermissionError fallback): {collection_dir}")

    client = get_qdrant_client()
    client.create_collection(
        collection_name=QDRANT_SHUNT_V2_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
    logger.info(f"  Created collection (dim={EMBEDDING_DIMENSION}, Cosine)")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text":          all_chunks[i]["text"],
                "source":        all_chunks[i]["source"],
                "chunk_type":    all_chunks[i]["chunk_type"],
                "shunt_types":   all_chunks[i]["shunt_types"],
                "flow_patterns": all_chunks[i]["flow_patterns"],
                "chunk_index":   all_chunks[i]["chunk_index"],
            },
        )
        for i in range(len(all_chunks))
    ]

    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start: start + batch_size]
        client.upsert(collection_name=QDRANT_SHUNT_V2_COLLECTION, points=batch)
        logger.info(f"  Upserted {min(start + batch_size, len(points))}/{len(points)} points")

    info = client.get_collection(QDRANT_SHUNT_V2_COLLECTION)
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Collection : {QDRANT_SHUNT_V2_COLLECTION}")
    logger.info(f"  Points     : {info.points_count}")
    logger.info(f"  Chunk types: docx_paragraph, context_window, synthetic_type, flow_pattern, rules")
    logger.info(f"  Storage    : {QDRANT_PATH}")

    # Breakdown by chunk type
    type_counts: dict[str, int] = {}
    for c in all_chunks:
        ct = c["chunk_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    for ct, count in sorted(type_counts.items()):
        logger.info(f"    {ct:22s}: {count}")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
