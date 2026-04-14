"""
Medical Book Ingestion Script
Loads text, chunks it, generates embeddings via Ollama,
and stores everything in Qdrant (local file-based — no Docker required).
"""

import os
import numpy as np
import requests
import time
import logging
from pathlib import Path
from PyPDF2 import PdfReader
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
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    SAMPLE_DATA_PATH,
    LOG_FILE,
    LOG_LEVEL,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant client factory (local file-based or remote)
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    if QDRANT_HOST:
        logger.info(f"Connecting to remote Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    os.makedirs(QDRANT_PATH, exist_ok=True)
    logger.info(f"Using local Qdrant storage at {QDRANT_PATH}")
    return QdrantClient(path=QDRANT_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return np.array(response.json()["embeddings"][0], dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PDF / text loading
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        pdf_text = ""
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total = len(reader.pages)
            logger.info(f"  {total} pages")
            for i, page in enumerate(reader.pages):
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(f"  Page {i + 1}/{total}...")
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
        if pdf_text.strip():
            logger.info(f"✓ Extracted {len(pdf_text)} characters")
            return pdf_text
        logger.warning("PDF extraction yielded empty text")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def load_medical_text() -> str:
    """Load all medical text: PDFs in project root + built-in knowledge base."""
    all_text = ""

    project_root = os.path.join(os.path.dirname(__file__), '..')
    pdf_files = sorted(f for f in os.listdir(project_root) if f.endswith('.pdf'))

    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF(s) in project root: {pdf_files}")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(project_root, pdf_file)
            pdf_text = extract_pdf_text(pdf_path)
            if pdf_text:
                all_text += "\n\n" + "=" * 80 + "\n"
                all_text += f"SOURCE: {pdf_file}\n"
                all_text += "=" * 80 + "\n\n"
                all_text += pdf_text
    else:
        logger.info("No PDFs found in project root — using built-in knowledge base only")

    logger.info("Including built-in medical knowledge base...")
    all_text += "\n\n" + "=" * 80 + "\n"
    all_text += "SOURCE: Built-in Medical Knowledge Base\n"
    all_text += "=" * 80 + "\n\n"
    all_text += create_sample_medical_text()

    logger.info(f"✓ Total text loaded: {len(all_text)} characters")
    return all_text or create_sample_medical_text()


def create_sample_medical_text() -> str:
    """Built-in venous disease knowledge base."""
    sample_text = """
VASCULAR MEDICINE AND HAEMODYNAMIC ASSESSMENT

CHAPTER 1: VENOUS SYSTEM ANATOMY AND PHYSIOLOGY

The venous system consists of superficial, deep and perforating veins. The superficial system includes the great saphenous vein (GSV), small saphenous vein (SSV), and their tributaries. The deep veins comprise the femoral, popliteal, tibial and peroneal veins. Perforating veins connect the superficial and deep systems.

The great saphenous vein runs from the medial ankle to the groin, joining the femoral vein at the saphenofemoral junction. The small saphenous vein ascends the posterior calf and typically joins the popliteal vein. Normal venous flow is centripetal (toward the heart) and unidirectional, maintained by bicuspid venous valves.

HAEMODYNAMIC PRINCIPLES IN VENOUS DISEASE

Three-compartment model: N1, N2, and N3 compartments.
- N1: Normal veins with competent valves
- N2: Veins with valve incompetence but no reflux below the entry point
- N3: Veins with reflux extending into the deep system

Entry Point (EP): The proximal location where reflux enters the incompetent superficial vein.

Re-entry Point (RP): Where the incompetent vein re-enters the deep system or competent superficial vein.

CHAPTER 2: CLASSIFICATION OF VENOUS SHUNTS

TYPE 1 SHUNT (Simple Reflux)
- Incompetent valve at the saphenofemoral junction
- Reflux in the great saphenous vein extending into tributaries
- No reflux in perforating veins
- Treatment: CHIVA-type closure of tributaries or saphenofemoral ligation

TYPE 2 SHUNT (Reflux with Perforating Involvement)
- Incompetence at saphenofemoral junction
- Reflux in the GSV
- Perforating veins in the medial thigh showing incompetence
- Treatment: CHIVA with hemodynamic correction at perforators

TYPE 3 SHUNT (Perforating Vein Origin)
- Incompetent perforating veins as primary entry point
- Incompetence at the mid-thigh or knee level
- Reflux in the saphenous vein secondary to perforator incompetence
- Treatment: Perforator ligation at entry point

ULTRASOUND ASSESSMENT PARAMETERS

1. Vein Diameter Measurement
   - Measured in transverse section at rest
   - GSV normal: < 3mm
   - GSV abnormal: > 5mm
   - Clinical significance: Progressive dilatation indicates disease progression

2. Reflux Duration
   - Normal: < 0.5 seconds
   - Pathological: > 1 second
   - Assessed with Valsalva or proximal compression

3. Flow Velocity
   - Normal antegrade flow: 0.3-0.9 m/s
   - Reflux velocity indicates severity
   - Low velocity reflux (<0.2 m/s) suggests chronicity

4. Valve Cusp Appearance
   - Normal: Thin, echogenic, coapt completely
   - Diseased: Thick, irregular, separated cusps

HEMODYNAMIC CLASSIFICATION (Widmer)

C0: No visible signs of venous disease
C1: Telangiectasias or reticular veins
C2: Varicose veins > 3mm
C3: Edema without skin changes
C4: Pigmentation or eczema
C5: Healed venous ulcer
C6: Active venous ulcer

ETIOLOGICAL CLASSIFICATION

Primary (Essential) Venous Insufficiency
- Valve incompetence without prior thrombosis
- Genetic predisposition
- Environmental factors: prolonged standing, pressure

Secondary Venous Insufficiency
- Post-thrombotic syndrome
- Iliac vein obstruction
- Pelvic mass compression

TREATMENT STRATEGIES

CONSERVATIVE MANAGEMENT
- Compression therapy: 20-30 mmHg for C2-C4
- Leg elevation
- Exercise programs
- Pharmacotherapy: Flavonoids, diosmin

ABLATION TECHNIQUES
- Radiofrequency ablation (RFA): 85° C, 120 seconds per segment
- Endovenous laser ablation (EVLA): 980nm or 1470nm wavelength
- Cyanoacrylate glue: VenaSeal system
- Sclerotherapy: Sodium tetradecyl sulfate or polidocanol

CHIVA METHOD (Conservative Hemodynamic Treatment)
- Hemodynamically meaningful truncation
- Preserves saphenous vein function for grafting
- Addresses hypoplasia by creating preferential reflux pathways
- Entry point treatment with tributary ligation
- Often avoids saphenofemoral or saphenopopliteal junction closure

INDICATIONS FOR TREATMENT
- C2-C6 with hemodynamic abnormality
- Progressive symptoms
- Recurrent ulceration
- Patient preference
- Cosmetic concerns

CONTRAINDICATIONS FOR ENDOVENOUS ABLATION
- Deep venous thrombosis in target vein
- Severe arterial insufficiency
- Systemic infection
- Pregnancy (relative)
- Inability to comply with post-procedural care

ASSESSMENT OF PROBE POSITIONING IN ULTRASOUND

Optimal probe positioning for venous assessment:
- Transverse (axial) view: Shows circular or oval vein cross-section
- Sagittal (longitudinal) view: Shows valves and flow direction
- Probe angulation: 15-30 degrees for optimal valve visualization
- Compression maneuver: Gentle posterior pressure to assess compressibility
- Dynamic assessment: Valsalva, limb elevation, muscle contraction

Doppler mode selection:
- B-mode (gray-scale): Structural assessment
- PW-Doppler: Single-site velocity measurement
- CW-Doppler: High-velocity flow assessment
- Color Doppler: Flow mapping and reflux detection

CLINICAL DECISION SUPPORT GUIDELINES

For GSV incompetence with C2 symptoms:
1. Confirm reflux duration > 1 second
2. Assess diameter and degree of incompetence
3. Evaluate for perforator involvement
4. Classify as Type 1, 2, or 3 shunt
5. Select treatment: CHIVA, ablation, or ligation

For below-knee insufficiency:
1. Assess small saphenous vein
2. Evaluate medial and lateral calf perforators
3. Determine if isolated perforator disease
4. Consider endovenous ablation of SSV with selective perforator treatment

For recurrent varicose veins:
1. Complete mapping of previous treatment site
2. Assess for neo-vascularisation
3. Evaluate arteriovenous communications
4. Plan revision strategy accordingly

ULTRASOUND PROBE GUIDANCE PRINCIPLES

Scanning technique for optimal visualisation:
- Start with B-mode transverse view at known reference points
- Systematically scan longitudinally along vein course
- Use color Doppler to confirm vein identity
- Optimise gain and frequency for target vein depth
- Use Valsalva to assess valve competence

Navigation targets in lower extremity venous mapping:
- Saphenofemoral junction: Medial groin at skin crease
- Saphenopopliteal junction: Popliteal fossa, medial aspect
- Medial calf perforators: Located medial gastrocnemius
- Lateral calf perforators: Located between peroneal and soleus
- Hunter's canal perforators: Located along medial thigh

Probe movement terminology:
- Medial/Lateral: Toward midline or away from midline
- Proximal/Distal: Toward heart or away from heart
- Superficial/Deep: Toward skin or toward deep structures
- Angular: Rotating probe plane

SUMMARY OF KEY HAEMODYNAMIC CONCEPTS

The N1-N2-N3 classification provides hemodynamic context:
- N1 compartment: Normal veins with intact valves
- N2 compartment: Incompetent vein with preserved distal valve
- N3 compartment: Segmental reflux involving multiple valve levels

Entry Point determination is critical for surgical planning:
- Primary entry point: Most proximal location of incompetence
- Secondary pathways: Tributaries and perforators carrying reflux
- Hemodynamic significance: Whether reflux affects outflow capacity

The CHIVA principle emphasises preservation of saphenous veins when possible while correcting hemodynamic abnormalities through strategic intervention at entry points and re-entry pathways.
"""
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    with open(SAMPLE_DATA_PATH, 'w') as f:
        f.write(sample_text)
    logger.info(f"Built-in knowledge base written to {SAMPLE_DATA_PATH}")
    return sample_text


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest_and_index():
    logger.info("=" * 70)
    logger.info("MEDICAL TEXT INGESTION → QDRANT INDEXING")
    logger.info("=" * 70)

    # 1. Load text
    logger.info("\n[1/4] Loading medical text (PDFs + built-in knowledge base)...")
    text = load_medical_text()
    logger.info(f"✓ {len(text)} characters loaded")

    # 2. Chunk
    logger.info(f"\n[2/4] Chunking (~{CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap)...")
    chunks = split_into_chunks(text)
    logger.info(f"✓ {len(chunks)} chunks created")

    # 3. Embed
    logger.info(f"\n[3/4] Generating embeddings with Ollama ({OLLAMA_EMBEDDING_MODEL})...")
    embeddings = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            logger.info(f"  {i + 1}/{len(chunks)} chunks embedded...")
        embeddings.append(get_embedding(chunk))
        time.sleep(0.05)
    logger.info(f"✓ {len(embeddings)} embeddings in {time.time()-t0:.1f}s")

    # 4. Upsert into Qdrant
    logger.info("\n[4/4] Upserting into Qdrant...")
    client = get_qdrant_client()

    # (Re)create collection — wipes any previous data in this collection
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        logger.info(f"  Dropped existing collection '{QDRANT_COLLECTION}'")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
    logger.info(f"  Created collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIMENSION}, Cosine)")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i], "chunk_index": i},
        )
        for i in range(len(chunks))
    ]

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        logger.info(f"  Upserted {min(start + batch_size, len(points))}/{len(points)} points")

    info = client.get_collection(QDRANT_COLLECTION)
    logger.info(f"\n{'='*70}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Collection : {QDRANT_COLLECTION}")
    logger.info(f"  Points     : {info.points_count}")
    logger.info(f"  Dimension  : {EMBEDDING_DIMENSION}")
    logger.info(f"  Distance   : Cosine")
    logger.info(f"  Storage    : {QDRANT_PATH}")
    logger.info(f"\nReady to use with Flask backend!")


if __name__ == '__main__':
    try:
        ingest_and_index()
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        raise
