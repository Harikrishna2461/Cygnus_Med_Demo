"""
Medical Book Ingestion Script
Loads text, chunks it, generates embeddings, and stores in FAISS
"""

import os
import json
import numpy as np
import faiss
import pickle
import requests
import time
import logging
from pathlib import Path
from PyPDF2 import PdfReader
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_DIMENSION,
    SAMPLE_DATA_PATH,
    LOG_FILE,
    LOG_LEVEL
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


def get_embedding(text):
    """Get embedding from Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        embedding = np.array(response.json()["embedding"], dtype=np.float32)
        return embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks by word count
    chunk_size and overlap are approximate word counts
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        pdf_text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            total_pages = len(pdf_reader.pages)
            logger.info(f"  PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                if (i + 1) % 10 == 0 or (i + 1) == total_pages:
                    logger.info(f"  Extracting page {i + 1}/{total_pages}...")
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
        
        if pdf_text.strip():
            logger.info(f"✓ Extracted {len(pdf_text)} characters from PDF")
            return pdf_text
        else:
            logger.warning("PDF extraction resulted in empty text")
            return ""
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        return ""


def load_medical_text():
    """Load medical text from files (PDF + sample)"""
    all_text = ""
    
    # Look for PDF files in project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    pdf_files = [f for f in os.listdir(project_root) if f.endswith('.pdf')]
    
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF file(s) in project root")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(project_root, pdf_file)
            pdf_text = extract_pdf_text(pdf_path)
            if pdf_text:
                all_text += "\n\n" + "=" * 80 + "\n"
                all_text += f"SOURCE: {pdf_file}\n"
                all_text += "=" * 80 + "\n\n"
                all_text += pdf_text
    
    # Also include sample data for reference
    logger.info("Including sample medical knowledge base...")
    all_text += "\n\n" + "=" * 80 + "\n"
    all_text += "SOURCE: Built-in Medical Knowledge Base\n"
    all_text += "=" * 80 + "\n\n"
    all_text += create_sample_medical_text()
    
    if all_text.strip():
        logger.info(f"✓ Loaded {len(all_text)} total characters from all sources")
        return all_text
    else:
        logger.warning("No text loaded from any source. Using sample data only.")
        return create_sample_medical_text()


def create_sample_medical_text():
    """Create comprehensive sample medical text about venous disease"""
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
2. Assess for neo-vascularization
3. Evaluate arteriovenous communications
4. Plan revision strategy accordingly

ULTRASOUND PROBE GUIDANCE PRINCIPLES

Scanning technique for optimal visualization:
- Start with B-mode transverse view at known reference points
- Systematically scan longitudinally along vein course
- Use color Doppler to confirm vein identity
- Optimize gain and frequency for target vein depth
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

The CHIVA principle emphasizes preservation of saphenous veins when possible while correcting hemodynamic abnormalities through strategic intervention at entry points and re-entry pathways.
"""
    
    # Save for future use
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    with open(SAMPLE_DATA_PATH, 'w') as f:
        f.write(sample_text)
    logger.info(f"Created sample medical text at {SAMPLE_DATA_PATH}")
    return sample_text


def ingest_and_index():
    """Main ingestion pipeline"""
    logger.info("=" * 70)
    logger.info("MEDICAL TEXT INGESTION AND FAISS INDEXING")
    logger.info("=" * 70)
    
    # Load text
    logger.info("\n[1/4] Loading medical text...")
    text = load_medical_text()
    logger.info(f"✓ Loaded {len(text)} characters")
    
    # Split into chunks
    logger.info("\n[2/4] Chunking text (target: {}-{} words)...".format(
        CHUNK_SIZE - CHUNK_OVERLAP, CHUNK_SIZE))
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    # Generate embeddings
    logger.info("\n[3/4] Generating embeddings with Ollama...")
    logger.info(f"Using model: {OLLAMA_EMBEDDING_MODEL}")
    
    embeddings_list = []
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            logger.info(f"  Processing chunk {i + 1}/{len(chunks)}...")
        
        embedding = get_embedding(chunk)
        embeddings_list.append(embedding)
        time.sleep(0.1)  # Small delay to avoid rate limiting
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Generated {len(embeddings_list)} embeddings in {elapsed:.2f}s")
    
    # Create FAISS index
    logger.info("\n[4/4] Creating FAISS index...")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    logger.info(f"Embeddings shape: {embeddings_array.shape}")
    
    # Create index
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(embeddings_array)
    
    # Save index and metadata
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    logger.info(f"✓ Saved FAISS index: {FAISS_INDEX_PATH}")
    logger.info(f"✓ Saved metadata: {FAISS_METADATA_PATH}")
    
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nIndex Statistics:")
    logger.info(f"  - Total chunks: {len(chunks)}")
    logger.info(f"  - Embedding dimension: {EMBEDDING_DIMENSION}")
    logger.info(f"  - Index type: FAISS IndexFlatL2")
    logger.info(f"\nReady to use with Flask backend!")


if __name__ == '__main__':
    try:
        ingest_and_index()
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        raise
