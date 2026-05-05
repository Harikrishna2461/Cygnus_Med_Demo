"""
Extract training data from CHIVA medical PDFs and prepare instruction-response pairs
for fine-tuning Mistral-7B as a domain expert model.

This script:
1. Extracts text from PDFs (Ligation_Knowledgebase, Shunt_Book, Classification Cheetsheet)
2. Creates Q&A pairs focused on shunt classification and ligation planning
3. Formats data for supervised fine-tuning
4. Combines domain knowledge with CHIVA decision rules
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PDF_DIRECTORY = Path("./books_articles")
OUTPUT_DIRECTORY = Path("./training_data_from_pdfs")
OUTPUT_FILE = OUTPUT_DIRECTORY / "training_pairs_from_medical_literature.jsonl"

# Document patterns to extract
CLASSIFICATION_PATTERNS = {
    "Type 1": r"(?:Type\s+1|SFJ\s+incompetent.*?retrograde.*?N2.*?N1)",
    "Type 2A": r"(?:Type\s+2A|SFJ\s+competent.*?N2.*?N3)",
    "Type 2B": r"(?:Type\s+2B|perforator.*?tributary)",
    "Type 2C": r"(?:Type\s+2C|perforator.*?N2.*?N1)",
    "Type 3": r"(?:Type\s+3|GSV.*?tributary.*?reflux)",
}

LIGATION_KEYWORDS = [
    "ligation", "ablation", "treatment", "procedure", "intervention",
    "EVLA", "foam sclerotherapy", "closure", "elimination"
]


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"  Extracting from {pdf_path.name} ({len(reader.pages)} pages)...")

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    print(f"    Warning: Could not extract page {page_num + 1}: {e}")

        return text
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return ""


def extract_all_pdfs() -> Dict[str, str]:
    """Extract text from all PDFs in the books_articles directory."""
    documents = {}

    if not PDF_DIRECTORY.exists():
        print(f"Error: {PDF_DIRECTORY} does not exist")
        return documents

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}")
        return documents

    print(f"Found {len(pdf_files)} PDF files. Extracting text...")
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            documents[pdf_path.stem] = text
            print(f"  ✓ Extracted {len(text)} characters")

    return documents


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION INSTRUCTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def create_classification_instructions(documents: Dict[str, str]) -> List[Dict]:
    """
    Create instruction-response pairs for shunt classification from extracted documents.
    """
    pairs = []

    for doc_name, text in documents.items():
        # Extract paragraphs related to shunt types
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if len(para) < 50:  # Skip very short paragraphs
                continue

            # Check if this paragraph discusses shunt types or classification
            for shunt_type, pattern in CLASSIFICATION_PATTERNS.items():
                if re.search(pattern, para, re.IGNORECASE):
                    # Create instruction: "What is Type 1 CHIVA shunt?"
                    instruction = f"Based on CHIVA classification rules, explain {shunt_type} CHIVA venous shunt."

                    # Use the paragraph as response
                    response = para.strip()

                    pairs.append({
                        "instruction": instruction,
                        "input": "",
                        "output": response,
                        "source": doc_name,
                        "type": "classification"
                    })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# LIGATION INSTRUCTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def create_ligation_instructions(documents: Dict[str, str]) -> List[Dict]:
    """
    Create instruction-response pairs for ligation planning from extracted documents.
    """
    pairs = []

    for doc_name, text in documents.items():
        paragraphs = text.split("\n\n")

        for i, para in enumerate(paragraphs):
            if len(para) < 100:
                continue

            # Check if paragraph contains ligation-related content
            if any(keyword in para.lower() for keyword in LIGATION_KEYWORDS):
                # Try to form a Q&A pair
                # If next paragraph exists, use it as context
                context = paragraphs[i+1] if i+1 < len(paragraphs) else ""

                # Create instruction based on content
                if "Type" in para and "treatment" in para.lower():
                    match = re.search(r"(Type\s+[0-9A-C]+)", para)
                    if match:
                        shunt_type = match.group(1)
                        instruction = f"What is the recommended ligation strategy for {shunt_type} CHIVA shunt?"
                    else:
                        instruction = "What are the ligation planning considerations for this shunt?"
                else:
                    instruction = "Explain the ligation planning approach described here."

                response = para.strip()
                if context:
                    response += f"\n\nAdditional context: {context.strip()}"

                pairs.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                    "source": doc_name,
                    "type": "ligation"
                })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED INSTRUCTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

CHIVA_RULES_INSTRUCTIONS = [
    {
        "instruction": "How do you determine if the SFJ (Saphenofemoral Junction) is competent or incompetent in CHIVA classification?",
        "output": "SFJ competence is determined by the presence of antegrade (EP) flow from N1 to N2:\n\n- SFJ is INCOMPETENT if: EP N1→N2 exists (antegrade flow from deep vein to GSV at SFJ)\n- SFJ is COMPETENT if: NO EP N1→N2 (no entry from deep vein at SFJ; perforators may show EP N2→N2 instead)\n\nThis is the critical first check in CHIVA classification.",
        "type": "rule_based"
    },
    {
        "instruction": "When SFJ is incompetent (EP N1→N2 present), what are the possible CHIVA types?",
        "output": "When SFJ is incompetent (EP N1→N2 exists), the shunt can be:\n\n1. TYPE 1: If EP N2→N3 is absent (direct retrograde N2→N1, no tributary involvement)\n2. TYPE 3: If EP N2→N3 exists AND retrograde flow is only in tributaries (RP N3 only)\n3. TYPE 1+2 (Complex): If both EP N2→N3 AND RP N2→N1 + RP N3 exist (requires elimination test to differentiate)",
        "type": "rule_based"
    },
    {
        "instruction": "What are the classification rules when SFJ is competent (no EP N1→N2)?",
        "output": "When SFJ is competent (no EP N1→N2), classification depends on perforator and tributary involvement:\n\n1. TYPE 2A: EP N2→N3 exists (competent GSV with tributary incompetence)\n2. TYPE 2B: Perforator entry (EP N2→N2) with retrograde N3 only (RP N3)\n3. TYPE 2C: Perforator entry (EP N2→N2) with retrograde in both perforator and tributaries (RP N2→N1 + RP N3)\n4. NO SHUNT: Perforator entry (EP N2→N2) but no retrograde flow anywhere",
        "type": "rule_based"
    },
    {
        "instruction": "When should you use the elimination test in CHIVA classification?",
        "output": "The elimination test is used specifically for Type 3 vs Type 1+2 differentiation when:\n\n- SFJ is incompetent (EP N1→N2 present)\n- EP N2→N3 exists (tributary reflux pathway)\n- BOTH RP N3 AND RP N2→N1 are present (complex reflux pattern)\n\nIf elimination test result is:\n- 'Reflux' → Type 1+2 (both reflux pathways are significant)\n- 'No Reflux' → Type 3 (tributary is the primary pathology)",
        "type": "rule_based"
    }
]


def get_rule_based_pairs() -> List[Dict]:
    """Return embedded CHIVA rule-based instruction pairs."""
    return CHIVA_RULES_INSTRUCTIONS


# ─────────────────────────────────────────────────────────────────────────────
# ANATOMICAL REFERENCE INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

ANATOMICAL_INSTRUCTIONS = [
    {
        "instruction": "What do N1, N2, and N3 represent in CHIVA anatomical classification?",
        "output": "CHIVA anatomical nodes:\n\n- N1 = Deep Venous System (femoral and popliteal veins)\n- N2 = Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV) main trunk\n- N3 = Tributary branches and superficial branches\n\nEP flows (antegrade, normal) move along these pathways normally, while RP flows (retrograde, abnormal reflux) indicate incompetence.",
        "type": "anatomy"
    },
    {
        "instruction": "What is the SFJ and what anatomical zone does it cover in CHIVA classification?",
        "output": "SFJ (Saphenofemoral Junction):\n\n- Location: Junction where Great Saphenous Vein (N2) meets the deep femoral vein (N1)\n- Anatomical zone: posYRatio ≤ 0.098 (proximal 9.8% of limb)\n- Clinical significance: Most common site of reflux entry in CHIVA Type 1 shunts\n- Classification role: Presence of EP N1→N2 at SFJ indicates SFJ incompetence, the starting point for CHIVA classification",
        "type": "anatomy"
    },
    {
        "instruction": "What is the Hunterian perforator and how does it relate to CHIVA classification?",
        "output": "Hunterian Perforator (Adductor Canal Perforator):\n\n- Location: Between N2 (GSV) and N1 (deep system) in the mid-thigh\n- Anatomical zone: 0.098 < posYRatio ≤ 0.353 (between 9.8% and 35.3% of limb length)\n- CHIVA significance: Perforator incompetence (EP N2→N2) is the entry point for Type 2B and Type 2C shunts\n- Classification role: When EP flow enters at Hunterian perforator, SFJ is assumed competent",
        "type": "anatomy"
    }
]


def get_anatomy_pairs() -> List[Dict]:
    """Return anatomical reference instruction pairs."""
    return ANATOMICAL_INSTRUCTIONS


# ─────────────────────────────────────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_training_pairs(pairs: List[Dict]) -> None:
    """Save training pairs to JSONL file."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\n✓ Saved {len(pairs)} training pairs to {OUTPUT_FILE}")


def create_summary_report(pairs: List[Dict]) -> None:
    """Create a summary report of training data."""
    report_path = OUTPUT_DIRECTORY / "training_data_summary.txt"

    type_counts = {}
    source_counts = {}

    for pair in pairs:
        pair_type = pair.get("type", "unknown")
        source = pair.get("source", "rule_based")

        type_counts[pair_type] = type_counts.get(pair_type, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    with open(report_path, 'w') as f:
        f.write("TRAINING DATA SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total pairs: {len(pairs)}\n\n")

        f.write("By Type:\n")
        for ptype, count in sorted(type_counts.items()):
            f.write(f"  - {ptype}: {count}\n")

        f.write("\nBy Source:\n")
        for source, count in sorted(source_counts.items()):
            f.write(f"  - {source}: {count}\n")

        f.write("\nSample Pairs:\n")
        f.write("-" * 60 + "\n")
        for i, pair in enumerate(pairs[:3]):
            f.write(f"\nPair {i+1}:\n")
            f.write(f"Type: {pair.get('type', 'unknown')}\n")
            f.write(f"Instruction: {pair.get('instruction', '')[:80]}...\n")
            f.write(f"Output: {pair.get('output', '')[:80]}...\n")

    print(f"✓ Saved summary to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("CHIVA Medical Literature Training Data Extraction")
    print("=" * 60)

    # Extract from PDFs
    print("\n1. Extracting from PDFs...")
    documents = extract_all_pdfs()

    if not documents:
        print("\nNo documents found. Proceeding with rule-based and anatomical data only.")
    else:
        print(f"\n✓ Extracted {len(documents)} documents")

    # Generate instruction pairs
    print("\n2. Generating instruction-response pairs...")

    all_pairs = []

    # From documents
    if documents:
        classification_pairs = create_classification_instructions(documents)
        ligation_pairs = create_ligation_instructions(documents)
        all_pairs.extend(classification_pairs)
        all_pairs.extend(ligation_pairs)
        print(f"  Classification pairs: {len(classification_pairs)}")
        print(f"  Ligation pairs: {len(ligation_pairs)}")

    # Rule-based
    rule_pairs = get_rule_based_pairs()
    all_pairs.extend(rule_pairs)
    print(f"  Rule-based pairs: {len(rule_pairs)}")

    # Anatomical
    anatomy_pairs = get_anatomy_pairs()
    all_pairs.extend(anatomy_pairs)
    print(f"  Anatomical pairs: {len(anatomy_pairs)}")

    # Save
    print(f"\n3. Saving {len(all_pairs)} total pairs...")
    save_training_pairs(all_pairs)
    create_summary_report(all_pairs)

    print("\n✓ Training data preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Review {OUTPUT_FILE}")
    print(f"2. Customize pairs as needed for your use case")
    print(f"3. Use with: python training_lora_improved.py --data {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
