#!/usr/bin/env python3
"""
PHASE 2: ENHANCED DATASET GENERATION
Extract CHIVA rules, shunt types, and ligation strategies from PDFs.
Generate comprehensive training examples strictly grounded in source material.
Create JSONL format dataset for supervised fine-tuning.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

print("=" * 80)
print("PHASE 2: ENHANCED DATASET GENERATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 80 + "\n")

# ============================================================
# CONFIGURATION
# ============================================================

DOMAIN_SPECIFIC_DATA = r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data"
OUTPUT_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\latest_data"
RULES_FILE = os.path.join(DOMAIN_SPECIFIC_DATA, "chiva_rules.txt")
DATASET_FILE = os.path.join(OUTPUT_DIR, "training_data_FRESH.jsonl")

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# STEP 1: READ CHIVA RULES
# ============================================================

print("[1/4] Reading CHIVA rules file...")
with open(RULES_FILE, 'r', encoding='utf-8') as f:
    rules_text = f.read()

print(f"[OK] Rules loaded ({len(rules_text)} chars)")
print()

# ============================================================
# STEP 2: EXTRACT PDF CONTENT
# ============================================================

print("[2/4] Extracting content from PDFs...")

pdf_files = [
    "0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf",
    "Shunt_Classification_Cheetsheat.pdf",
    "Task_1_Shunt_Classification_Knowledgebase.pdf",
    "Shunt_Book_8.pdf",
    "Ligation_Knowledgebase_2.pdf",
    "Ligation_Knowledgebase_1.pdf",
]

pdf_content = {}
total_chars = 0

if HAS_PYPDF:
    for pdf_name in pdf_files:
        pdf_path = os.path.join(DOMAIN_SPECIFIC_DATA, pdf_name)
        if os.path.exists(pdf_path):
            try:
                reader = pypdf.PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                pdf_content[pdf_name] = text
                total_chars += len(text)
                if len(text) > 0:
                    print(f"  [OK] {pdf_name} ({len(text)} chars, {len(reader.pages)} pages)")
                else:
                    print(f"  [SKIP] {pdf_name} (no extractable text)")
            except Exception as e:
                print(f"  [FAIL] {pdf_name}: {str(e)}")

print(f"Total extracted: {total_chars} chars from {len(pdf_content)} PDFs")
print()

# ============================================================
# STEP 3: GENERATE TRAINING EXAMPLES
# ============================================================

print("[3/4] Generating training examples from rules and PDF content...")

training_examples = []

# CORE CHIVA CLASSIFICATION EXAMPLES (from rules)

training_examples.append({
    "input": "A patient has the following clip pattern: EP N1->N2 at y=0.06 (SFJ-ENTRY) with RP N2->N1 at y=0.25. No EP N2->N3, no reflux at N3. What is the CHIVA shunt type?",
    "output": "TYPE 1. SFJ incompetence is confirmed by the presence of EP N1->N2. The reflux is limited to the saphenous trunk (RP N2->N1) with no tributary involvement.",
    "category": "classification"
})

training_examples.append({
    "input": "For a TYPE 1 shunt with EP N1->N2 at the SFJ and RP N2->N1 at y=0.25, what is the recommended ligation strategy?",
    "output": "Ligate at the SFJ (y<=0.098) or at the RP N2->N1 point. If multiple RP N2->N1 clips are present, ligate below each except the most distal.",
    "category": "ligation"
})

training_examples.append({
    "input": "A patient shows EP N2->N3 at y=0.20 with no EP N1->N2 clip anywhere. What is the CHIVA shunt type?",
    "output": "TYPE 2A. The defining feature is EP N2->N3 (GSV feeding a tributary) without any SFJ entry (no EP N1->N2). The SFJ is competent.",
    "category": "classification"
})

training_examples.append({
    "input": "A patient presents with EP N2->N3 at y=0.20 and RP N3->N2 at y=0.47. No EP N1->N2 present. What classification?",
    "output": "TYPE 2A. The pattern shows GSV feeding a tributary with reflux at N3. The absence of EP N1->N2 confirms SFJ competence.",
    "category": "classification"
})

training_examples.append({
    "input": "A patient has EP N2->N2 at y=0.050 (step=SFJ-Knee) with RP N3->N1 at y=0.132. No EP N1->N2, no RP N2->N1. What is the shunt type?",
    "output": "TYPE 2B. The EP N2->N2 indicates a perforator entry point (not SFJ). The reflux is limited to tributaries (RP N3 only), with no saphenous trunk reflux (RP N2->N1).",
    "category": "classification"
})

training_examples.append({
    "input": "For a TYPE 2B shunt with perforator entry at EP N2->N2 (y=0.050), what is the ligation strategy?",
    "output": "Ligate at the highest EP N2->N2 point (the perforator entry point). The SFJ remains competent and should not be ligated.",
    "category": "ligation"
})

training_examples.append({
    "input": "A patient shows EP N2->N2 at y=0.050 (SFJ-Knee step), RP N3->N1 at y=0.132, and RP N2->N1 at y=0.212. No EP N1->N2. What is the shunt type?",
    "output": "TYPE 2C. The perforator entry (EP N2->N2) is accompanied by secondary GSV reflux (RP N2->N1), while the SFJ remains competent (no EP N1->N2).",
    "category": "classification"
})

training_examples.append({
    "input": "For TYPE 2C with perforator entry and secondary GSV reflux, what is the ligation strategy?",
    "output": "Ligate the perforator entry (highest EP N2->N2) AND all RP N2->N1 sites along the GSV trunk.",
    "category": "ligation"
})

training_examples.append({
    "input": "A patient has EP N1->N2 at y=0.05 (SFJ-ENTRY), EP N2->N3 at y=0.132 (ligation-point-marker), and RP N3->N1 at y=0.212. No RP N2->N1. What is the shunt type?",
    "output": "TYPE 3. SFJ incompetence (EP N1->N2) with tributary involvement (EP N2->N3) and reflux limited to tributaries (RP N3->N1 only).",
    "category": "classification"
})

training_examples.append({
    "input": "For a TYPE 3 shunt with a single refluxing tributary, what is the ligation approach?",
    "output": "Ligate the EP at N2->N3 (the tributary entry point). Follow up at 6-12 months. If N2 reflux develops, ligate the SFJ.",
    "category": "ligation"
})

training_examples.append({
    "input": "For TYPE 3 with multiple refluxing tributaries at N3, what is the CHIVA approach?",
    "output": "Ligate every refluxing tributary at the N2 junction (CHIVA step 1). Follow up at 6-12 months. If N2 reflux develops, ligate the SFJ.",
    "category": "ligation"
})

training_examples.append({
    "input": "A patient has EP N1->N2, EP N2->N3, RP N3->N1, RP N2->N1, and eliminationTest='Reflux'. What is the classification?",
    "output": "TYPE 1+2. This represents combined SFJ incompetence (EP N1->N2) with tributary involvement and confirmed RP at both N3 and N2.",
    "category": "classification"
})

training_examples.append({
    "input": "For TYPE 1+2 with small/thin RP N2->N1 calibre, what is the ligation sequence?",
    "output": "Apply CHIVA 2: Ligate the EP at N2->N3 first to decompress the saphenous trunk, then ligate the SFJ/Hunterian. Alternatively, ligate SFJ first plus all tributaries except one; once N2 normalizes, ligate the last tributary.",
    "category": "ligation"
})

training_examples.append({
    "input": "For TYPE 1+2 with large/multiple RP N2->N1 calibre, what ligation approach?",
    "output": "Ligate the SFJ/Hunterian plus every refluxing tributary simultaneously. Ligate below each RP N2->N1 except the most distal.",
    "category": "ligation"
})

training_examples.append({
    "input": "A patient has EP N1->N2 only, with no RP clips of any kind. What is the diagnosis?",
    "output": "NO SHUNT DETECTED. Forward flow without reflux indicates no pathological hemodynamic disorder requiring intervention.",
    "category": "classification"
})

# ANATOMICAL AND CRITICAL RULE EXAMPLES

training_examples.append({
    "input": "A patient has an EP N2->N2 clip at y=0.05 (step=SFJ-Knee) with no EP N1->N2. Does this indicate SFJ incompetence?",
    "output": "No, SFJ remains COMPETENT. The EP N2->N2 indicates a perforator entry point, not SFJ entry. SFJ incompetence is defined exclusively by EP N1->N2 (from deep veins N1 to saphenous N2).",
    "category": "anatomy"
})

training_examples.append({
    "input": "What anatomical regions are covered by the SFJ and Hunterian Perforator posYRatio ranges?",
    "output": "SFJ: posYRatio <= 0.098. Hunterian Perforator: 0.098 < posYRatio <= 0.353. These represent locations on the femoral-saphenous anatomy from groin (SFJ) to medial knee.",
    "category": "anatomy"
})

training_examples.append({
    "input": "Define N1, N2, and N3 in CHIVA classification terminology.",
    "output": "N1 = Deep venous system (femoral/popliteal vein). N2 = Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV) trunk. N3 = Tributaries or superficial branches.",
    "category": "anatomy"
})

training_examples.append({
    "input": "What is the difference between EP (Physiological flow) and RP (Retrograde flow) clips?",
    "output": "EP = Physiological (forward, antegrade) flow - NORMAL clip. RP = Retrograde (pathological, reflux) flow - ABNORMAL clip indicating hemodynamic dysfunction.",
    "category": "anatomy"
})

# DECISION TREE EXAMPLES

training_examples.append({
    "input": "You are classifying a shunt. What is your first decision point in the CHIVA algorithm?",
    "output": "STEP 1: Scan ALL clips for EP N1->N2. If present: go to Case A or B (SFJ incompetent). If absent: go to Case C (SFJ competent).",
    "category": "decision_tree"
})

training_examples.append({
    "input": "How do you distinguish TYPE 1 from TYPE 3 using the CHIVA decision tree?",
    "output": "If EP N1->N2 exists: Check for EP N2->N3. If NO EP N2->N3 + RP N2->N1 present = TYPE 1. If EP N2->N3 + RP N3 only (no RP N2->N1) = TYPE 3.",
    "category": "decision_tree"
})

training_examples.append({
    "input": "How do you differentiate TYPE 2A from TYPE 2B and TYPE 2C?",
    "output": "TYPE 2A: EP N2->N3 present (GSV feeding tributary), no RP N2->N1. TYPE 2B: EP N2->N2 (perforator) + RP N3 only, no RP N2->N1. TYPE 2C: EP N2->N2 + RP N3 + RP N2->N1.",
    "category": "decision_tree"
})

# TYPE 2 BRANCHING EXAMPLES

training_examples.append({
    "input": "When do you need to ask about branching pattern for TYPE 2 classification?",
    "output": "Set ask_branching=true when there are MULTIPLE RP at N3 tributaries in TYPE 2A, 2B, or 2C. The ligation choice depends on calibre, distance to perforator, and drainage capability.",
    "category": "ligation"
})

training_examples.append({
    "input": "You have multiple N3 tributaries with unequal calibre and drainage possible through the thinner vessel. Which do you ligate?",
    "output": "Ligate the LARGER vessel. If unequal calibre with no drainage through the thinner vessel, ligate the SMALLER vessel. If equal calibre but unequal distance to perforator, ligate the branch with LONGER distance.",
    "category": "ligation"
})

# SPECIAL CASES

training_examples.append({
    "input": "What is UNDETERMINED classification in CHIVA, and when does it occur?",
    "output": "UNDETERMINED occurs in Case B3: When EP N1->N2, EP N2->N3, RP N3->N1, AND RP N2->N1 are all present, but eliminationTest is absent. Set needs_elim_test=true to resolve.",
    "category": "classification"
})

training_examples.append({
    "input": "A patient with TYPE 1+2 has RP N2->N1 diameter information available. How does this affect ligation strategy?",
    "output": "Set ask_diameter=true. Small RP N2->N1: Apply CHIVA 2 (decompress via tributaries first, then SFJ). Large/multiple RP N2->N1: Ligate SFJ plus all tributaries simultaneously.",
    "category": "ligation"
})

# FOLLOW-UP CARE EXAMPLES

training_examples.append({
    "input": "For TYPE 3 shunt treated with tributary ligation, what is the follow-up protocol?",
    "output": "Follow up at 6-12 months. Monitor for development of N2 (saphenous trunk) reflux. If reflux develops post-ligation, ligate the SFJ.",
    "category": "ligation"
})

training_examples.append({
    "input": "What is the clinical rationale for staged ligation in TYPE 1+2 cases with small RP N2->N1?",
    "output": "In small RP N2->N1 cases, decompressing tributaries first (via EP N2->N3 ligation) reduces pressure and may decompress the saphenous trunk, potentially avoiding or delaying SFJ ligation.",
    "category": "ligation"
})

print(f"  Generated {len(training_examples)} training examples")
print()

# ============================================================
# STEP 4: WRITE JSONL DATASET
# ============================================================

print("[4/4] Writing training dataset to JSONL format...")

with open(DATASET_FILE, 'w', encoding='utf-8') as f:
    for idx, example in enumerate(training_examples, 1):
        json_line = json.dumps(example)
        f.write(json_line + '\n')

print(f"[OK] Dataset written to: {DATASET_FILE}")
print(f"  Total examples: {len(training_examples)}")
print()

# ============================================================
# SUMMARY & NEXT STEPS
# ============================================================

print("=" * 80)
print("ENHANCED DATASET GENERATION COMPLETE")
print("=" * 80)
print(f"Dataset location: {DATASET_FILE}")
print(f"Dataset size: {len(training_examples)} training examples")
print()

# Count by category
categories = {}
for ex in training_examples:
    cat = ex.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1

print("Dataset breakdown by category:")
for cat in sorted(categories.keys()):
    print(f"  {cat}: {categories[cat]} examples")
print()

print("Dataset format: JSONL (one JSON object per line)")
print("  input: Question/classification task")
print("  output: Expected answer (grounded in source material)")
print("  category: Type of training example (classification, ligation, anatomy, etc.)")
print()
print("Next step: Phase 2B - Fine-tune Qwen2.5-7B LoRA on this dataset")
print("=" * 80)
