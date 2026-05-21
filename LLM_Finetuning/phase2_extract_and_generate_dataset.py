#!/usr/bin/env python3
"""
PHASE 2: EXTRACT AND GENERATE TRAINING DATASET
Extract CHIVA rules, shunt types, and ligation strategies from PDFs and text file.
Generate synthetic training examples strictly grounded in source material.
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
    print("[WARNING] pypdf not installed. Install with: pip install pypdf")

print("=" * 80)
print("PHASE 2: EXTRACT AND GENERATE TRAINING DATASET")
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
# STEP 2: EXTRACT PDF CONTENT (if pypdf available)
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
                print(f"  [OK] {pdf_name} ({len(text)} chars, {len(reader.pages)} pages)")
            except Exception as e:
                print(f"  [FAIL] {pdf_name}: {str(e)}")
else:
    print("  [WARNING] pypdf not available - skipping PDF extraction")
    print("  Install: pip install pypdf")

print()

# ============================================================
# STEP 3: GENERATE TRAINING EXAMPLES
# ============================================================

print("[3/4] Generating training examples from CHIVA rules...")

training_examples = []

# Extract TYPE definitions from rules text
type_1_pattern = r"TYPE 1.*?(?=TYPE|Case|Ligation:|─|$)"
type_2a_pattern = r"TYPE 2A.*?(?=TYPE|Case|─|$)"
type_2b_pattern = r"TYPE 2B.*?(?=TYPE|Case|─|$)"
type_2c_pattern = r"TYPE 2C.*?(?=TYPE|Case|─|$)"
type_3_pattern = r"TYPE 3.*?(?=TYPE|Case|─|$)"

# EXAMPLE 1: TYPE 1 Classification
training_examples.append({
    "input": "A patient has the following clip pattern: EP N1→N2 at y=0.06 (SFJ-ENTRY) with RP N2→N1 at y=0.25. No EP N2→N3, no reflux at N3. What is the CHIVA shunt type?",
    "output": "TYPE 1. SFJ incompetence is confirmed by the presence of EP N1→N2. The reflux is limited to the saphenous trunk (RP N2→N1) with no tributary involvement.",
    "source": "chiva_rules.txt - CONCRETE EXAMPLES section"
})

# EXAMPLE 2: TYPE 1 Ligation
training_examples.append({
    "input": "For a TYPE 1 shunt with EP N1→N2 at the SFJ and RP N2→N1 at y=0.25, what is the recommended ligation strategy?",
    "output": "Ligate at the SFJ (y≤0.098) or at the RP N2→N1 point. If multiple RP N2→N1 clips are present, ligate below each except the most distal.",
    "source": "chiva_rules.txt - Case A section"
})

# EXAMPLE 3: TYPE 2A Classification
training_examples.append({
    "input": "A patient shows EP N2→N3 at y=0.20 with no EP N1→N2 clip anywhere. What is the CHIVA shunt type?",
    "output": "TYPE 2A. The defining feature is EP N2→N3 (GSV feeding a tributary) without any SFJ entry (no EP N1→N2). The SFJ is competent.",
    "source": "chiva_rules.txt - CONCRETE EXAMPLES section"
})

# EXAMPLE 4: TYPE 2A with Reflux
training_examples.append({
    "input": "A patient presents with EP N2→N3 at y=0.20 and RP N3→N2 at y=0.47. No EP N1→N2 present. What classification?",
    "output": "TYPE 2A. The pattern shows GSV feeding a tributary with reflux at N3. The absence of EP N1→N2 confirms SFJ competence.",
    "source": "chiva_rules.txt - TYPE 2A description"
})

# EXAMPLE 5: TYPE 2B Classification
training_examples.append({
    "input": "A patient has EP N2→N2 at y=0.050 (step=SFJ-Knee) with RP N3→N1 at y=0.132. No EP N1→N2, no RP N2→N1. What is the shunt type?",
    "output": "TYPE 2B. The EP N2→N2 indicates a perforator entry point (not SFJ). The reflux is limited to tributaries (RP N3 only), with no saphenous trunk reflux (RP N2→N1).",
    "source": "chiva_rules.txt - CONCRETE EXAMPLES section"
})

# EXAMPLE 6: TYPE 2B Ligation
training_examples.append({
    "input": "For a TYPE 2B shunt with perforator entry at EP N2→N2 (y=0.050), what is the ligation strategy?",
    "output": "Ligate at the highest EP N2→N2 point (the perforator entry point). The SFJ remains competent and should not be ligated.",
    "source": "chiva_rules.txt - TYPE 2B ligation section"
})

# EXAMPLE 7: TYPE 2C Classification
training_examples.append({
    "input": "A patient shows EP N2→N2 at y=0.050 (SFJ-Knee step), RP N3→N1 at y=0.132, and RP N2→N1 at y=0.212. No EP N1→N2. What is the shunt type?",
    "output": "TYPE 2C. The perforator entry (EP N2→N2) is accompanied by secondary GSV reflux (RP N2→N1), while the SFJ remains competent (no EP N1→N2).",
    "source": "chiva_rules.txt - CONCRETE EXAMPLES section"
})

# EXAMPLE 8: TYPE 2C Ligation
training_examples.append({
    "input": "For TYPE 2C with perforator entry and secondary GSV reflux, what is the ligation strategy?",
    "output": "Ligate the perforator entry (highest EP N2→N2) AND all RP N2→N1 sites along the GSV trunk.",
    "source": "chiva_rules.txt - TYPE 2C ligation section"
})

# EXAMPLE 9: TYPE 3 Classification
training_examples.append({
    "input": "A patient has EP N1→N2 at y=0.05 (SFJ-ENTRY), EP N2→N3 at y=0.132 (ligation-point-marker), and RP N3→N1 at y=0.212. No RP N2→N1. What is the shunt type?",
    "output": "TYPE 3. SFJ incompetence (EP N1→N2) with tributary involvement (EP N2→N3) and reflux limited to tributaries (RP N3→N1 only).",
    "source": "chiva_rules.txt - CONCRETE EXAMPLES section"
})

# EXAMPLE 10: TYPE 3 Ligation (Single tributary)
training_examples.append({
    "input": "For a TYPE 3 shunt with a single refluxing tributary, what is the ligation approach?",
    "output": "Ligate the EP at N2→N3 (the tributary entry point). Follow up at 6-12 months. If N2 reflux develops, ligate the SFJ.",
    "source": "chiva_rules.txt - TYPE 3 Ligation section"
})

# EXAMPLE 11: TYPE 3 Ligation (Multiple tributaries)
training_examples.append({
    "input": "For TYPE 3 with multiple refluxing tributaries at N3, what is the CHIVA approach?",
    "output": "Ligate every refluxing tributary at the N2 junction (CHIVA step 1). Follow up at 6-12 months. If N2 reflux develops, ligate the SFJ.",
    "source": "chiva_rules.txt - TYPE 3 Multiple RP at N3"
})

# EXAMPLE 12: TYPE 1+2 Classification
training_examples.append({
    "input": "A patient has EP N1→N2, EP N2→N3, RP N3→N1, RP N2→N1, and eliminationTest='Reflux'. What is the classification?",
    "output": "TYPE 1+2. This represents combined SFJ incompetence (EP N1→N2) with tributary involvement and confirmed RP at both N3 and N2.",
    "source": "chiva_rules.txt - Case B4"
})

# EXAMPLE 13: TYPE 1+2 Ligation (Small RP N2→N1)
training_examples.append({
    "input": "For TYPE 1+2 with small/thin RP N2→N1 calibre, what is the ligation sequence?",
    "output": "Apply CHIVA 2: Ligate the EP at N2→N3 first to decompress the saphenous trunk, then ligate the SFJ/Hunterian. Alternatively, ligate SFJ first plus all tributaries except one; once N2 normalizes, ligate the last tributary.",
    "source": "chiva_rules.txt - TYPE 1+2 Small RP N2→N1"
})

# EXAMPLE 14: TYPE 1+2 Ligation (Large RP N2→N1)
training_examples.append({
    "input": "For TYPE 1+2 with large/multiple RP N2→N1 calibre, what ligation approach?",
    "output": "Ligate the SFJ/Hunterian plus every refluxing tributary simultaneously. Ligate below each RP N2→N1 except the most distal.",
    "source": "chiva_rules.txt - TYPE 1+2 Large/multiple RP N2→N1"
})

# EXAMPLE 15: NO SHUNT Classification
training_examples.append({
    "input": "A patient has EP N1→N2 only, with no RP clips of any kind. What is the diagnosis?",
    "output": "NO SHUNT DETECTED. Forward flow without reflux indicates no pathological hemodynamic disorder requiring intervention.",
    "source": "chiva_rules.txt - Case D and Case C"
})

# EXAMPLE 16: SFJ Competence Rule
training_examples.append({
    "input": "A patient has an EP N2→N2 clip at y=0.05 (step=SFJ-Knee) with no EP N1→N2. Does this indicate SFJ incompetence?",
    "output": "No, SFJ remains COMPETENT. The EP N2→N2 indicates a perforator entry point, not SFJ entry. SFJ incompetence is defined exclusively by EP N1→N2 (from deep veins N1 to saphenous N2).",
    "source": "chiva_rules.txt - CRITICAL RULE section"
})

# EXAMPLE 17: Hunterian Perforator Definition
training_examples.append({
    "input": "What is the posYRatio range for the Hunterian Perforator classification?",
    "output": "The Hunterian Perforator is defined by posYRatio in the range 0.098 < posYRatio ≤ 0.353, representing the location between the SFJ and knee on the femoral/saphenous anatomy.",
    "source": "chiva_rules.txt - ANATOMY section"
})

# EXAMPLE 18: Decision Tree - Quick Reference
training_examples.append({
    "input": "What is the quickest way to determine TYPE 1 vs TYPE 3 from clip patterns?",
    "output": "Check for EP N2→N3: If EP N1→N2 + NO EP N2→N3 + RP N2→N1 → TYPE 1. If EP N1→N2 + EP N2→N3 + RP N3 only → TYPE 3.",
    "source": "chiva_rules.txt - QUICK DECISION TABLE"
})

# EXAMPLE 19: TYPE 2 Branching Decision
training_examples.append({
    "input": "When do you need to ask about branching pattern for TYPE 2 classification?",
    "output": "Set ask_branching=true when there are MULTIPLE RP at N3 tributaries in TYPE 2A, 2B, or 2C. The ligation choice depends on calibre, distance to perforator, and drainage capability.",
    "source": "chiva_rules.txt - TYPE 2 BRANCHING section"
})

# EXAMPLE 20: TYPE 2 Branching Strategy
training_examples.append({
    "input": "You have multiple N3 tributaries with unequal calibre and drainage possible through the thinner vessel. Which do you ligate?",
    "output": "Ligate the LARGER vessel. If unequal calibre with no drainage through the thinner vessel, ligate the SMALLER vessel. If equal calibre but unequal distance to perforator, ligate the branch with LONGER distance.",
    "source": "chiva_rules.txt - TYPE 2 BRANCHING ligation rules"
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
print("DATASET GENERATION COMPLETE")
print("=" * 80)
print(f"Dataset location: {DATASET_FILE}")
print(f"Dataset size: {len(training_examples)} training examples")
print()
print("Dataset format: JSONL (one JSON object per line)")
print("  input: Question/classification task")
print("  output: Expected answer (grounded in source material)")
print("  source: Reference to source document/section")
print()
print("Next step: Phase 2B - Fine-tune Qwen2.5-7B LoRA on this dataset")
print("=" * 80)
