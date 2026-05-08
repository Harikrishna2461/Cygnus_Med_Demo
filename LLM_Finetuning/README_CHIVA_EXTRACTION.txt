================================================================================
CHIVA MEDICAL TRAINING DATA EXTRACTION - COMPREHENSIVE GUIDE
================================================================================

PROJECT: Fine-tuning Mistral-7B on CHIVA venous disease classification
DATE: 2026-05-06
LOCATION: C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\

================================================================================
WHAT WAS EXTRACTED
================================================================================

TOTAL: 1,214 unique clinical training examples extracted from medical PDFs

These examples contain:
- CHIVA venous shunt classifications (Type 1, 2, 2A, 2B, 2C, 3, 4, 5, 1+2, No shunt)
- Hemodynamic flow patterns (N1, N2, N3, EP, RP, reflux descriptions)
- Ligation and treatment strategies (CHIVA 1, CHIVA 2, high tie, etc.)
- Clinical presentations and findings
- Real-world case descriptions from published medical literature

================================================================================
PRIMARY OUTPUT FILE
================================================================================

FILE: CHIVA_Training_Dataset_Final.json (1.54 MB)

Structure:
{
  "metadata": {
    "total_examples": 1214,
    "examples_by_document": { ... },
    "examples_by_shunt_type": { ... },
    "extraction_date": "2026-05-06",
    "description": "...",
    "classification_types": [...]
  },
  "examples": [
    {
      "source_document": "...",
      "page_or_section": "Page 123",
      "instruction": "Clinical case description (400 chars)",
      "output": "Full clinical text",
      "shunt_type": "Type classification",
      "ligation_strategy": "Treatment approach (if available)",
      "clinical_notes": "Flow patterns and findings (if available)"
    },
    ...
  ]
}

All examples are REAL clinical content extracted directly from PDFs.
NO examples were synthesized or invented.

================================================================================
SUPPLEMENTARY FILES
================================================================================

1. CHIVA_Training_Dataset_Final.csv (0.30 MB)
   - CSV format for viewing in Excel/spreadsheet applications
   - Fields: source_document, page_or_section, shunt_type, ligation_strategy, clinical_notes, instruction_snippet
   - Easier to browse and filter than JSON

2. chiva_clinical_examples.json (1.28 MB, 1,016 examples)
   - Initial comprehensive extraction from main PDFs
   - Contains more raw content than final deduplicated set
   - Useful if you want to trace which extraction yielded which example

3. chiva_refined_clinical_cases.json (0.26 MB, 206 examples)
   - Highest confidence examples (explicitly identified Type classifications)
   - Subset of 1,214 examples
   - Good starting point if you want to focus on most certain cases first

4. EXTRACTION_SUMMARY.txt
   - Detailed extraction report with statistics and methodology
   - Explains "Unknown" category and limitations
   - Documents extraction process and tools used

5. Domain_Specific_Data/ folder
   - Original source PDFs (in Domain_Specific_Data subfolder)
   - Reference for locating specific examples by page number

================================================================================
DATA BREAKDOWN
================================================================================

BY DOCUMENT:
- 0-Saphenous-Vein-Sparing-Strategies.pdf: 865 examples (71.2%)
- Saphenous-Vein-Sparing.pdf: 167 examples (13.8%)
- Shunt_Classification_Cheetsheat.pdf: 82 examples (6.8%)
- Ligation_Knowledgebase_1.pdf: 57 examples (4.7%)
- Shunt_Book_8.pdf: 43 examples (3.5%)

BY SHUNT TYPE:
- Unknown/Unclassified: 809 (66.6%) - has flow patterns but no explicit type
- Type 3: 139 (11.4%)
- Type 2: 74 (6.1%)
- Type 1: 53 (4.4%)
- Type 5: 51 (4.2%)
- Type 4: 41 (3.4%)
- Type 1+2: 29 (2.4%)
- Type 2C: 6 (0.5%)
- Type 2B: 5 (0.4%)
- No shunt: 4 (0.3%)
- Type 2A: 3 (0.2%)

================================================================================
HOW TO USE THIS DATA
================================================================================

FOR MODEL FINE-TUNING:

1. Load the JSON file:
   import json
   with open('CHIVA_Training_Dataset_Final.json', 'r', encoding='utf-8') as f:
       data = json.load(f)

   examples = data['examples']

2. Create training examples (instruction-output pairs):
   for example in examples:
       instruction = example['instruction']
       shunt_type = example['shunt_type']
       ligation = example['ligation_strategy']

       # Format for your fine-tuning framework
       training_pair = {
           "input": instruction,
           "output": f"Shunt Type: {shunt_type}\nLigation: {ligation or 'N/A'}"
       }

3. Filter by confidence (optional):
   # Use only explicitly classified examples
   high_confidence = [ex for ex in examples if ex['shunt_type'] != 'Unknown']

   # Use only examples with ligation strategy
   with_strategy = [ex for ex in examples if ex['ligation_strategy']]

4. Split into train/val/test:
   import random
   random.shuffle(examples)

   train_size = int(0.7 * len(examples))
   val_size = int(0.15 * len(examples))

   train = examples[:train_size]
   val = examples[train_size:train_size+val_size]
   test = examples[train_size+val_size:]

FOR DATA ANALYSIS:

1. Analyze distribution:
   from collections import Counter
   types = [ex['shunt_type'] for ex in examples]
   distribution = Counter(types)

2. Export to other formats:
   - CSV provided for spreadsheet viewing
   - Can convert to other formats as needed

3. Cross-reference with source:
   Use page_or_section field to find examples in original PDFs
   for manual verification or deeper study

================================================================================
UNDERSTANDING "UNKNOWN" EXAMPLES
================================================================================

WHY ARE 809 EXAMPLES MARKED "UNKNOWN"?

The extraction system looks for explicit Type classifications in text.
Many clinical examples describe flow patterns without stating "Type X":
- "EP N1→N2 with RP N2→N1" (describes Type 1 pattern but doesn't say "Type 1")
- "GSV fed by perforator with N3 reflux" (describes Type 2B pattern)
- "N2→N3 flow to refluxing tributary" (describes Type 2A pattern)

These UNKNOWN examples still contain:
✓ Real clinical hemodynamic information
✓ Flow pattern descriptions (N1, N2, N3, EP, RP)
✓ Treatment implications
✓ Valuable context for decision-making

They are suitable for training because they show how clinicians reason about
venous problems, even if the formal classification isn't explicit.

RECOMMENDATION:
Use all 1,214 examples for training, or if you want highest confidence:
- Use 405 examples with explicit Type classifications (Unknown removed)
- Use 206 refined high-confidence examples for critical models

================================================================================
REFERENCE: CHIVA CLASSIFICATION RULES
================================================================================

See included chiva_rules.txt for detailed classification rules:

Quick Summary:
- Type 1: EP N1→N2 (SFJ entry) + RP N2→N1, no EP N2→N3
- Type 2A: EP N2→N3 (perforator entry), no EP N1→N2
- Type 2B: EP N2→N2 (perforator) + RP N3 only
- Type 2C: EP N2→N2 (perforator) + RP N3 + RP N2→N1
- Type 3: EP N1→N2 + EP N2→N3 + RP N3 only
- Type 4: Supra-inguinal reflux sources
- Type 5: Supra-inguinal reflux with extensive network
- Type 1+2: Combined Type 1 and Type 2 patterns
- No shunt: No reflux present

N1 = Deep venous system
N2 = Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV)
N3 = Tributaries/superficial branches
EP = Physiological (antegrade) flow
RP = Retrograde (pathological) flow / Reflux

================================================================================
INACCESSIBLE DOCUMENTS
================================================================================

Two source PDFs could not be extracted:

1. Task_1_Shunt_Classification_Knowledgebase.pdf (22.5 MB)
   - Status: No text extraction possible
   - Likely cause: Image-based PDF or PDF corruption
   - Would require: Tesseract OCR (needs poppler library)

2. Ligation_Knowledgebase_2.pdf (3.2 MB)
   - Status: No text extraction possible
   - Likely cause: Encrypted or corrupted
   - Would require: OCR or direct PDF manipulation tools

If you need data from these PDFs, consider:
- Installing poppler and Tesseract for OCR processing
- Using specialized medical PDF readers
- Manual extraction of critical sections

================================================================================
VALIDATION & QUALITY ASSURANCE
================================================================================

Data Quality Checks:
✓ All 1,214 examples have required fields
✓ All examples extracted directly from source PDFs (verified by page reference)
✓ No synthetic or invented examples
✓ JSON validation: All files parse correctly
✓ UTF-8 encoding: All unicode characters preserved
✓ Deduplication: Near-duplicate examples removed

Content Characteristics:
✓ Average instruction length: ~380 characters (meaningful content)
✓ Average output length: ~800 characters (detailed text)
✓ Medical terminology preserved (N1, N2, N3, EP, RP, SFJ, GSV, SSV, etc.)
✓ Flow pattern descriptions present in ~66% of examples
✓ Ligation strategy information in ~47% of examples

If you identify issues or have suggestions for improvement:
1. Check page references in source PDFs
2. Review original context in source document
3. Cross-validate against CHIVA classification rules

================================================================================
FILE LOCATIONS
================================================================================

All outputs created in: C:\Users\Krish\Downloads\LLM_Finetuning\

Primary file:
  CHIVA_Training_Dataset_Final.json

Supporting files:
  CHIVA_Training_Dataset_Final.csv
  chiva_clinical_examples.json
  chiva_refined_clinical_cases.json
  EXTRACTION_SUMMARY.txt (detailed report)
  README_CHIVA_EXTRACTION.txt (this file)

Source PDFs:
  Domain_Specific_Data/0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf
  Domain_Specific_Data/Ligation_Knowledgebase_1.pdf
  Domain_Specific_Data/Shunt_Book_8.pdf
  Domain_Specific_Data/Shunt_Classification_Cheetsheat.pdf
  Domain_Specific_Data/Task_1_Shunt_Classification_Knowledgebase.pdf (inaccessible)
  Domain_Specific_Data/Ligation_Knowledgebase_2.pdf (inaccessible)
  Domain_Specific_Data/chiva_rules.txt (reference document)

Python scripts (for reference):
  extract_chiva_comprehensive.py (main extraction script)
  extract_refined_cases.py (high-confidence case extraction)
  finalize_dataset.py (consolidation and deduplication)
  verify_dataset.py (validation script)

================================================================================
CONTACT & NOTES
================================================================================

Project Owner: Krish (claudekumar07@gmail.com)
Extraction Date: 2026-05-06
Total Extraction Time: ~30-45 minutes
Tools Used: Python 3.14, PyMuPDF, pdfplumber, regex

Purpose: Train Mistral-7B model on CHIVA venous disease classification
Hardware: RTX 5090 32GB VRAM available for fine-tuning

Next Steps:
1. Review extracted data quality
2. Filter by shunt type or confidence level as needed
3. Format for your fine-tuning framework (HuggingFace, Ollama, etc.)
4. Train and validate model
5. Test on validation set before production use

================================================================================
END OF GUIDE
================================================================================
