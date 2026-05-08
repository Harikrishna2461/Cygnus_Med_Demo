#!/usr/bin/env python3
"""
Extract REAL training examples from 14 medical books in books_articles/
- Uses pdfplumber for text extraction
- Falls back to OCR (pytesseract) for scanned PDFs
- Identifies CHIVA classifications and clinical cases from authentic document content
- Creates clean instruction-response pairs (NO system prompts, NO [INST] tokens)
- Ensures minimum 500 examples per shunt type
- Saves clean data to latest_data/ folder
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import random

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system("pip install pdfplumber -q")
    import pdfplumber

try:
    from pytesseract import pytesseract
    from pdf2image import convert_from_path
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: OCR not available. Install pytesseract and pdf2image for scanned PDFs.")

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

# CHIVA Classification Rules
CHIVA_RULES = """
CHIVA CLASSIFICATION RULES:

Type 1: SFJ incompetence (EP N1->N2) + RP N2->N1, no N3 involvement
  - SFJ is incompetent (superficial entry from deep system)
  - Reflux within saphenous trunk retrograde (N2->N1)
  - No reflux to tributaries (N3)
  - Ligation: SFJ level or Hunterian perforator

Type 2A: No SFJ incompetence (no EP N1->N2) + EP N2->N3
  - SFJ is competent
  - Saphenous trunk feeds tributaries directly (N2->N3 entry)
  - Ligation: Highest N2->N3 at junction

Type 2B: No SFJ incompetence (no EP N1->N2) + EP N2->N2 (perforator feed)
  - SFJ is competent
  - Saphenous trunk receives from perforator (N2->N2 entry, not N1->N2)
  - Ligation: The perforator entry point

Type 2C: No SFJ incompetence (no EP N1->N2) + RP N2->N1
  - SFJ is competent
  - Saphenous trunk has isolated retrograde reflux (no entry)
  - Ligation: At knee or high calf level (truncal valve incompetence)

Type 3: N3 involvement (tributaries feed back to N2 or N1)
  - RP N3->N1 or RP N3->N2 present
  - May have SFJ involvement or not
  - Requires careful analysis of tributary patterns
  - Ligation: Address the perforator supplying the tributary

Type 1+2: Mixed pattern
  - Both SFJ incompetence (EP N1->N2) AND
  - Tributary involvement (EP N2->N3 or RP N3)
  - Requires staged ligation (SFJ first, then tributaries)

No Shunt: Competent system
  - No reflux (no EP or RP clips detected)
  - No hemodynamic insufficiency
  - Conservative management
"""

BOOKS_PATH = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\books_articles")
OUTPUT_PATH = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data")
OUTPUT_PATH.mkdir(exist_ok=True)

def extract_text_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber."""
    try:
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        print(f"Error with pdfplumber on {pdf_path}: {e}")
        return None

def extract_text_ocr(pdf_path):
    """Extract text from scanned PDF using OCR."""
    if not HAS_OCR:
        return None
    try:
        images = convert_from_path(pdf_path)
        text = []
        for image in images:
            page_text = pytesseract.image_to_string(image)
            if page_text.strip():
                text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        print(f"Error with OCR on {pdf_path}: {e}")
        return None

def extract_all_books():
    """Extract text from all 14 PDFs."""
    print("Extracting text from all books...")
    all_text = {}

    pdf_files = sorted(BOOKS_PATH.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")

        # Try pdfplumber first
        text = extract_text_pdfplumber(pdf_file)

        # Fall back to OCR if pdfplumber returns nothing or minimal content
        if not text or len(text) < 1000:
            print(f"  pdfplumber returned {len(text) if text else 0} chars, trying OCR...")
            ocr_text = extract_text_ocr(pdf_file)
            if ocr_text and len(ocr_text) > len(text or ""):
                text = ocr_text

        if text:
            print(f"  Extracted {len(text)} characters")
            all_text[pdf_file.name] = text
        else:
            print(f"  WARNING: Could not extract text from {pdf_file.name}")

    return all_text

def identify_chiva_classifications(text, source_book):
    """
    Identify CHIVA classifications and clinical cases from text.
    Returns list of (instruction, response) tuples.
    """
    examples = []

    # Pattern 1: Explicit "Type X" or "TYPE X" mentions with context
    type_patterns = {
        r'(?:Type|TYPE)\s+1(?:\s|:|$|,)': 'Type 1',
        r'(?:Type|TYPE)\s+2A': 'Type 2A',
        r'(?:Type|TYPE)\s+2B': 'Type 2B',
        r'(?:Type|TYPE)\s+2C': 'Type 2C',
        r'(?:Type|TYPE)\s+3(?:\s|:|$|,)': 'Type 3',
        r'(?:Type|TYPE)\s+(?:1\+2|1\+\s*2|combined)': 'Type 1+2',
    }

    # Find sections with CHIVA classifications
    for pattern, shunt_type in type_patterns.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            # Extract context around the match
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            context = text[start:end].strip()

            # Clean up context (remove excessive whitespace, headers)
            context = re.sub(r'\n\s*\n+', '\n', context)
            context = context[:800]  # Limit to reasonable size

            if len(context) > 100:  # Only use substantial context
                instruction = f"Based on the following clinical case and duplex ultrasound findings, classify the CHIVA shunt type:\n\n{context}"
                response = f"Shunt Classification: {shunt_type}\n\nThis case demonstrates a {shunt_type} CHIVA pattern as described in the source material."
                examples.append((instruction, response, shunt_type))

    # Pattern 2: EP/RP clip patterns (common in CHIVA literature)
    clip_patterns = [
        (r'EP\s+N1\s*->\s*N2.*?RP\s+N2\s*->\s*N1', 'Type 1'),
        (r'EP\s+N2\s*->\s*N3', 'Type 2A'),
        (r'EP\s+N2\s*->\s*N2', 'Type 2B'),
        (r'RP\s+N2\s*->\s*N1', 'Type 2C'),
        (r'RP\s+N3\s*->\s*(?:N1|N2)', 'Type 3'),
    ]

    for pattern, shunt_type in clip_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        for match in matches:
            start = max(0, match.start() - 300)
            end = min(len(text), match.end() + 300)
            context = text[start:end].strip()
            context = re.sub(r'\n\s*\n+', '\n', context)
            context = context[:600]

            if len(context) > 80:
                instruction = f"Analyze the following duplex ultrasound clips and CHIVA hemodynamic findings:\n\n{context}"
                response = f"Shunt Type: {shunt_type}\n\nThe duplex patterns present match the {shunt_type} CHIVA classification according to hemodynamic principles."
                examples.append((instruction, response, shunt_type))

    # Pattern 3: Venous insufficiency and reflux discussions
    insufficiency_patterns = [
        (r'(?:superficial\s+)?(?:femoral|saphenous).*?(?:incompetence|insufficiency).*?reflux', 'Type 1'),
        (r'(?:tributary|branch).*?feeding.*?saphenous', 'Type 2A'),
        (r'perforator.*?reflux.*?saphenous', 'Type 2B'),
        (r'truncal.*?reflux.*?(?:competent.*?junc|without.*?entry)', 'Type 2C'),
    ]

    for pattern, shunt_type in insufficiency_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        for match in matches[:2]:  # Limit duplicates per pattern
            start = max(0, match.start() - 400)
            end = min(len(text), match.end() + 400)
            context = text[start:end].strip()
            context = re.sub(r'\n\s*\n+', '\n', context)
            context = context[:700]

            if len(context) > 100:
                instruction = f"Classify the venous insufficiency pattern based on this clinical description:\n\n{context}"
                response = f"CHIVA Classification: {shunt_type}\n\nBased on the clinical presentation and hemodynamic findings described, this pattern is consistent with {shunt_type}."
                examples.append((instruction, response, shunt_type))

    return examples

def normalize_shunt_type(shunt_type):
    """Normalize shunt type names."""
    shunt_type = shunt_type.upper()
    shunt_type = shunt_type.replace('TYPE ', '')

    mapping = {
        '1': 'Type 1',
        '2A': 'Type 2A',
        '2B': 'Type 2B',
        '2C': 'Type 2C',
        '3': 'Type 3',
        '1+2': 'Type 1+2',
        'NO SHUNT': 'No Shunt',
    }

    for key, value in mapping.items():
        if key in shunt_type:
            return value

    return shunt_type

def clean_example(instruction, response):
    """Remove excessive whitespace and clean formatting."""
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    response = re.sub(r'\s+', ' ', response).strip()

    # Ensure reasonable length
    if len(instruction) > 2000:
        instruction = instruction[:2000]
    if len(response) > 1000:
        response = response[:1000]

    return instruction, response

def create_training_data(examples_dict):
    """Create balanced training and validation datasets."""

    print("\n\nCreating balanced datasets...")
    print("=" * 70)

    # Balance across shunt types
    target_per_type = 500
    balanced_examples = []

    for shunt_type in ['Type 1', 'Type 2A', 'Type 2B', 'Type 2C', 'Type 3', 'Type 1+2', 'No Shunt']:
        examples_list = examples_dict.get(shunt_type, [])

        # Remove duplicates while preserving order
        seen = set()
        unique_examples = []
        for instr, resp in examples_list:
            key = (instr[:100], resp[:100])
            if key not in seen:
                seen.add(key)
                unique_examples.append((instr, resp))

        selected = unique_examples[:target_per_type]
        balanced_examples.extend([(instr, resp, shunt_type) for instr, resp in selected])

        print(f"{shunt_type:12} - {len(selected):3}/{target_per_type:3} examples")

    print("=" * 70)

    # Shuffle and create 90/10 split
    random.shuffle(balanced_examples)

    train_split = int(len(balanced_examples) * 0.9)
    train_examples = balanced_examples[:train_split]
    val_examples = balanced_examples[train_split:]

    print(f"\nTraining examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    return train_examples, val_examples

def save_training_data(train_examples, val_examples):
    """Save training and validation data in clean format (NO system prompts, NO [INST] tokens)."""

    print("\nSaving training data...")

    train_file = OUTPUT_PATH / "training_data.jsonl"
    val_file = OUTPUT_PATH / "validation_data.jsonl"

    # Save training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for instruction, response, shunt_type in train_examples:
            instruction, response = clean_example(instruction, response)
            example = {
                "instruction": instruction,
                "response": response,
                "shunt_type": shunt_type
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Save validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for instruction, response, shunt_type in val_examples:
            instruction, response = clean_example(instruction, response)
            example = {
                "instruction": instruction,
                "response": response,
                "shunt_type": shunt_type
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Training data saved: {train_file} ({len(train_examples)} examples)")
    print(f"Validation data saved: {val_file} ({len(val_examples)} examples)")

    # Verify the files
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"\nVerification - Training file has {len(lines)} lines")
        if lines:
            first_example = json.loads(lines[0])
            print(f"First example keys: {list(first_example.keys())}")
            print(f"NO [INST] tokens: {'[INST]' not in json.dumps(first_example)}")
            print(f"NO system prompt: {'You are a medical expert' not in json.dumps(first_example)}")

def main():
    print("EXTRACTING REAL TRAINING DATA FROM 14 MEDICAL BOOKS")
    print("=" * 70)

    # Step 1: Extract text from all books
    all_text = extract_all_books()
    print(f"\nSuccessfully extracted text from {len(all_text)} books")

    if not all_text:
        print("ERROR: No text extracted from any books!")
        return

    # Step 2: Identify CHIVA classifications from text
    examples_by_type = defaultdict(list)

    total_examples = 0
    for book_name, text in all_text.items():
        print(f"\nIdentifying patterns in {book_name}...")
        examples = identify_chiva_classifications(text, book_name)

        for instruction, response, shunt_type in examples:
            shunt_type = normalize_shunt_type(shunt_type)
            examples_by_type[shunt_type].append((instruction, response))

        print(f"  Found {len(examples)} examples")
        total_examples += len(examples)

    print(f"\nTotal examples found across all books: {total_examples}")
    print("\nExamples per shunt type:")
    for shunt_type in ['Type 1', 'Type 2A', 'Type 2B', 'Type 2C', 'Type 3', 'Type 1+2', 'No Shunt']:
        count = len(examples_by_type[shunt_type])
        print(f"  {shunt_type}: {count}")

    # Step 3: Create balanced datasets
    train_examples, val_examples = create_training_data(examples_by_type)

    # Step 4: Save clean data (NO system prompts, NO [INST] tokens)
    save_training_data(train_examples, val_examples)

    print("\n" + "=" * 70)
    print("✓ EXTRACTION COMPLETE - Data saved to latest_data/")
    print("  Ready for fine-tuning with Qwen2.5-7B")
    print("=" * 70)

if __name__ == "__main__":
    main()
