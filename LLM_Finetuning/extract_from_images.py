#!/usr/bin/env python3
"""
Extract text from image-based PDFs using pdfplumber
Then parse into CHIVA training examples
"""

import json
import re
from pathlib import Path

# First, install pdfplumber if not present
try:
    import pdfplumber
except ImportError:
    import subprocess
    import sys
    print("Installing pdfplumber...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

pdf_files = [
    "Domain_Specific_Data/Task_1_Shunt_Classification_Knowledgebase.pdf",
    "Domain_Specific_Data/Ligation_Knowledgebase_2.pdf"
]

extracted_examples = []

for pdf_path in pdf_files:
    print(f"\n{'='*80}")
    print(f"Processing: {Path(pdf_path).name}")
    print(f"{'='*80}")

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        continue

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   Total pages: {len(pdf.pages)}")
            page_count = 0

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()

                if not text or len(text.strip()) < 50:
                    continue

                page_count += 1

                # Extract any shunt type mentions
                shunt_types = []
                for match in re.finditer(r'Type\s+([1-6A-C+\s]+)', text, re.IGNORECASE):
                    shunt_types.append(match.group(1).strip())

                # Create example from page content
                example = {
                    "instruction": f"Analyze CHIVA venous shunt: {text[:200]}...",
                    "input": "",
                    "output": text[:500],
                    "source_document": Path(pdf_path).name,
                    "page_number": page_num,
                    "shunt_type": shunt_types[0] if shunt_types else "Unknown",
                    "type": "classification" if "classify" in text.lower() else "ligation",
                    "difficulty": "intermediate",
                    "source": "extracted_from_books"
                }

                extracted_examples.append(example)

            print(f"   ✓ Extracted {page_count} pages with content")

    except Exception as e:
        print(f"   ❌ Error processing file: {e}")
        continue

print(f"\n{'='*80}")
print(f"EXTRACTION SUMMARY")
print(f"{'='*80}")
print(f"Total examples from image PDFs: {len(extracted_examples)}")

# Load existing examples
try:
    with open('CHIVA_Training_Dataset_Final.json', 'r') as f:
        existing_examples = json.load(f)
    print(f"Existing examples: {len(existing_examples)}")
except:
    existing_examples = []
    print("No existing dataset found")

# Merge all examples
all_examples = existing_examples + extracted_examples
print(f"Total merged examples: {len(all_examples)}")

# Remove duplicates based on content hash
seen = set()
unique_examples = []
for ex in all_examples:
    key = (ex.get('instruction', '')[:100], ex.get('shunt_type', ''))
    if key not in seen:
        seen.add(key)
        unique_examples.append(ex)

print(f"After deduplication: {len(unique_examples)}")

# Split into train/val
import random
random.shuffle(unique_examples)

split_idx = int(len(unique_examples) * 0.9)
train_data = unique_examples[:split_idx]
val_data = unique_examples[split_idx:]

print(f"\nTraining examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")

# Save as JSONL for training
with open('chiva_training_data_2000.jsonl', 'w', encoding='utf-8') as f:
    for ex in train_data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

with open('chiva_validation_data_200.jsonl', 'w', encoding='utf-8') as f:
    for ex in val_data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"\nSaved training data to: chiva_training_data_2000.jsonl")
print(f"Saved validation data to: chiva_validation_data_200.jsonl")

# Also save as merged JSON
with open('CHIVA_FINAL_MERGED_DATASET.json', 'w') as f:
    json.dump({
        "total_examples": len(unique_examples),
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "data": unique_examples
    }, f, indent=2)

print(f"Saved merged dataset to: CHIVA_FINAL_MERGED_DATASET.json")
print(f"\n{'='*80}")
print(f"EXTRACTION COMPLETE - Ready for Qwen3-4B fine-tuning!")
print(f"{'='*80}")
