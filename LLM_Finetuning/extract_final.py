#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

try:
    import pdfplumber
except ImportError:
    import subprocess
    print("Installing pdfplumber...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

pdf_files = [
    "Domain_Specific_Data/Task_1_Shunt_Classification_Knowledgebase.pdf",
    "Domain_Specific_Data/Ligation_Knowledgebase_2.pdf"
]

extracted_examples = []

for pdf_path in pdf_files:
    print(f"\nProcessing: {Path(pdf_path).name}")

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
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

                shunt_types = []
                for match in re.finditer(r'Type\s+([1-6A-C+\s]+)', text, re.IGNORECASE):
                    shunt_types.append(match.group(1).strip())

                example = {
                    "instruction": f"Analyze CHIVA venous shunt: {text[:200]}",
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

            print(f"   Extracted {page_count} pages with content")

    except Exception as e:
        print(f"   Error: {str(e)}")

print(f"\nExtracted from image PDFs: {len(extracted_examples)} examples")

# Load existing
try:
    with open('CHIVA_Training_Dataset_Final.json', 'r', encoding='utf-8') as f:
        existing = json.load(f)
    print(f"Existing examples: {len(existing)}")
except:
    existing = []

# Merge
all_examples = existing + extracted_examples
print(f"Total before dedup: {len(all_examples)}")

# Deduplicate
seen = set()
unique_examples = []
for ex in all_examples:
    key = (ex.get('instruction', '')[:100], ex.get('shunt_type', ''))
    if key not in seen:
        seen.add(key)
        unique_examples.append(ex)

print(f"After deduplication: {len(unique_examples)}")

# Split
import random
random.shuffle(unique_examples)
split_idx = int(len(unique_examples) * 0.9)
train_data = unique_examples[:split_idx]
val_data = unique_examples[split_idx:]

print(f"\nTraining examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")

# Save JSONL
with open('training_datasets/training_data.jsonl', 'w', encoding='utf-8') as f:
    for ex in train_data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

with open('training_datasets/validation_data.jsonl', 'w', encoding='utf-8') as f:
    for ex in val_data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"\nSaved training data to: training_datasets/training_data.jsonl")
print(f"Saved validation data to: training_datasets/validation_data.jsonl")

# Save merged JSON
with open('CHIVA_FINAL_2000_DATASET.json', 'w', encoding='utf-8') as f:
    json.dump(unique_examples, f, ensure_ascii=False, indent=2)

print(f"Saved complete dataset to: CHIVA_FINAL_2000_DATASET.json")
print(f"\nREADY FOR QWEN3-4B FINE-TUNING!")
print(f"Total examples: {len(unique_examples)}")
