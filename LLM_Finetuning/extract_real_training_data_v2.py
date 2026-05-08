#!/usr/bin/env python3
"""
Enhanced extraction of REAL training examples from 14 medical books.
Focuses on longer clinical narratives and case descriptions.
Creates clean instruction-response pairs (NO system prompts, NO [INST] tokens)
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
    os.system("pip install pdfplumber -q")
    import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

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
        print(f"Error extracting {pdf_path.name}: {e}")
        return None

def extract_all_books():
    """Extract text from all PDFs."""
    print("=" * 70)
    print("STEP 1: Extracting text from 14 books...")
    print("=" * 70)

    all_text = {}
    pdf_files = sorted(BOOKS_PATH.glob("*.pdf"))

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i:2}/{len(pdf_files)}] {pdf_file.name[:60]:60}", end=" ... ")
        text = extract_text_pdfplumber(pdf_file)

        if text:
            print(f"{len(text):>10} chars")
            all_text[pdf_file.name] = text
        else:
            print("SKIPPED")

    return all_text

def split_into_paragraphs(text):
    """Split text into meaningful paragraphs."""
    # Split on multiple newlines
    paragraphs = re.split(r'\n\s*\n+', text)
    # Clean up
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    return paragraphs

def extract_clinical_cases(all_text):
    """
    Extract clinical cases and relevant passages from books.
    Focus on sections discussing CHIVA classifications, hemodynamics, and case descriptions.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Extracting clinical cases and CHIVA discussions...")
    print("=" * 70)

    all_examples = defaultdict(list)
    total_paragraphs = 0

    for book_name, text in all_text.items():
        print(f"\nProcessing: {book_name[:50]:50}", end=" ")

        paragraphs = split_into_paragraphs(text)
        total_paragraphs += len(paragraphs)

        # Keywords indicating different CHIVA types
        type1_keywords = ['type 1', 'SFJ incompetence', 'superficial entry', 'EP N1->N2', 'RP N2->N1']
        type2a_keywords = ['type 2a', 'tributary feed', 'EP N2->N3', 'branch feed', 'no SFJ']
        type2b_keywords = ['type 2b', 'perforator feed', 'EP N2->N2']
        type2c_keywords = ['type 2c', 'truncal reflux', 'saphenous reflux', 'isolated reflux']
        type3_keywords = ['type 3', 'tributary reflux', 'RP N3', 'branch reflux']
        noshunt_keywords = ['competent', 'normal', 'no reflux', 'no shunt', 'hemodynamically normal']

        type1_examples = []
        type2a_examples = []
        type2b_examples = []
        type2c_examples = []
        type3_examples = []
        noshunt_examples = []

        for para in paragraphs:
            para_lower = para.lower()

            # Type 1 - SFJ incompetence
            if any(kw in para_lower for kw in type1_keywords):
                type1_examples.append(para)

            # Type 2A - Tributary feed without SFJ
            if any(kw in para_lower for kw in type2a_keywords):
                type2a_examples.append(para)

            # Type 2B - Perforator feed
            if any(kw in para_lower for kw in type2b_keywords):
                type2b_examples.append(para)

            # Type 2C - Truncal reflux
            if any(kw in para_lower for kw in type2c_keywords):
                type2c_examples.append(para)

            # Type 3 - Tributary reflux
            if any(kw in para_lower for kw in type3_keywords):
                type3_examples.append(para)

            # No Shunt - Normal/competent systems
            if any(kw in para_lower for kw in noshunt_keywords) and 'reflux' not in para_lower:
                noshunt_examples.append(para)

        # Add to global examples
        all_examples['Type 1'].extend(type1_examples[:100])  # Limit per book
        all_examples['Type 2A'].extend(type2a_examples[:100])
        all_examples['Type 2B'].extend(type2b_examples[:100])
        all_examples['Type 2C'].extend(type2c_examples[:100])
        all_examples['Type 3'].extend(type3_examples[:100])
        all_examples['No Shunt'].extend(noshunt_examples[:100])

        total = len(type1_examples) + len(type2a_examples) + len(type2b_examples) + \
                len(type2c_examples) + len(type3_examples) + len(noshunt_examples)
        print(f"extracted {total} relevant passages")

    print(f"\nTotal paragraphs analyzed: {total_paragraphs:,}")
    print("\nExamples by type before deduplication:")
    for shunt_type in ['Type 1', 'Type 2A', 'Type 2B', 'Type 2C', 'Type 3', 'No Shunt']:
        count = len(all_examples[shunt_type])
        print(f"  {shunt_type:12} - {count:5} examples")

    return all_examples

def deduplicate(examples):
    """Remove duplicate examples while preserving order."""
    seen = set()
    unique = []
    for ex in examples:
        # Use first 100 chars as signature
        sig = ex[:100].lower()
        if sig not in seen:
            seen.add(sig)
            unique.append(ex)
    return unique

def create_training_pairs(all_examples):
    """
    Convert raw text passages into instruction-response pairs.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Creating instruction-response pairs...")
    print("=" * 70)

    training_pairs = defaultdict(list)

    for shunt_type, paragraphs in all_examples.items():
        # Deduplicate
        unique_paragraphs = deduplicate(paragraphs)

        for para in unique_paragraphs:
            # Clean the paragraph
            para = re.sub(r'\s+', ' ', para).strip()

            if len(para) < 100 or len(para) > 2000:
                continue

            # Create instruction-response pair
            instruction = (
                f"Based on the following clinical findings and case description, "
                f"identify the CHIVA shunt classification and explain the hemodynamic reasoning:\n\n{para}"
            )

            if shunt_type == 'No Shunt':
                response = (
                    f"CHIVA Classification: {shunt_type}\n\n"
                    f"Analysis: This case demonstrates a hemodynamically normal venous system with "
                    f"competent valves and no evidence of reflux. No CHIVA ligation is required. "
                    f"Conservative management is appropriate."
                )
            elif shunt_type == 'Type 1+2':
                response = (
                    f"CHIVA Classification: {shunt_type}\n\n"
                    f"Analysis: This case shows a mixed pattern with both SFJ incompetence and "
                    f"tributary involvement. A two-stage CHIVA procedure is typically recommended, "
                    f"with SFJ ligation performed first, followed by tributary management."
                )
            else:
                response = (
                    f"CHIVA Classification: {shunt_type}\n\n"
                    f"Analysis: This case demonstrates the characteristic hemodynamic pattern of {shunt_type}. "
                    f"The clinical findings and ultrasound clips match the CHIVA classification criteria for this type. "
                    f"Appropriate ligation strategy should be selected based on the specific anatomical entry points identified."
                )

            training_pairs[shunt_type].append((instruction, response))

    print("\nTraining pairs created per type:")
    for shunt_type in ['Type 1', 'Type 2A', 'Type 2B', 'Type 2C', 'Type 3', 'Type 1+2', 'No Shunt']:
        count = len(training_pairs[shunt_type])
        print(f"  {shunt_type:12} - {count:5} pairs")

    return training_pairs

def balance_and_split(training_pairs):
    """Balance dataset and create 90/10 train/val split."""
    print("\n" + "=" * 70)
    print("STEP 4: Balancing and splitting data...")
    print("=" * 70)

    # Target: try to get at least 100 per type (can't hit 500 with current extraction)
    target_per_type = 100
    all_train = []
    all_val = []

    print("\nBalanced distribution:")
    for shunt_type in ['Type 1', 'Type 2A', 'Type 2B', 'Type 2C', 'Type 3', 'Type 1+2', 'No Shunt']:
        pairs = training_pairs.get(shunt_type, [])

        # Limit to target
        selected = pairs[:target_per_type]

        # 90/10 split
        split_idx = int(len(selected) * 0.9)
        train = selected[:split_idx]
        val = selected[split_idx:]

        for instr, resp in train:
            all_train.append((instr, resp, shunt_type))
        for instr, resp in val:
            all_val.append((instr, resp, shunt_type))

        print(f"  {shunt_type:12} - train: {len(train):3}, val: {len(val):2}")

    print(f"\nTotal: {len(all_train)} training + {len(all_val)} validation")

    # Shuffle
    random.shuffle(all_train)
    random.shuffle(all_val)

    return all_train, all_val

def save_data(train_pairs, val_pairs):
    """Save to jsonl format (clean, no system prompts, no [INST])."""
    print("\n" + "=" * 70)
    print("STEP 5: Saving to latest_data/")
    print("=" * 70)

    train_file = OUTPUT_PATH / "training_data.jsonl"
    val_file = OUTPUT_PATH / "validation_data.jsonl"

    # Save training
    with open(train_file, 'w', encoding='utf-8') as f:
        for instruction, response, shunt_type in train_pairs:
            obj = {
                "instruction": instruction,
                "response": response,
                "shunt_type": shunt_type
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # Save validation
    with open(val_file, 'w', encoding='utf-8') as f:
        for instruction, response, shunt_type in val_pairs:
            obj = {
                "instruction": instruction,
                "response": response,
                "shunt_type": shunt_type
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"\n✓ Training data: {train_file}")
    print(f"  {len(train_pairs)} examples")
    print(f"\n✓ Validation data: {val_file}")
    print(f"  {len(val_pairs)} examples")

    # Verify clean format
    with open(train_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        first_obj = json.loads(first_line)

    print("\nVerification:")
    print(f"  ✓ Keys: {list(first_obj.keys())}")
    print(f"  ✓ NO [INST] tokens: {not '[INST]' in json.dumps(first_obj)}")
    print(f"  ✓ NO system prompt: {not 'You are a medical expert' in json.dumps(first_obj)}")

    return train_file, val_file

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "EXTRACTING REAL TRAINING DATA FROM 14 MEDICAL BOOKS".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    # Step 1: Extract all text
    all_text = extract_all_books()
    if not all_text:
        print("ERROR: No text extracted!")
        return

    # Step 2: Extract clinical cases
    all_examples = extract_clinical_cases(all_text)

    # Step 3: Create training pairs
    training_pairs = create_training_pairs(all_examples)

    # Step 4: Balance and split
    train_pairs, val_pairs = balance_and_split(training_pairs)

    # Step 5: Save
    train_file, val_file = save_data(train_pairs, val_pairs)

    print("\n" + "=" * 70)
    print("✓ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nReady for fine-tuning with Qwen2.5-7B")
    print("Files saved in: latest_data/")

if __name__ == "__main__":
    main()
