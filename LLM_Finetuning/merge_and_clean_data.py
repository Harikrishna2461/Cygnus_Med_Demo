#!/usr/bin/env python3
"""
Merge new extracted data (237 examples) with old training data (5544 + 640 examples).
Remove ALL system prompt prefixes and [INST] tokens.
Create flexible, clean training data for general domain knowledge absorption.
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
import random

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

OLD_TRAIN = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\training_data.jsonl")
OLD_VAL = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\validation_data.jsonl")
NEW_TRAIN = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\training_data.jsonl")
NEW_VAL = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\validation_data.jsonl")
OUTPUT_DIR = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data")

def clean_example(text):
    """Remove system prompts and [INST] tokens."""
    # Remove the system prompt prefix
    text = re.sub(
        r"You are a medical expert specialising in venous and lymphatic disorders.*?Answer questions accurately using clinical and scientific knowledge\.\s*\n\n",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove [INST] and </INST> tokens
    text = text.replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "")

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def parse_old_format(text):
    """
    Parse old format: "[INST] system prompt\n\ninstruction [/INST] response </s>"
    Extract instruction and response.
    """
    # Remove wrapper tokens
    text = text.replace("<s>", "").replace("</s>", "")

    # Try to split on [/INST]
    if "[/INST]" in text:
        parts = text.split("[/INST]")
        if len(parts) == 2:
            inst_part = parts[0].replace("[INST]", "").strip()
            resp_part = parts[1].strip()

            # Remove system prompt from instruction
            inst_part = re.sub(
                r"You are a medical expert specialising in venous and lymphatic disorders.*?Answer questions accurately using clinical and scientific knowledge\.\s*\n\n",
                "",
                inst_part,
                flags=re.DOTALL
            )

            inst_part = inst_part.strip()

            if inst_part and resp_part:
                return inst_part, resp_part

    return None, None

def load_old_data():
    """Load and clean old training and validation data."""
    print("Loading old training data...")
    all_examples = []
    skipped = 0

    # Load old training data
    if OLD_TRAIN.exists():
        with open(OLD_TRAIN, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')

                    instruction, response = parse_old_format(text)
                    if instruction and response:
                        all_examples.append({
                            'instruction': instruction,
                            'response': response,
                            'source': 'old_training'
                        })
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1

    print(f"  Loaded {len(all_examples)} training examples (skipped {skipped})")

    # Load old validation data
    print("Loading old validation data...")
    if OLD_VAL.exists():
        with open(OLD_VAL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')

                    instruction, response = parse_old_format(text)
                    if instruction and response:
                        all_examples.append({
                            'instruction': instruction,
                            'response': response,
                            'source': 'old_validation'
                        })
                except json.JSONDecodeError:
                    pass

    return all_examples

def load_new_data():
    """Load new extracted data (already clean)."""
    print("Loading new extracted data...")
    all_examples = []

    # Load new training
    if NEW_TRAIN.exists():
        with open(NEW_TRAIN, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    all_examples.append({
                        'instruction': obj.get('instruction', ''),
                        'response': obj.get('response', ''),
                        'shunt_type': obj.get('shunt_type', 'unknown'),
                        'source': 'new_training'
                    })
                except json.JSONDecodeError:
                    pass

    # Load new validation
    if NEW_VAL.exists():
        with open(NEW_VAL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    all_examples.append({
                        'instruction': obj.get('instruction', ''),
                        'response': obj.get('response', ''),
                        'shunt_type': obj.get('shunt_type', 'unknown'),
                        'source': 'new_validation'
                    })
                except json.JSONDecodeError:
                    pass

    print(f"  Loaded {len(all_examples)} extracted examples")
    return all_examples

def deduplicate_examples(examples):
    """Remove duplicate examples based on instruction+response signature."""
    print("Deduplicating examples...")
    seen = {}
    unique = []
    duplicates = 0

    for ex in examples:
        instr = ex['instruction'][:100].lower()
        resp = ex['response'][:100].lower()
        sig = f"{instr}|{resp}"

        if sig not in seen:
            seen[sig] = True
            unique.append(ex)
        else:
            duplicates += 1

    print(f"  Found and removed {duplicates} duplicates")
    print(f"  Unique examples: {len(unique)}")
    return unique

def validate_examples(examples):
    """Remove invalid examples (empty, too short, still have system prompts)."""
    print("Validating examples...")
    valid = []
    invalid = 0

    for ex in examples:
        instr = ex['instruction'].strip()
        resp = ex['response'].strip()

        # Check for minimum length
        if len(instr) < 20 or len(resp) < 20:
            invalid += 1
            continue

        # Check for system prompt remnants
        if "You are a medical expert" in instr or "You are a medical expert" in resp:
            invalid += 1
            continue

        # Check for [INST] remnants
        if "[INST]" in instr or "[INST]" in resp or "[/INST]" in instr or "[/INST]" in resp:
            invalid += 1
            continue

        valid.append(ex)

    print(f"  Removed {invalid} invalid examples")
    print(f"  Valid examples: {len(valid)}")
    return valid

def create_final_split(examples):
    """Create 90/10 train/val split."""
    print("Creating 90/10 train/val split...")

    random.shuffle(examples)

    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:]

    print(f"  Training: {len(train)} examples")
    print(f"  Validation: {len(val)} examples")
    print(f"  Total: {len(examples)} examples")

    return train, val

def save_final_data(train, val):
    """Save merged, cleaned data to latest_data/"""
    print("\nSaving final merged data...")

    train_file = OUTPUT_DIR / "training_data.jsonl"
    val_file = OUTPUT_DIR / "validation_data.jsonl"

    # Save training
    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train:
            obj = {
                'instruction': ex['instruction'],
                'response': ex['response']
            }
            if 'shunt_type' in ex:
                obj['shunt_type'] = ex['shunt_type']

            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # Save validation
    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val:
            obj = {
                'instruction': ex['instruction'],
                'response': ex['response']
            }
            if 'shunt_type' in ex:
                obj['shunt_type'] = ex['shunt_type']

            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"✓ Training: {train_file} ({len(train)} examples)")
    print(f"✓ Validation: {val_file} ({len(val)} examples)")

    # Verify clean format
    print("\nVerification:")
    with open(train_file, 'r', encoding='utf-8') as f:
        first_line = json.loads(f.readline())

    has_inst = '[INST]' in json.dumps(first_line)
    has_prompt = 'You are a medical expert' in json.dumps(first_line)

    print(f"  ✓ NO [INST] tokens: {not has_inst}")
    print(f"  ✓ NO system prompt: {not has_prompt}")
    print(f"  ✓ Keys: {list(first_line.keys())}")

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "MERGING OLD AND NEW DATA - REMOVING ALL PREFIXES".center(68) + "║")
    print("╚" + "=" * 68 + "╝\n")

    # Load both old and new data
    old_examples = load_old_data()
    new_examples = load_new_data()

    print(f"\nTotal loaded: {len(old_examples) + len(new_examples)} examples")
    print(f"  Old data: {len(old_examples)}")
    print(f"  New data: {len(new_examples)}")

    # Combine
    all_examples = old_examples + new_examples

    # Deduplicate
    unique_examples = deduplicate_examples(all_examples)

    # Validate (remove invalid, system prompts, etc.)
    valid_examples = validate_examples(unique_examples)

    # Create final split
    train, val = create_final_split(valid_examples)

    # Save
    save_final_data(train, val)

    print("\n" + "=" * 70)
    print("✓ MERGE COMPLETE")
    print("=" * 70)
    print(f"\nCombined dataset ready for flexible domain knowledge training")
    print(f"Files saved in: latest_data/")
    print("\nTraining notes:")
    print("  • Mix of CHIVA classifications, venous anatomy, and clinical reasoning")
    print("  • No system prompts - model learns from examples directly")
    print("  • Ready for fine-tuning with Qwen2.5-7B (2-3 epochs recommended)")

if __name__ == "__main__":
    main()
