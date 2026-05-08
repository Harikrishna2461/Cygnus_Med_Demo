#!/usr/bin/env python3
"""
CORRECT MERGE: Keep training and validation separate
- Combine: old training (5544) + new training (211) = 5755 training
- Combine: old validation (617) + new validation (26) = 643 validation
- Remove duplicates from each set separately
- NO re-splitting
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

OLD_TRAIN = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\training_data.jsonl")
OLD_VAL = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\validation_data.jsonl")
NEW_TRAIN = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\training_data.jsonl")
NEW_VAL = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\validation_data.jsonl")
OUTPUT_DIR = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data")

def clean_system_prompt(text):
    """Remove system prompt prefix"""
    text = re.sub(
        r"You are a medical expert specialising in venous and lymphatic disorders.*?Answer questions accurately using clinical and scientific knowledge\.\s*\n\n",
        "",
        text,
        flags=re.DOTALL
    )
    text = text.replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "")
    return text.strip()

def parse_old_format(text):
    """Extract instruction and response from old [INST] format"""
    text = text.replace("<s>", "").replace("</s>", "")

    if "[/INST]" in text:
        parts = text.split("[/INST]")
        if len(parts) == 2:
            inst_part = parts[0].replace("[INST]", "").strip()
            resp_part = parts[1].strip()

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

def load_and_clean(file_path, source_name):
    """Load file and clean examples"""
    print(f"Loading {source_name}...", end=" ")
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)

                # Check if old format (has 'text' field) or new format (has 'instruction' field)
                if 'text' in obj:
                    instruction, response = parse_old_format(obj['text'])
                    if instruction and response:
                        examples.append({
                            'instruction': instruction,
                            'response': response
                        })
                elif 'instruction' in obj and 'response' in obj:
                    examples.append({
                        'instruction': obj['instruction'],
                        'response': obj['response']
                    })
            except:
                pass

    print(f"OK {len(examples)} examples")
    return examples

def deduplicate(examples):
    """Remove duplicates"""
    seen = set()
    unique = []
    for ex in examples:
        sig = (ex['instruction'][:100].lower(), ex['response'][:100].lower())
        if sig not in seen:
            seen.add(sig)
            unique.append(ex)
    return unique

def save_data(examples, output_file):
    """Save examples to jsonl"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            obj = {
                'instruction': ex['instruction'],
                'response': ex['response']
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print("="*70)
print("CORRECT MERGE: KEEP TRAINING & VALIDATION SEPARATE")
print("="*70)

# ============================================================
# LOAD & MERGE TRAINING DATA
# ============================================================
print("\n" + "="*70)
print("MERGING TRAINING DATA")
print("="*70)

old_train = load_and_clean(OLD_TRAIN, "Old training data")
new_train = load_and_clean(NEW_TRAIN, "New training data")

print(f"\nCombining: {len(old_train)} + {len(new_train)} = {len(old_train) + len(new_train)}")
train_combined = old_train + new_train

print("Deduplicating training data...", end=" ")
train_unique = deduplicate(train_combined)
duplicates_removed = len(train_combined) - len(train_unique)
print(f"OK Removed {duplicates_removed} duplicates")

# ============================================================
# LOAD & MERGE VALIDATION DATA
# ============================================================
print("\n" + "="*70)
print("MERGING VALIDATION DATA")
print("="*70)

old_val = load_and_clean(OLD_VAL, "Old validation data")
new_val = load_and_clean(NEW_VAL, "New validation data")

print(f"\nCombining: {len(old_val)} + {len(new_val)} = {len(old_val) + len(new_val)}")
val_combined = old_val + new_val

print("Deduplicating validation data...", end=" ")
val_unique = deduplicate(val_combined)
duplicates_removed_val = len(val_combined) - len(val_unique)
print(f"OK Removed {duplicates_removed_val} duplicates")

# ============================================================
# SAVE
# ============================================================
print("\n" + "="*70)
print("SAVING MERGED DATA")
print("="*70)

train_file = OUTPUT_DIR / "training_data.jsonl"
val_file = OUTPUT_DIR / "validation_data.jsonl"

print(f"\nSaving training data...", end=" ")
save_data(train_unique, train_file)
print(f"OK {len(train_unique)} examples")

print(f"Saving validation data...", end=" ")
save_data(val_unique, val_file)
print(f"OK {len(val_unique)} examples")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("MERGE COMPLETE")
print("="*70)

print(f"""
FINAL DATASET:
  Training data:   {len(train_unique):,} examples
  Validation data: {len(val_unique):,} examples
  Total:           {len(train_unique) + len(val_unique):,} examples

BREAKDOWN:
  Old training:    {len(old_train):,} examples
  New training:    {len(new_train):,} examples
  -> Training total: {len(train_unique):,} examples

  Old validation:  {len(old_val):,} examples
  New validation:  {len(new_val):,} examples
  -> Validation total: {len(val_unique):,} examples

DEDUPLICATION:
  Training duplicates removed:   {duplicates_removed}
  Validation duplicates removed: {duplicates_removed_val}
  Total duplicates removed:      {duplicates_removed + duplicates_removed_val}

FILES SAVED:
  * {train_file}
  * {val_file}

READY FOR FINE-TUNING WITH QDora
""")

# Verify
print("Verification sample:")
with open(train_file, 'r', encoding='utf-8') as f:
    first = json.loads(f.readline())
    has_inst = '[INST]' in json.dumps(first)
    has_prompt = 'You are a medical expert' in json.dumps(first)
    print(f"  NO [INST]: {not has_inst}")
    print(f"  NO system prompt: {not has_prompt}")
    print(f"  Keys: {list(first.keys())}")
