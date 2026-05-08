#!/usr/bin/env python3
"""
FINAL CORRECT MERGE
- Load old training from training_datasets/training_data.jsonl (5544 examples)
- Load new training from latest_data/training_data_FRESH.jsonl (211 examples)
- Combine keeping TRAINING separate from VALIDATION
- NO re-splitting, keep original train/val boundary
"""

import json
import re
from pathlib import Path

OLD_TRAIN = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\training_data.jsonl")
OLD_VAL = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\training_datasets\validation_data.jsonl")
NEW_TRAIN_FRESH = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\training_data_FRESH.jsonl")
NEW_VAL_FRESH = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data\validation_data_FRESH.jsonl")
OUTPUT_DIR = Path(r"c:\Users\Krish\Downloads\LLM_Finetuning\latest_data")

def parse_old_format(text):
    """Extract instruction and response from old [INST] format"""
    text = text.replace("<s>", "").replace("</s>", "")

    if "[/INST]" in text:
        parts = text.split("[/INST]")
        if len(parts) == 2:
            inst = parts[0].replace("[INST]", "").strip()
            resp = parts[1].strip()

            inst = re.sub(
                r"You are a medical expert specialising in venous and lymphatic disorders.*?Answer questions accurately using clinical and scientific knowledge\.\s*\n\n",
                "",
                inst,
                flags=re.DOTALL
            )
            inst = inst.strip()

            if inst and resp:
                return inst, resp
    return None, None

def load_old_training():
    """Load old training data from training_datasets/"""
    print("Loading old training data (5544 examples)...", end=" ")
    examples = []

    with open(OLD_TRAIN, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                inst, resp = parse_old_format(obj['text'])
                if inst and resp:
                    examples.append({'instruction': inst, 'response': resp})
            except:
                pass

    print(f"OK loaded {len(examples)}")
    return examples

def load_old_validation():
    """Load old validation data from training_datasets/"""
    print("Loading old validation data (617 examples)...", end=" ")
    examples = []

    with open(OLD_VAL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                inst, resp = parse_old_format(obj['text'])
                if inst and resp:
                    examples.append({'instruction': inst, 'response': resp})
            except:
                pass

    print(f"OK loaded {len(examples)}")
    return examples

def load_new_training():
    """Load fresh extracted training data"""
    print("Loading new training data (211 examples)...", end=" ")
    examples = []

    with open(NEW_TRAIN_FRESH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'instruction' in obj and 'response' in obj:
                    examples.append({
                        'instruction': obj['instruction'],
                        'response': obj['response']
                    })
            except:
                pass

    print(f"OK loaded {len(examples)}")
    return examples

def load_new_validation():
    """Load fresh extracted validation data"""
    print("Loading new validation data (26 examples)...", end=" ")
    examples = []

    with open(NEW_VAL_FRESH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'instruction' in obj and 'response' in obj:
                    examples.append({
                        'instruction': obj['instruction'],
                        'response': obj['response']
                    })
            except:
                pass

    print(f"OK loaded {len(examples)}")
    return examples

def deduplicate(examples):
    """Remove duplicates based on instruction+response signature"""
    seen = set()
    unique = []

    for ex in examples:
        sig = (ex['instruction'][:100].lower(), ex['response'][:100].lower())
        if sig not in seen:
            seen.add(sig)
            unique.append(ex)

    return unique

def save_jsonl(examples, output_file):
    """Save examples to jsonl"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            obj = {
                'instruction': ex['instruction'],
                'response': ex['response']
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print("="*70)
print("FINAL CORRECT MERGE")
print("="*70)

# Load all data
print("\nLOADING DATA:")
old_train = load_old_training()
new_train = load_new_training()
old_val = load_old_validation()
new_val = load_new_validation()

# Merge training
print("\nMERGING TRAINING DATA:")
print(f"  Old: {len(old_train)} + New: {len(new_train)} = {len(old_train) + len(new_train)} total")
train_combined = old_train + new_train

print("  Deduplicating...", end=" ")
train_unique = deduplicate(train_combined)
train_removed = len(train_combined) - len(train_unique)
print(f"OK removed {train_removed} duplicates")
print(f"  Final training: {len(train_unique)} examples")

# Merge validation
print("\nMERGING VALIDATION DATA:")
print(f"  Old: {len(old_val)} + New: {len(new_val)} = {len(old_val) + len(new_val)} total")
val_combined = old_val + new_val

print("  Deduplicating...", end=" ")
val_unique = deduplicate(val_combined)
val_removed = len(val_combined) - len(val_unique)
print(f"OK removed {val_removed} duplicates")
print(f"  Final validation: {len(val_unique)} examples")

# Save
print("\nSAVING MERGED DATA:")
train_file = OUTPUT_DIR / "training_data.jsonl"
val_file = OUTPUT_DIR / "validation_data.jsonl"

print(f"  Saving training...", end=" ")
save_jsonl(train_unique, train_file)
print(f"OK")

print(f"  Saving validation...", end=" ")
save_jsonl(val_unique, val_file)
print(f"OK")

# Summary
print("\n" + "="*70)
print("MERGE COMPLETE")
print("="*70)

print(f"""
FINAL DATASET:
  Training:   {len(train_unique):,} examples
  Validation: {len(val_unique):,} examples
  Total:      {len(train_unique) + len(val_unique):,} examples

BREAKDOWN:
  Training:  {len(old_train)} (old) + {len(new_train)} (new) = {len(train_unique)} (merged, minus {train_removed} duplicates)
  Validation: {len(old_val)} (old) + {len(new_val)} (new) = {len(val_unique)} (merged, minus {val_removed} duplicates)

DATA CLEANLINESS:
  - NO [INST] tokens
  - NO system prompts
  - Clean instruction-response pairs
  - Ready for fine-tuning
""")

print("Files saved:")
print(f"  {train_file}")
print(f"  {val_file}")
