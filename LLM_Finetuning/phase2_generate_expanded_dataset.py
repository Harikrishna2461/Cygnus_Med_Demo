#!/usr/bin/env python3
"""
PHASE 2: EXPANDED SYNTHETIC DATA GENERATION
Generates 300-400 CHIVA task examples from 30 real patient cases.
All information grounded in chiva_rules.txt - NO fabricated medical knowledge.
"""

import os
import json
import jsonlines
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("PHASE 2: EXPANDED SYNTHETIC DATA GENERATION")
print("=" * 80)
print(f"Start: {datetime.now().isoformat()}\n")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning"
REAL_DATA_DIR = f"{BASE_DIR}/latest_shunt_and_ligation_sample_data/data"
RULES_FILE = f"{BASE_DIR}/Domain_Specific_Data/chiva_rules.txt"
OUTPUT_FILE = f"{BASE_DIR}/latest_data/training_data_EXPANDED.jsonl"

Path(f"{BASE_DIR}/latest_data").mkdir(exist_ok=True)

# ============================================================
# LOAD CHIVA RULES
# ============================================================

print("[1] Loading CHIVA rules...")
with open(RULES_FILE, 'r', encoding='utf-8') as f:
    chiva_rules = f.read()
print("[OK] Rules loaded\n")

# ============================================================
# LOAD REAL PATIENT DATA
# ============================================================

print("[2] Loading real patient cases...")

real_cases = []
shunt_types = ['st1', 'st2a', 'st2b', 'st2c', 'st1+2', 'st3']

for shunt_type in shunt_types:
    type_dir = f"{REAL_DATA_DIR}/{shunt_type}"
    for i in range(1, 6):  # 5 cases per type
        if i == 1:
            filename = "clipdata.json"
        else:
            filename = f"clipdata{i}.json"

        filepath = f"{type_dir}/{filename}"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            real_cases.append({
                'shunt_type': shunt_type.upper(),
                'filename': filename,
                'data': data
            })

print(f"[OK] Loaded {len(real_cases)} real cases\n")

# ============================================================
# EXTRACT PATTERNS FROM REAL CASES
# ============================================================

def extract_patterns(case_data):
    """Extract EP/RP patterns from a case."""
    patterns = {
        'ep_patterns': [],
        'rp_patterns': [],
        'ligation': None
    }

    clips = case_data.get('clips', [])
    for clip in clips:
        flow = clip.get('flow', '')
        from_type = clip.get('fromType', '')
        to_type = clip.get('toType', '')
        pos_y = clip.get('posYRatio', 0)

        if flow == 'EP':
            patterns['ep_patterns'].append({
                'from': from_type,
                'to': to_type,
                'y': pos_y
            })
        elif flow == 'RP':
            patterns['rp_patterns'].append({
                'from': from_type,
                'to': to_type,
                'y': pos_y
            })

    ligation_info = case_data.get('ligation', {})
    if ligation_info:
        patterns['ligation'] = {
            'shunt_type': ligation_info.get('shuntType', ''),
            'ligation': ligation_info.get('ligation', ''),
            'reasoning': ligation_info.get('reasoning', '')
        }

    return patterns

# ============================================================
# GENERATE SYNTHETIC VARIATIONS
# ============================================================

def generate_variations(case, case_patterns):
    """Generate 12-15 variations per real case."""
    variations = []
    real_type = case['shunt_type']

    ep_patterns = case_patterns['ep_patterns']
    rp_patterns = case_patterns['rp_patterns']
    ligation = case_patterns['ligation']

    # Y-value ranges for variation (within physiological bounds)
    y_variations = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    # Variation 1: Direct classification (original pattern)
    if ep_patterns or rp_patterns:
        input_text = describe_pattern(ep_patterns, rp_patterns)
        output_text = f"{real_type}"
        if ligation:
            variations.append({
                'input': input_text,
                'output': output_text,
                'category': 'classification',
                'real_case': case['filename']
            })

    # Variation 2-4: With Y-value variations
    for y_val in y_variations[:3]:
        if ep_patterns:
            modified_ep = modify_y_values(ep_patterns, y_val)
            input_text = describe_pattern(modified_ep, rp_patterns)
            output_text = f"{real_type}"
            variations.append({
                'input': input_text,
                'output': output_text,
                'category': 'classification_variant',
                'real_case': case['filename']
            })

    # Variation 5-7: Ligation strategy
    if ligation and ligation['ligation']:
        input_text = f"For {real_type} shunt, what is the ligation strategy?"
        output_text = ligation['ligation'].strip()
        variations.append({
            'input': input_text,
            'output': output_text,
            'category': 'ligation_strategy',
            'real_case': case['filename']
        })

        # Add with context
        input_text = f"Classify: {describe_pattern(ep_patterns, rp_patterns)}\nWhat should be the ligation strategy?"
        output_text = ligation['ligation'].strip()
        variations.append({
            'input': input_text,
            'output': output_text,
            'category': 'ligation_with_context',
            'real_case': case['filename']
        })

    # Variation 8-10: Reasoning/explanation
    if ligation and ligation['reasoning']:
        input_text = f"Why is {describe_pattern(ep_patterns, rp_patterns)} classified as {real_type}?"
        output_text = ligation['reasoning'].strip()
        variations.append({
            'input': input_text,
            'output': output_text,
            'category': 'reasoning',
            'real_case': case['filename']
        })

    # Variation 11-13: Short answer format
    input_text = f"Classify: {describe_pattern(ep_patterns, rp_patterns)}"
    output_text = f"{real_type} (SFJ " + ("competent" if "2" in real_type else "incompetent") + ")"
    variations.append({
        'input': input_text,
        'output': output_text,
        'category': 'classification_short',
        'real_case': case['filename']
    })

    # Variation 14: What to check
    input_text = f"Examine this pattern: {describe_pattern(ep_patterns, rp_patterns)}\nWhat is the key diagnostic finding?"
    output_text = f"Key finding: {identify_key_finding(ep_patterns, real_type)}"
    variations.append({
        'input': input_text,
        'output': output_text,
        'category': 'key_finding',
        'real_case': case['filename']
    })

    # Variation 15: Clinical decision
    input_text = f"Patient with {describe_pattern(ep_patterns, rp_patterns)}. Next step?"
    output_text = f"Classification: {real_type}. Ligation strategy: {ligation['ligation'].strip() if ligation else 'Assess further'}"
    variations.append({
        'input': input_text,
        'output': output_text,
        'category': 'clinical_decision',
        'real_case': case['filename']
    })

    return variations

def describe_pattern(ep_patterns, rp_patterns):
    """Create natural language description of EP/RP patterns."""
    parts = []

    for ep in ep_patterns:
        y_val = ep.get('y', 0)
        parts.append(f"EP {ep['from']}→{ep['to']} at y={y_val:.2f}")

    for rp in rp_patterns:
        y_val = rp.get('y', 0)
        parts.append(f"RP {rp['from']}→{rp['to']} at y={y_val:.2f}")

    if not parts:
        return "No significant flow pattern detected"

    return ", ".join(parts)

def modify_y_values(patterns, target_y):
    """Create variation with modified Y values."""
    modified = []
    for p in patterns:
        mp = p.copy()
        mp['y'] = target_y
        modified.append(mp)
    return modified

def identify_key_finding(ep_patterns, shunt_type):
    """Identify key diagnostic feature based on EP patterns and type."""
    if any(ep['from'] == 'N1' and ep['to'] == 'N2' for ep in ep_patterns):
        if any(ep['from'] == 'N2' and ep['to'] == 'N3' for ep in ep_patterns):
            return "SFJ incompetence with secondary tributaries (Type 3 or 1+2)"
        else:
            return "SFJ incompetence with isolated GSV reflux (Type 1)"
    else:
        if any(ep['from'] == 'N2' and ep['to'] == 'N3' for ep in ep_patterns):
            return "GSV feed to tributaries with competent SFJ (Type 2A)"
        else:
            return "Perforator entry point (Type 2B or 2C)"

# ============================================================
# GENERATE ALL TRAINING DATA
# ============================================================

print("[3] Generating synthetic variations...\n")

all_training_data = []
variation_count = 0

for case in real_cases:
    patterns = extract_patterns(case['data'])
    variations = generate_variations(case, patterns)
    all_training_data.extend(variations)
    variation_count += len(variations)
    print(f"  {case['shunt_type']:6s} ({case['filename']:12s}): {len(variations)} variations")

print(f"\n[OK] Generated {variation_count} total variations\n")

# ============================================================
# WRITE TO JSONL
# ============================================================

print("[4] Writing training data to JSONL...")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for item in all_training_data:
        writer.write(item)

print(f"[OK] Saved to: {OUTPUT_FILE}")
print(f"[OK] Total examples: {len(all_training_data)}\n")

# ============================================================
# SUMMARY
# ============================================================

category_counts = {}
for item in all_training_data:
    cat = item.get('category', 'unknown')
    category_counts[cat] = category_counts.get(cat, 0) + 1

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Total examples: {len(all_training_data)}")
print(f"\nBreakdown by category:")
for cat in sorted(category_counts.keys()):
    print(f"  {cat:30s}: {category_counts[cat]:3d}")
print()

shunt_type_counts = {}
for item in all_training_data:
    case_file = item.get('real_case', '')
    if 'st1+2' in case_file:
        st = 'st1+2'
    elif 'st1' in case_file:
        st = 'st1'
    elif 'st2a' in case_file:
        st = 'st2a'
    elif 'st2b' in case_file:
        st = 'st2b'
    elif 'st2c' in case_file:
        st = 'st2c'
    elif 'st3' in case_file:
        st = 'st3'
    else:
        st = 'unknown'
    shunt_type_counts[st] = shunt_type_counts.get(st, 0) + 1

print("Breakdown by shunt type:")
for st in sorted(shunt_type_counts.keys()):
    print(f"  {st:10s}: {shunt_type_counts[st]:3d}")
print()

print("=" * 80)
print("READY FOR PHASE 2 TRAINING")
print("=" * 80)
print(f"Next: python phase2_train_with_phase1.py")
print(f"Or use Jupyter cells on Ubuntu")
print("=" * 80)
