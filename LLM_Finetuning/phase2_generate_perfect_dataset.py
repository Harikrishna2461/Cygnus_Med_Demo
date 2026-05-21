#!/usr/bin/env python3
"""
PHASE 2: PERFECT SYNTHETIC DATASET GENERATION
High-quality CHIVA training examples with proper type names and ligation strategies.
All information grounded in chiva_rules.txt - NO fabricated medical info.
"""

import os
import json
import jsonlines
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("GENERATING PERFECT DATASET FOR PHASE 2 TRAINING")
print("=" * 80)

BASE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning"
REAL_DATA_DIR = f"{BASE_DIR}/latest_shunt_and_ligation_sample_data/data"
OUTPUT_FILE = f"{BASE_DIR}/latest_data/training_data_PERFECT.jsonl"

Path(f"{BASE_DIR}/latest_data").mkdir(exist_ok=True)

# ============================================================
# LIGATION STRATEGIES (FROM CHIVA RULES)
# ============================================================

LIGATION_STRATEGIES = {
    'TYPE 1': 'Ligate at the SFJ (y ≤ 0.098) or Hunterian perforator (y ≤ 0.353). If multiple RP N2→N1 present, ligate below each except the most distal.',
    'TYPE 2A': 'Ligate the highest EP at N2→N3 junction. If multiple tributaries: ligate based on calibre, distance to perforator, and drainage capability.',
    'TYPE 2B': 'Ligate the highest EP N2→N2 (perforator entry point). If multiple tributaries at N3: consider calibre and distance to perforator.',
    'TYPE 2C': 'Ligate the perforator entry (highest EP N2→N2) AND all RP N2→N1 sites along the GSV.',
    'TYPE 3': 'Single RP at N3: Ligate EP at N2→N3, follow up 6-12 months for GSV reflux development. Multiple RP at N3: Ligate every refluxing tributary at N2 junction.',
    'TYPE 1+2': 'Depends on RP N2→N1 calibre. Small: Apply CHIVA 2 (EP N2→N3 first, then SFJ). Large/multiple: Ligate SFJ/Hunterian + every refluxing tributary simultaneously.',
    'NO SHUNT': 'No ligation needed. Monitor for future shunt development.'
}

TYPE_DESCRIPTIONS = {
    'TYPE 1': 'SFJ/Hunterian incompetence with isolated GSV reflux. EP N1→N2 present, RP N2→N1, no EP N2→N3.',
    'TYPE 2A': 'Competent SFJ with GSV feeding tributaries. No EP N1→N2, EP N2→N3 present.',
    'TYPE 2B': 'Perforator entry to GSV with tributary reflux. No EP N1→N2, EP N2→N2 present, RP N3.',
    'TYPE 2C': 'Perforator entry with secondary GSV reflux. No EP N1→N2, EP N2→N2 present, RP N3 and RP N2→N1.',
    'TYPE 3': 'SFJ incompetence with tributary reflux only. EP N1→N2, EP N2→N3, RP N3 only (no RP N2→N1).',
    'TYPE 1+2': 'SFJ incompetence with both isolated and tributary reflux. EP N1→N2, EP N2→N3, RP N3, RP N2→N1 (eliminationTest="Reflux").',
    'NO SHUNT': 'No pathological reflux detected. Either no RP clips or only EP without RP.',
}

# ============================================================
# LOAD REAL PATIENT DATA
# ============================================================

print("\nLoading real patient cases...")

real_cases = []
shunt_types = ['st1', 'st2a', 'st2b', 'st2c', 'st1+2', 'st3']

for shunt_type in shunt_types:
    type_dir = f"{REAL_DATA_DIR}/{shunt_type}"
    for i in range(1, 6):
        filename = "clipdata.json" if i == 1 else f"clipdata{i}.json"
        filepath = f"{type_dir}/{filename}"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            real_cases.append({
                'shunt_type': shunt_type.upper().replace('ST', 'TYPE ').replace('+', '+'),
                'filename': filename,
                'data': data
            })

print(f"Loaded {len(real_cases)} real cases")

# ============================================================
# EXTRACT PATTERNS
# ============================================================

def extract_patterns(case_data):
    clips = case_data.get('clips', [])
    ep_clips = []
    rp_clips = []

    for clip in clips:
        flow = clip.get('flow', '')
        from_type = clip.get('fromType', '')
        to_type = clip.get('toType', '')
        pos_y = clip.get('posYRatio', 0)

        clip_info = {
            'from': from_type,
            'to': to_type,
            'y': f"{pos_y:.3f}",
            'y_float': pos_y
        }

        if flow == 'EP':
            ep_clips.append(clip_info)
        elif flow == 'RP':
            rp_clips.append(clip_info)

    return ep_clips, rp_clips

def format_clip(clip):
    return f"{clip['from']}→{clip['to']} at y={clip['y']}"

# ============================================================
# GENERATE TRAINING EXAMPLES
# ============================================================

print("Generating training examples...\n")

all_examples = []

for case in real_cases:
    ep_clips, rp_clips = extract_patterns(case['data'])
    shunt_type = case['shunt_type']

    # Get proper type name - case['shunt_type'] is already like 'TYPE 1', 'TYPE 2A', etc
    if '1+2' in shunt_type:
        type_name = 'TYPE 1+2'
    elif '2A' in shunt_type:
        type_name = 'TYPE 2A'
    elif '2B' in shunt_type:
        type_name = 'TYPE 2B'
    elif '2C' in shunt_type:
        type_name = 'TYPE 2C'
    elif '2' in shunt_type and '+' not in shunt_type and 'A' not in shunt_type and 'B' not in shunt_type and 'C' not in shunt_type:
        type_name = 'TYPE 2'
    elif '3' in shunt_type:
        type_name = 'TYPE 3'
    elif '1' in shunt_type:
        type_name = 'TYPE 1'
    else:
        type_name = shunt_type

    # Build pattern description
    pattern_parts = []
    for ep in ep_clips:
        pattern_parts.append(f"EP {format_clip(ep)}")
    for rp in rp_clips:
        pattern_parts.append(f"RP {format_clip(rp)}")

    if not pattern_parts:
        continue

    pattern_str = ", ".join(pattern_parts)

    # ============================================================
    # EXAMPLE 1: CLASSIFICATION WITH FULL CONTEXT
    # ============================================================

    all_examples.append({
        'input': f"Classify the CHIVA shunt type: {pattern_str}",
        'output': f"{type_name}. {TYPE_DESCRIPTIONS.get(type_name, 'Shunt detected')}",
        'category': 'classification_full',
        'source': case['filename'],
        'shunt_type': type_name
    })

    # ============================================================
    # EXAMPLE 2: CLASSIFICATION WITH EXPLANATION
    # ============================================================

    all_examples.append({
        'input': f"What is the CHIVA type for: {pattern_str}?",
        'output': type_name,
        'category': 'classification_short',
        'source': case['filename'],
        'shunt_type': type_name
    })

    # ============================================================
    # EXAMPLE 3: LIGATION STRATEGY
    # ============================================================

    ligation = LIGATION_STRATEGIES.get(type_name, 'Assess further')
    all_examples.append({
        'input': f"Patient has {type_name} shunt ({pattern_str}). What is the ligation strategy?",
        'output': ligation,
        'category': 'ligation_strategy',
        'source': case['filename'],
        'shunt_type': type_name
    })

    # ============================================================
    # EXAMPLE 4: LIGATION ONLY (SHORT)
    # ============================================================

    all_examples.append({
        'input': f"For {type_name} shunt, what is the ligation approach?",
        'output': ligation.split('.')[0] + '.',
        'category': 'ligation_short',
        'source': case['filename'],
        'shunt_type': type_name
    })

    # ============================================================
    # EXAMPLE 5: PATTERN RECOGNITION
    # ============================================================

    key_features = []
    if any('N1' in ep['from'] and 'N2' in ep['to'] for ep in ep_clips):
        key_features.append('SFJ incompetence')
    if any(ep['from'] == 'N2' and ep['to'] == 'N3' for ep in ep_clips):
        key_features.append('GSV tributaries')
    if any(rp['to'] == 'N3' for rp in rp_clips):
        key_features.append('tributary reflux')
    if any(rp['to'] == 'N2' for rp in rp_clips):
        key_features.append('GSV reflux')

    features_str = ', '.join(key_features) if key_features else 'reflux pattern'

    all_examples.append({
        'input': f"Identify the key features: {pattern_str}",
        'output': f"Key features: {features_str}. Classification: {type_name}.",
        'category': 'pattern_recognition',
        'source': case['filename'],
        'shunt_type': type_name
    })

    # ============================================================
    # EXAMPLE 6: CLINICAL DECISION
    # ============================================================

    all_examples.append({
        'input': f"Clinical case with {pattern_str}. What is your assessment?",
        'output': f"Assessment: {type_name}. Management: {ligation}",
        'category': 'clinical_decision',
        'source': case['filename'],
        'shunt_type': type_name
    })

    print(f"  {type_name:10s} ({case['filename']:12s}): 6 examples")

# ============================================================
# WRITE TO JSONL
# ============================================================

print(f"\nWriting {len(all_examples)} examples to JSONL...")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for item in all_examples:
        writer.write(item)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PERFECT DATASET CREATED")
print("=" * 80)

print(f"\nTotal examples: {len(all_examples)}")
print(f"File: {OUTPUT_FILE}\n")

# Count by type
type_counts = {}
for ex in all_examples:
    st = ex.get('shunt_type', 'UNKNOWN')
    type_counts[st] = type_counts.get(st, 0) + 1

print("Breakdown by shunt type:")
for st in sorted(type_counts.keys()):
    print(f"  {st:10s}: {type_counts[st]:3d} examples")

# Count by category
cat_counts = {}
for ex in all_examples:
    cat = ex.get('category', 'unknown')
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

print("\nBreakdown by category:")
for cat in sorted(cat_counts.keys()):
    print(f"  {cat:30s}: {cat_counts[cat]:3d} examples")

print("\n" + "=" * 80)
print("DATASET READY FOR TRAINING")
print("Use: training_data_PERFECT.jsonl")
print("=" * 80)
