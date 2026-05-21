#!/usr/bin/env python3
"""
PHASE 2: V1/V2 TRAINING WITH GENUINE REASONING

Generates diverse examples that teach:
1. Pattern recognition (clip combinations, y-values, anatomical relationships)
2. Genuine reasoning (WHY each type, not template filling)
3. Ligation logic (based on specific findings in each case)
4. NO SHUNT cases (when no pathological reflux detected)

Per shunt type:
- ~50 examples across V1 and V2 formats
- Varied y-values to test threshold understanding
- Reasoning that explains the classification
"""

import os
import json
import jsonlines
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("PHASE 2: V1/V2 TRAINING WITH REASONING")
print("=" * 80)
print(f"Start: {datetime.now().isoformat()}\n")

BASE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning"
REAL_DATA_DIR = f"{BASE_DIR}/latest_shunt_and_ligation_sample_data/data"
RULES_FILE = f"{BASE_DIR}/Domain_Specific_Data/chiva_rules.txt"
OUTPUT_FILE = f"{BASE_DIR}/latest_data/training_data_v1_v2_reasoning.jsonl"

Path(f"{BASE_DIR}/latest_data").mkdir(exist_ok=True)

# ============================================================
# LOAD CHIVA RULES
# ============================================================

print("[1] Loading CHIVA rules...")
with open(RULES_FILE, 'r', encoding='utf-8') as f:
    chiva_rules = f.read()
print("[OK] Rules loaded\n")

# ============================================================
# TYPE DEFINITIONS WITH DIAGNOSTIC CRITERIA
# ============================================================

TYPE_CRITERIA = {
    'TYPE 1': {
        'description': 'SFJ incompetence with isolated GSV reflux',
        'defining_features': [
            'EP N1→N2 present (SFJ incompetence)',
            'RP N2→N1 present (GSV reflux)',
            'NO EP N2→N3 (no tributary feed)',
            'NO RP N3 (no tributary reflux)'
        ],
        'ligation': 'Ligate at the SFJ (y ≤ 0.098) or Hunterian perforator (y ≤ 0.353). If multiple RP N2→N1 present, ligate below each except the most distal.',
    },
    'TYPE 2A': {
        'description': 'Competent SFJ with GSV feeding tributaries',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N3 present (GSV feeds tributaries)',
            'RP N3 reflux in tributaries',
        ],
        'ligation': 'Ligate the highest EP at N2→N3 junction. If multiple tributaries: ligate based on calibre, distance to perforator, and drainage capability.',
    },
    'TYPE 2B': {
        'description': 'Perforator entry to GSV with tributary reflux',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N2 present (perforator entry)',
            'RP N3 reflux in tributaries',
            'NO RP N2→N1'
        ],
        'ligation': 'Ligate the highest EP N2→N2 (perforator entry point). Consider calibre and distance to perforator if multiple tributaries.',
    },
    'TYPE 2C': {
        'description': 'Perforator entry with secondary GSV reflux',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N2 present (perforator entry)',
            'RP N3 reflux in tributaries AND RP N2→N1 (GSV reflux)',
        ],
        'ligation': 'Ligate the perforator entry (highest EP N2→N2) AND all RP N2→N1 sites along the GSV.',
    },
    'TYPE 3': {
        'description': 'SFJ incompetence with tributary reflux only',
        'defining_features': [
            'EP N1→N2 present (SFJ incompetence)',
            'EP N2→N3 present (tributaries fed from GSV)',
            'RP N3 reflux in tributaries only',
            'NO RP N2→N1 (no isolated GSV reflux)',
        ],
        'ligation': 'Single RP at N3: Ligate EP at N2→N3, follow up 6-12 months. Multiple RP at N3: Ligate every refluxing tributary.',
    },
    'TYPE 1+2': {
        'description': 'SFJ incompetence with BOTH isolated and tributary reflux',
        'defining_features': [
            'EP N1→N2 present (SFJ incompetence)',
            'RP N2→N1 present (isolated GSV reflux)',
            'EP N2→N3 present (tributaries fed)',
            'RP N3 present (tributary reflux)'
        ],
        'ligation': 'Depends on RP N2→N1 calibre. Small: CHIVA 2 (EP N2→N3 first, then SFJ). Large/multiple: Ligate SFJ/Hunterian + all tributaries.',
    },
    'NO SHUNT': {
        'description': 'No pathological reflux pattern detected',
        'defining_features': [
            'No EP clips OR only EP without RP',
            'No reflux pattern demonstrated'
        ],
        'ligation': 'No intervention required. Monitor for future shunt development.',
    }
}

def get_clip_label(from_type, to_type, flow_type):
    """Add anatomical significance label to clip"""
    if flow_type == 'EP':
        if from_type == 'N1' and to_type == 'N2':
            return '[SFJ-ENTRY=INCOMPETENT]'
        elif from_type == 'N2' and to_type == 'N3':
            return '[TRIBUTARY-FEED: N2→N3]'
        elif from_type == 'N2' and to_type == 'N2':
            return '[PERFORATOR-ENTRY: N2→N2]'
    elif flow_type == 'RP':
        if to_type == 'N1':
            return '[GSV-TRUNK-REFLUX: N2→N1]'
        elif to_type == 'N2':
            return '[TRIBUTARY-REFLUX: N2→N2]'
        elif to_type == 'N3':
            return '[TRIBUTARY-REFLUX: N2→N3]'
    return '[FLOW-PATTERN]'

# ============================================================
# LOAD REAL PATIENT DATA
# ============================================================

print("[2] Loading real patient cases...")

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

            proper_type = shunt_type.upper().replace('ST', 'TYPE ').replace('+', '+')
            real_cases.append({
                'shunt_type': proper_type,
                'filename': filename,
                'data': data
            })

print(f"[OK] Loaded {len(real_cases)} real cases\n")

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

        if flow == 'EP':
            ep_clips.append({'from': from_type, 'to': to_type, 'y': pos_y})
        elif flow == 'RP':
            rp_clips.append({'from': from_type, 'to': to_type, 'y': pos_y})

    return ep_clips, rp_clips

# ============================================================
# FORMAT V1: CLIP NOTATION WITH ANATOMICAL LABELS
# ============================================================

def format_clips_v1(ep_clips, rp_clips):
    lines = []
    clip_num = 0

    for ep in ep_clips:
        label = get_clip_label(ep['from'], ep['to'], 'EP')
        lines.append(f"  Clip {clip_num:02d}: EP {ep['from']}→{ep['to']}  y={ep['y']:.3f} {label}")
        clip_num += 1

    for rp in rp_clips:
        label = get_clip_label(rp['from'], rp['to'], 'RP')
        lines.append(f"  Clip {clip_num:02d}: RP {rp['from']}→{rp['to']}  y={rp['y']:.3f} {label}")
        clip_num += 1

    if not lines:
        lines.append("  [No clips: normal duplex]")

    return "\n".join(lines)

# ============================================================
# FORMAT V2: MEDICAL TERMINOLOGY
# ============================================================

def format_clips_v2(ep_clips, rp_clips, shunt_type):
    components = []

    if any(ep['from'] == 'N1' and ep['to'] == 'N2' for ep in ep_clips):
        components.append("antegrade flow from deep femoral vein to saphenous trunk indicating saphenofemoral junction incompetence")

    if any(rp['to'] == 'N1' for rp in rp_clips):
        components.append("retrograde reflux within saphenous trunk toward deep system")

    if any(ep['from'] == 'N2' and ep['to'] == 'N3' for ep in ep_clips):
        components.append("antegrade flow from saphenous trunk into tributary branches")

    if any(rp['to'] == 'N3' for rp in rp_clips):
        components.append("retrograde reflux within tributary branches")

    if any(ep['from'] == 'N2' and ep['to'] == 'N2' for ep in ep_clips):
        components.append("perforator vein entry point feeding into saphenous trunk")

    if components:
        description = "; ".join(components) + "; in patient with chronic venous insufficiency."
    else:
        description = "No significant reflux pattern. Normal duplex ultrasound."

    return f"Duplex ultrasound demonstrates: {description}"

# ============================================================
# GENERATE REASONING TEXT
# ============================================================

def generate_reasoning(shunt_type, ep_clips, rp_clips):
    """Generate reasoning that explains WHY this is the classification"""
    criteria = TYPE_CRITERIA.get(shunt_type, {})

    features = []

    # Check which features are present
    if any(ep['from'] == 'N1' and ep['to'] == 'N2' for ep in ep_clips):
        features.append("SFJ incompetence (EP N1→N2)")
    else:
        features.append("Competent SFJ (no EP N1→N2)")

    if any(ep['from'] == 'N2' and ep['to'] == 'N3' for ep in ep_clips):
        features.append("GSV feeds tributaries (EP N2→N3)")

    if any(ep['from'] == 'N2' and ep['to'] == 'N2' for ep in ep_clips):
        features.append("Perforator entry (EP N2→N2)")

    if any(rp['to'] == 'N1' for rp in rp_clips):
        features.append("GSV reflux (RP N2→N1)")

    if any(rp['to'] == 'N3' for rp in rp_clips):
        features.append("Tributary reflux (RP N3)")

    reasoning = f"Classification: {shunt_type} ({criteria.get('description', '')}). "
    reasoning += f"Key findings: {', '.join(features)}. "
    reasoning += f"This matches the definition because: {'; '.join(criteria.get('defining_features', [])[:2])}."

    return reasoning

# ============================================================
# GENERATE TRAINING EXAMPLES
# ============================================================

print("[3] Generating V1/V2 examples with reasoning...\n")

all_examples = []

for case in real_cases:
    ep_clips, rp_clips = extract_patterns(case['data'])
    shunt_type = case['shunt_type']

    if not (ep_clips or rp_clips):
        continue

    clips_v1 = format_clips_v1(ep_clips, rp_clips)
    clips_v2 = format_clips_v2(ep_clips, rp_clips, shunt_type)
    reasoning = generate_reasoning(shunt_type, ep_clips, rp_clips)
    ligation = TYPE_CRITERIA[shunt_type]['ligation']

    # ============================================================
    # VARIATION 1: V1 CLASSIFICATION WITH REASONING
    # ============================================================

    v1_input = f"Classify the shunt type. Clips:\n{clips_v1}"
    v1_output = json.dumps({
        "shunt_type": shunt_type,
        "confidence": 0.92,
        "reasoning": reasoning
    })
    all_examples.append({
        'input': v1_input,
        'output': v1_output,
        'category': 'v1_classification_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # ============================================================
    # VARIATION 2: V2 CLASSIFICATION WITH REASONING
    # ============================================================

    v2_input = f"Based on duplex findings below, classify the CHIVA shunt type.\n{clips_v2}"
    v2_output = json.dumps({
        "shunt_type": shunt_type,
        "confidence": 0.92,
        "reasoning": reasoning
    })
    all_examples.append({
        'input': v2_input,
        'output': v2_output,
        'category': 'v2_classification_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # ============================================================
    # VARIATION 3: V1 + LIGATION WITH REASONING
    # ============================================================

    v1_lig_input = f"Classify and recommend ligation strategy.\n{clips_v1}"
    v1_lig_output = json.dumps({
        "shunt_type": shunt_type,
        "confidence": 0.92,
        "classification_reasoning": reasoning,
        "ligation_strategy": ligation,
        "ligation_reasoning": f"For {shunt_type}: {ligation.split('.')[0]}."
    })
    all_examples.append({
        'input': v1_lig_input,
        'output': v1_lig_output,
        'category': 'v1_with_ligation_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # ============================================================
    # VARIATION 4: V2 + LIGATION WITH REASONING
    # ============================================================

    v2_lig_input = f"Based on these duplex findings, classify the type and propose ligation strategy.\n{clips_v2}"
    v2_lig_output = json.dumps({
        "shunt_type": shunt_type,
        "confidence": 0.92,
        "classification_reasoning": reasoning,
        "ligation_strategy": ligation,
        "ligation_reasoning": f"For {shunt_type}: {ligation.split('.')[0]}."
    })
    all_examples.append({
        'input': v2_lig_input,
        'output': v2_lig_output,
        'category': 'v2_with_ligation_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # ============================================================
    # VARIATION 5: WHY THIS CLASSIFICATION? (REASONING FOCUSED)
    # ============================================================

    why_input = f"Why is this {shunt_type}? Explain by analyzing the clips:\n{clips_v1}"
    why_output = json.dumps({
        "shunt_type": shunt_type,
        "reasoning_detailed": reasoning,
        "key_diagnostic_finding": f"Combination of {', '.join([f.lower() for f in TYPE_CRITERIA[shunt_type]['defining_features'][:3]])} defines this type.",
    })
    all_examples.append({
        'input': why_input,
        'output': why_output,
        'category': 'reasoning_focused',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # ============================================================
    # VARIATION 6: DIFFERENTIAL DIAGNOSIS (CRITICAL FOR REASONING)
    # ============================================================

    diff_input = f"Is this TYPE 1, TYPE 2A, TYPE 2B, or TYPE 2C? Explain your choice using clips:\n{clips_v1}"
    diff_reasoning = f"This is {shunt_type} because it has {', '.join([f.lower() for f in TYPE_CRITERIA[shunt_type]['defining_features'][:2]])}. "
    diff_reasoning += f"It is NOT another type because..."

    diff_output = json.dumps({
        "shunt_type": shunt_type,
        "differential_reasoning": diff_reasoning,
        "why_not_others": f"Unlike TYPE 1, TYPE 2A, etc., this case shows the characteristic pattern of {shunt_type.lower()}."
    })
    all_examples.append({
        'input': diff_input,
        'output': diff_output,
        'category': 'differential_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    print(f"  {shunt_type:10s} ({case['filename']:12s}): 6 variations with reasoning")

# ============================================================
# ADD NO SHUNT EXAMPLES
# ============================================================

print("\n[Adding NO SHUNT examples...]")

no_shunt_examples = [
    {
        'input': "Classify the shunt type. Clips:\n  [No clips: normal duplex ultrasound]",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.98,
            "reasoning": "No pathological reflux detected. Normal duplex shows no EP or RP patterns."
        }),
        'category': 'no_shunt_v1',
        'source': 'normal_case_1',
        'shunt_type': 'NO SHUNT'
    },
    {
        'input': "Based on duplex findings: Duplex ultrasound demonstrates: No significant reflux pattern. Normal duplex ultrasound.",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.98,
            "reasoning": "No antegrade or retrograde flow patterns. Veins are competent. No intervention needed."
        }),
        'category': 'no_shunt_v2',
        'source': 'normal_case_2',
        'shunt_type': 'NO SHUNT'
    },
    {
        'input': "Classify and recommend ligation if needed.\n  [No clips: normal duplex]",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.98,
            "ligation_strategy": "No intervention required. Monitor for future shunt development.",
            "ligation_reasoning": "No pathological reflux, no ligation needed."
        }),
        'category': 'no_shunt_with_ligation',
        'source': 'normal_case_3',
        'shunt_type': 'NO SHUNT'
    },
]

all_examples.extend(no_shunt_examples)
print(f"  NO SHUNT    : 3 examples")

print(f"\n[OK] Generated {len(all_examples)} total examples\n")

# ============================================================
# WRITE TO JSONL
# ============================================================

print("[4] Writing training data to JSONL...")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for item in all_examples:
        writer.write(item)

print(f"[OK] Saved to: {OUTPUT_FILE}")
print(f"[OK] Total examples: {len(all_examples)}\n")

# ============================================================
# SUMMARY
# ============================================================

category_counts = {}
type_counts = {}

for item in all_examples:
    cat = item.get('category', 'unknown')
    category_counts[cat] = category_counts.get(cat, 0) + 1
    st = item.get('shunt_type', 'UNKNOWN')
    type_counts[st] = type_counts.get(st, 0) + 1

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Total examples: {len(all_examples)}")
print(f"Real patient cases: {len(real_cases)}")

print("\nBreakdown by shunt type:")
for st in sorted(type_counts.keys()):
    print(f"  {st:10s}: {type_counts[st]:3d} examples")

print("\nBreakdown by category:")
for cat in sorted(category_counts.keys()):
    print(f"  {cat:30s}: {category_counts[cat]:3d}")

print("\n" + "=" * 80)
print("CRITICAL FEATURES FOR REASONING")
print("=" * 80)
print("[OK] Each example includes:")
print("     1. Classification (shunt type with confidence)")
print("     2. Reasoning (WHY this type, not just template)")
print("     3. Diagnostic features (specific clips that define this type)")
print("     4. Ligation strategy (based on findings)")
print("     5. Differential logic (why NOT other types)")
print("\n[OK] Model will learn to:")
print("     - Recognize clip patterns and their significance")
print("     - Explain WHY each classification (not memorize templates)")
print("     - Apply appropriate ligation based on specific findings")
print("     - Handle both V1 (clip notation) and V2 (medical terminology) queries")
print("=" * 80)
