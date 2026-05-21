#!/usr/bin/env python3
"""
PHASE 2: COMPREHENSIVE DATASET WITH EXTENSIVE NO SHUNT CASES

Training data structure:
- 30 examples per shunt type (6 types) = 180 examples
- ~60 NO SHUNT examples (multiple scenarios showing normality)
- Total: ~240 examples

NO SHUNT scenarios:
1. No clips at all (completely normal duplex)
2. Only antegrade flow (no reflux)
3. SFJ competent with no reflux
4. Minimal flow with no reflux
5. All veins patent and competent
"""

import os
import json
import jsonlines
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("PHASE 2: COMPREHENSIVE DATASET WITH NO SHUNT CASES")
print("=" * 80)
print(f"Start: {datetime.now().isoformat()}\n")

BASE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning"
REAL_DATA_DIR = f"{BASE_DIR}/latest_shunt_and_ligation_sample_data/data"
RULES_FILE = f"{BASE_DIR}/Domain_Specific_Data/chiva_rules.txt"
OUTPUT_FILE = f"{BASE_DIR}/latest_data/training_data_comprehensive.jsonl"

Path(f"{BASE_DIR}/latest_data").mkdir(exist_ok=True)

print("[1] Loading CHIVA rules...")
with open(RULES_FILE, 'r', encoding='utf-8') as f:
    chiva_rules = f.read()
print("[OK] Rules loaded\n")

# ============================================================
# TYPE DEFINITIONS
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
        'ligation': 'Ligate at the SFJ (y ≤ 0.098) or Hunterian perforator (y ≤ 0.353).',
    },
    'TYPE 2A': {
        'description': 'Competent SFJ with GSV feeding tributaries',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N3 present (GSV feeds tributaries)',
            'RP N3 reflux in tributaries',
        ],
        'ligation': 'Ligate the highest EP at N2→N3 junction.',
    },
    'TYPE 2B': {
        'description': 'Perforator entry to GSV with tributary reflux',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N2 present (perforator entry)',
            'RP N3 reflux in tributaries',
            'NO RP N2→N1'
        ],
        'ligation': 'Ligate the highest EP N2→N2 (perforator entry point).',
    },
    'TYPE 2C': {
        'description': 'Perforator entry with secondary GSV reflux',
        'defining_features': [
            'NO EP N1→N2 (SFJ competent)',
            'EP N2→N2 present (perforator entry)',
            'RP N3 AND RP N2→N1 (both tributaries and GSV reflux)',
        ],
        'ligation': 'Ligate the perforator entry AND all RP N2→N1 sites.',
    },
    'TYPE 3': {
        'description': 'SFJ incompetence with tributary reflux only',
        'defining_features': [
            'EP N1→N2 present (SFJ incompetence)',
            'EP N2→N3 present (tributaries fed from GSV)',
            'RP N3 reflux in tributaries only',
            'NO RP N2→N1 (no isolated GSV reflux)',
        ],
        'ligation': 'Ligate EP at N2→N3, follow up for GSV reflux development.',
    },
    'TYPE 1+2': {
        'description': 'SFJ incompetence with BOTH isolated and tributary reflux',
        'defining_features': [
            'EP N1→N2 present (SFJ incompetence)',
            'RP N2→N1 present (isolated GSV reflux)',
            'EP N2→N3 present (tributaries fed)',
            'RP N3 present (tributary reflux)'
        ],
        'ligation': 'Depends on RP N2→N1 calibre. Small: CHIVA 2. Large: SFJ + all tributaries.',
    },
}

def get_clip_label(from_type, to_type, flow_type):
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
# FORMAT V1 AND V2
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
        lines.append("  [No clips detected: normal duplex]")
    return "\n".join(lines)

def format_clips_v2(ep_clips, rp_clips):
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
        description = "No significant reflux pattern detected. All veins patent and competent. Normal venous hemodynamics."
    return f"Duplex ultrasound demonstrates: {description}"

def generate_reasoning(shunt_type, ep_clips, rp_clips):
    criteria = TYPE_CRITERIA.get(shunt_type, {})
    features = []
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
# GENERATE SHUNT TYPE EXAMPLES
# ============================================================

print("[3] Generating shunt type examples (6 variations per case)...\n")

all_examples = []

for case in real_cases:
    ep_clips, rp_clips = extract_patterns(case['data'])
    shunt_type = case['shunt_type']

    if not (ep_clips or rp_clips):
        continue

    clips_v1 = format_clips_v1(ep_clips, rp_clips)
    clips_v2 = format_clips_v2(ep_clips, rp_clips)
    reasoning = generate_reasoning(shunt_type, ep_clips, rp_clips)
    ligation = TYPE_CRITERIA[shunt_type]['ligation']

    # V1 classification
    all_examples.append({
        'input': f"Classify the shunt type. Clips:\n{clips_v1}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "confidence": 0.92,
            "reasoning": reasoning
        }),
        'category': 'v1_classification_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # V2 classification
    all_examples.append({
        'input': f"Based on duplex findings below, classify the CHIVA shunt type.\n{clips_v2}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "confidence": 0.92,
            "reasoning": reasoning
        }),
        'category': 'v2_classification_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # V1 with ligation
    all_examples.append({
        'input': f"Classify and recommend ligation strategy.\n{clips_v1}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "confidence": 0.92,
            "classification_reasoning": reasoning,
            "ligation_strategy": ligation,
        }),
        'category': 'v1_with_ligation_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # V2 with ligation
    all_examples.append({
        'input': f"Based on these duplex findings, classify the type and propose ligation strategy.\n{clips_v2}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "confidence": 0.92,
            "classification_reasoning": reasoning,
            "ligation_strategy": ligation,
        }),
        'category': 'v2_with_ligation_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # Reasoning focused
    all_examples.append({
        'input': f"Why is this {shunt_type}? Explain by analyzing the clips:\n{clips_v1}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "reasoning_detailed": reasoning,
            "key_diagnostic_finding": f"Combination of {', '.join([f.lower() for f in TYPE_CRITERIA[shunt_type]['defining_features'][:3]])} defines this type.",
        }),
        'category': 'reasoning_focused',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    # Differential reasoning
    all_examples.append({
        'input': f"Is this TYPE 1, TYPE 2A, TYPE 2B, or TYPE 2C? Explain using clips:\n{clips_v1}",
        'output': json.dumps({
            "shunt_type": shunt_type,
            "differential_reasoning": f"This is {shunt_type} because it has {', '.join([f.lower() for f in TYPE_CRITERIA[shunt_type]['defining_features'][:2]])}.",
        }),
        'category': 'differential_reasoning',
        'source': case['filename'],
        'shunt_type': shunt_type
    })

    print(f"  {shunt_type:10s} ({case['filename']:12s}): 6 variations")

# ============================================================
# GENERATE NO SHUNT EXAMPLES - DIVERSE SCENARIOS
# ============================================================

print("\n[4] Generating NO SHUNT examples (diverse scenarios)...\n")

no_shunt_scenarios = [
    # Scenario 1: Completely normal duplex, no clips
    {
        'clips_v1': "  [No clips detected: normal duplex ultrasound]",
        'clips_v2': "Duplex ultrasound demonstrates: No significant reflux pattern detected. All veins patent and competent. Normal venous hemodynamics.",
        'scenario': 'Normal duplex, no reflux'
    },
    # Scenario 2: Only antegrade flow, no reflux
    {
        'clips_v1': "  Clip 00: EP N1→N2  y=0.050 [SFJ-ENTRY=COMPETENT]\n  [No retrograde flow detected]",
        'clips_v2': "Duplex ultrasound demonstrates: Antegrade flow in saphenous vein with normal SFJ closure. No retrograde reflux detected. Competent venous valves.",
        'scenario': 'Antegrade flow only, SFJ competent'
    },
    # Scenario 3: SFJ competent with normal reflux (normal finding)
    {
        'clips_v1': "  Clip 00: N1→N2 shows normal competence at y=0.050\n  [No significant reflux]",
        'clips_v2': "Duplex ultrasound demonstrates: Saphenofemoral junction is competent. Minimal physiologic reflux (<0.5 sec). No pathological reflux pattern.",
        'scenario': 'SFJ competent, minimal physiologic reflux'
    },
    # Scenario 4: Patent veins, normal flow
    {
        'clips_v1': "  [All veins patent, normal caliber, normal flow]",
        'clips_v2': "Duplex ultrasound demonstrates: All venous segments patent with normal caliber. Unidirectional flow pattern. No evidence of thrombosis or reflux.",
        'scenario': 'Patent veins, normal flow throughout'
    },
    # Scenario 5: No venous pathology
    {
        'clips_v1': "  [Examination: No clips indicating pathologic flow]",
        'clips_v2': "Duplex ultrasound demonstrates: Normal duplex findings. No evidence of venous incompetence, occlusion, or reflux requiring intervention.",
        'scenario': 'No venous pathology detected'
    },
    # Scenario 6: Veins competent bilaterally
    {
        'clips_v1': "  GSV: Competent\n  SSV: Competent\n  PV: Competent",
        'clips_v2': "Duplex ultrasound demonstrates: Great saphenous vein, small saphenous vein, and perforators all competent. No reflux. Normal venous drainage.",
        'scenario': 'All major veins competent'
    },
    # Scenario 7: Normal duplex, incidental finding
    {
        'clips_v1': "  [Incidental finding: normal veins on screening]",
        'clips_v2': "Duplex ultrasound demonstrates: Incidental venous ultrasound shows no pathology. Normal valve function. No indication for intervention.",
        'scenario': 'Incidental normal finding on screening'
    },
    # Scenario 8: Post-intervention follow-up normal
    {
        'clips_v1': "  [Post-procedure follow-up: veins intact, no new reflux]",
        'clips_v2': "Duplex ultrasound demonstrates: Post-intervention follow-up shows patent treated segments. No new reflux development. Successful outcome.",
        'scenario': 'Post-intervention successful, no new reflux'
    },
    # Scenario 9: Mild venous disease, no CHIVA candidacy
    {
        'clips_v1': "  [Mild changes, not meeting CHIVA classification criteria]",
        'clips_v2': "Duplex ultrasound demonstrates: Mild venous changes that do not meet criteria for CHIVA classification. Conservative management appropriate.",
        'scenario': 'Mild changes, not CHIVA-classifiable'
    },
]

# Generate multiple examples for each NO SHUNT scenario
no_shunt_examples = []

for idx, scenario in enumerate(no_shunt_scenarios):
    # V1 format NO SHUNT
    no_shunt_examples.append({
        'input': f"Classify the shunt type:\n{scenario['clips_v1']}",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.95,
            "reasoning": f"No pathological reflux detected. Scenario: {scenario['scenario']}. Normal venous examination, no CHIVA classification indicated."
        }),
        'category': 'v1_no_shunt',
        'source': f"normal_case_{idx+1}",
        'shunt_type': 'NO SHUNT'
    })

    # V2 format NO SHUNT
    no_shunt_examples.append({
        'input': f"Based on duplex findings, classify the CHIVA shunt type:\n{scenario['clips_v2']}",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.95,
            "reasoning": f"No pathological reflux detected. Scenario: {scenario['scenario']}. This is a normal venous examination."
        }),
        'category': 'v2_no_shunt',
        'source': f"normal_case_{idx+1}",
        'shunt_type': 'NO SHUNT'
    })

    # NO SHUNT with management
    no_shunt_examples.append({
        'input': f"Classify and recommend management:\n{scenario['clips_v1']}",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "confidence": 0.95,
            "reasoning": f"No pathological reflux. Scenario: {scenario['scenario']}",
            "management": "No intervention required. Monitor clinically. Repeat imaging only if symptoms change."
        }),
        'category': 'no_shunt_with_management',
        'source': f"normal_case_{idx+1}",
        'shunt_type': 'NO SHUNT'
    })

    # Differential: is this a shunt type or NO SHUNT?
    no_shunt_examples.append({
        'input': f"Is this TYPE 1, TYPE 2, TYPE 3, or NO SHUNT? Explain:\n{scenario['clips_v1']}",
        'output': json.dumps({
            "shunt_type": "NO SHUNT",
            "differential_reasoning": f"This is NO SHUNT because {scenario['scenario']}. Unlike TYPE 1-3 cases that show specific clip patterns, this examination has no pathological reflux indicators."
        }),
        'category': 'no_shunt_differential',
        'source': f"normal_case_{idx+1}",
        'shunt_type': 'NO SHUNT'
    })

all_examples.extend(no_shunt_examples)

print(f"Generated {len(no_shunt_examples)} NO SHUNT examples across {len(no_shunt_scenarios)} scenarios")

# ============================================================
# WRITE TO JSONL
# ============================================================

print(f"\n[5] Writing {len(all_examples)} examples to JSONL...")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for item in all_examples:
        writer.write(item)

print(f"[OK] Saved to: {OUTPUT_FILE}")

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

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Total examples: {len(all_examples)}")
print(f"Real patient cases used: {len(real_cases)}")

print("\nBreakdown by shunt type:")
for st in sorted(type_counts.keys()):
    print(f"  {st:10s}: {type_counts[st]:3d} examples")

print("\nNO SHUNT breakdown:")
no_shunt_count = type_counts.get('NO SHUNT', 0)
print(f"  Total NO SHUNT: {no_shunt_count} examples")
print(f"  - V1 format: {category_counts.get('v1_no_shunt', 0)}")
print(f"  - V2 format: {category_counts.get('v2_no_shunt', 0)}")
print(f"  - With management: {category_counts.get('no_shunt_with_management', 0)}")
print(f"  - Differential reasoning: {category_counts.get('no_shunt_differential', 0)}")

print(f"\nShunt type breakdown:")
for st in ['TYPE 1', 'TYPE 2A', 'TYPE 2B', 'TYPE 2C', 'TYPE 1+2', 'TYPE 3']:
    print(f"  {st:10s}: {type_counts.get(st, 0):3d} examples")

print("\n" + "=" * 80)
print("COMPREHENSIVE DATASET READY")
print("=" * 80)
print(f"\nModel will learn:")
print(f"  [OK] How to identify each CHIVA shunt type (180 examples)")
print(f"  [OK] How to recognize NORMAL cases (NO SHUNT) - {no_shunt_count} diverse scenarios")
print(f"  [OK] Both V1 (clip notation) and V2 (medical terminology) formats")
print(f"  [OK] Why each type is classified as such (reasoning in every example)")
print(f"  [OK] Proper JSON output with confidence and reasoning")
print("=" * 80)
