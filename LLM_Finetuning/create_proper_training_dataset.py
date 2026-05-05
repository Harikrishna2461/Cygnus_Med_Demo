"""
Create proper training and validation datasets for CHIVA shunt classification and ligation planning.
Combines real CHIVA medical knowledge with synthetic case variations.
Structured for supervised fine-tuning of Mistral-7B.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# CHIVA KNOWLEDGE BASE - From Medical Literature
# ─────────────────────────────────────────────────────────────────────────────

CHIVA_CLASSIFICATION_KNOWLEDGE = {
    "Type 1": {
        "definition": "SFJ incompetent with direct retrograde flow to deep vein without tributary involvement",
        "key_findings": [
            "EP N1→N2 (antegrade flow from deep vein to GSV at SFJ)",
            "RP N2→N1 (retrograde flow from GSV back to deep vein)",
            "NO EP N2→N3 (no tributary incompetence)",
            "NO RP at N3 (tributaries not involved)"
        ],
        "hemodynamics": "Direct reflux from GSV to deep system through incompetent SFJ. Simplest CHIVA shunt with localized incompetence.",
        "clinical_significance": "Most common CHIVA type. Pure SFJ disease without secondary tributary involvement.",
        "treatment": "SFJ ablation sufficient. No tributary treatment needed. Success rate high with isolated SFJ treatment.",
        "elimination_test": "Not applicable for Type 1 (no Type 3 confusion)"
    },
    "Type 2A": {
        "definition": "SFJ competent with competent perforator(s) allowing tributaries to reflux directly into deep system",
        "key_findings": [
            "NO EP N1→N2 (SFJ is competent)",
            "EP N2→N3 (antegrade flow from GSV to tributaries)",
            "RP N3→N2 or RP N3→N1 (tributaries reflux back)"
        ],
        "hemodynamics": "Competent SFJ and perforators allow tributaries to act as exclusive reflux source. Reflux enters deep system via perforators.",
        "clinical_significance": "Second most common pattern. Pure tributary incompetence with competent main truncal junctions.",
        "treatment": "Perforator ligation or ablation. GSV may be competent; focus on ablating tributary incompetence.",
        "treatment_details": "EVLA or foam sclerotherapy of tributaries. Some cases benefit from perforator ligation if perforator-driven.",
        "elimination_test": "Not applicable for Type 2A"
    },
    "Type 2B": {
        "definition": "SFJ competent with incompetent perforator(s) allowing isolated tributary reflux",
        "key_findings": [
            "NO EP N1→N2 (SFJ competent)",
            "EP N2→N2 (perforator antegrade entry point)",
            "RP N3 ONLY (tributary reflux)",
            "NO RP N2→N1 (no deep vein reflux)"
        ],
        "hemodynamics": "Perforator incompetence is sole entry point. Reflux limited to tributary system. Deep vein not involved directly.",
        "clinical_significance": "Pure perforator-driven tributary incompetence. Clean hemodynamic pattern.",
        "treatment": "Perforator ligation. High success rate with isolated perforator treatment.",
        "elimination_test": "Not applicable for Type 2B"
    },
    "Type 2C": {
        "definition": "SFJ competent with incompetent perforator(s) AND secondary deep vein reflux from tributaries",
        "key_findings": [
            "NO EP N1→N2 (SFJ competent)",
            "EP N2→N2 (perforator entry point)",
            "RP N3 present (tributary reflux)",
            "RP N2→N1 present (reflux from tributary system into deep vein)"
        ],
        "hemodynamics": "Perforator drives initial reflux, which then propagates secondarily to deep vein. More complex hemodynamics than Type 2B.",
        "clinical_significance": "Perforator-driven shunt with secondary deep system involvement. Requires more aggressive treatment.",
        "treatment": "Perforator ligation plus potential GSV ablation. May need combined approach.",
        "treatment_details": "Perforator ligation is primary. If deep reflux persists, secondary GSV/SSV treatment may be needed.",
        "elimination_test": "Not applicable for Type 2C"
    },
    "Type 3": {
        "definition": "SFJ incompetent with tributary involvement as exclusive reflux source. Deep vein not secondarily refluxing.",
        "key_findings": [
            "EP N1→N2 (SFJ incompetent)",
            "EP N2→N3 (antegrade to tributaries)",
            "RP N3 ONLY (tributary reflux only)",
            "NO RP N2→N1 (no deep vein reflux) OR RP N2→N1 + elim='No Reflux'"
        ],
        "hemodynamics": "SFJ allows antegrade flow to tributaries, which reflux back. Deep vein remains patent without secondary reflux.",
        "clinical_significance": "Mixed pattern but with tributary as exclusive/primary pathology. Distinguishable from Type 1+2 via elimination test.",
        "treatment": "GSV ablation plus tributary treatment. Perforator treatment usually not required.",
        "treatment_details": "SFJ closure (EVLA/RFA of GSV) addresses primary incompetence. Tributary ablation for symptomatic branches.",
        "elimination_test": "Critical for Type 3 vs Type 1+2 differentiation. 'No Reflux' after tributary elimination → Type 3"
    },
    "Type 1+2": {
        "definition": "SFJ incompetent with complex reflux through BOTH direct retrograde (like Type 1) AND tributary pathways (like Type 2)",
        "key_findings": [
            "EP N1→N2 (SFJ incompetent)",
            "EP N2→N3 (antegrade to tributaries)",
            "RP N3 present (tributary reflux)",
            "RP N2→N1 present (direct deep vein reflux)",
            "elim='Reflux' or elim not done (cannot isolate tributary as sole cause)"
        ],
        "hemodynamics": "Complex multi-pathway reflux. Both direct GSV-to-deep and GSV-to-tributary-to-deep pathways active.",
        "clinical_significance": "Most complex CHIVA shunt pattern. Multiple simultaneous reflux pathways.",
        "treatment": "GSV ablation essential. Perforator/tributary treatment may also be needed.",
        "treatment_details": "Primary: SFJ closure with EVLA/RFA of GSV. Secondary: Consider perforator/tributary ablation if residual symptoms.",
        "elimination_test": "Critical for Type 1+2 vs Type 3. 'Reflux' persists after tributary elimination → Type 1+2"
    }
}

LIGATION_PLANNING_KNOWLEDGE = {
    "General Principles": {
        "SFJ Ligation": "Standard approach for Type 1, Type 1+2, Type 3. Can be done via EVLA (endovenous laser), RFA (radiofrequency), or open surgery.",
        "Perforator Ligation": "Key for Type 2B and Type 2C. Hunterian perforator most commonly ligated. Can improve outcomes significantly.",
        "Tributary Ablation": "For symptomatic tributaries with reflux. EVLA, foam sclerotherapy, or ligation depending on size/location.",
        "Compression Therapy": "Adjunctive measure. 30-40 mmHg compression hose post-procedure improves healing and symptom relief.",
        "Follow-up Imaging": "Duplex ultrasound at 2-4 weeks post-procedure to assess treatment efficacy and identify failures."
    },
    "EVLA (Endovenous Laser Ablation)": {
        "indication": "Primary treatment for SFJ incompetence and larger tributaries (>3mm)",
        "procedure": "1064nm or 1470nm laser catheter placed endovenously, thermal energy ablates vein wall",
        "advantages": "Minimally invasive, short recovery, good cosmesis, high success rates (95%+)",
        "complications": "DVT (0.1-0.5%), burns, nerve injury (rare with tumescent anesthesia)",
        "success_rate": "95-98% at 1 year for SFJ ablation",
        "time_to_symptom_relief": "4-6 weeks"
    },
    "RFA (Radiofrequency Ablation)": {
        "indication": "Similar to EVLA, alternative for SFJ and large tributaries",
        "procedure": "RF catheter generates 60°C temperatures to ablate vein",
        "advantages": "Segmental ablation option, potentially less painful than EVLA",
        "complications": "DVT risk similar to EVLA, rare thermal injury",
        "success_rate": "95-97% at 1 year",
        "time_to_symptom_relief": "4-6 weeks"
    },
    "Foam Sclerotherapy": {
        "indication": "Tributaries, smaller veins, patients refusing surgery/endovenous procedures",
        "procedure": "Sclerosing agent (sodium tetradecyl sulfate, polidocanol) mixed to foam injected under ultrasound guidance",
        "advantages": "Office-based, no anesthesia, multiple sessions possible, lowest cost",
        "complications": "Matting (new spider veins), DVT (rare with proper technique), arterial injection (avoid)",
        "success_rate": "80-90% for initial response, higher with repeat sessions",
        "time_to_symptom_relief": "2-4 weeks per session"
    },
    "Open Ligation": {
        "indication": "Failed endovenous procedures, thrombotic disease, emergency settings",
        "procedure": "Surgical incision, identification, and ligation of incompetent vein",
        "advantages": "Definitive treatment, no recurrence if properly ligated",
        "complications": "Infection, hematoma, nerve injury, need for general anesthesia",
        "success_rate": "98%+ if properly performed",
        "time_to_symptom_relief": "6-8 weeks post-op"
    },
    "Treatment Selection by Type": {
        "Type 1": "EVLA/RFA of GSV at SFJ (primary). High success with SFJ treatment alone. Consider perforator if deep reflux persists.",
        "Type 2A": "Foam sclerotherapy or EVLA of tributaries. Perforator ligation if perforator-driven reflux identified.",
        "Type 2B": "Perforator ligation (definitive). GSV usually spared unless separately incompetent.",
        "Type 2C": "Perforator ligation (primary). May require secondary GSV ablation if residual deep reflux.",
        "Type 3": "GSV ablation (EVLA/RFA) + tributary ablation as indicated. Address SFJ incompetence primarily.",
        "Type 1+2": "GSV ablation mandatory. Concurrent perforator/tributary treatment often needed for optimal outcomes."
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED TRAINING DATA CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_classification_instruction(shunt_type: str, clips: List[Dict]) -> Dict:
    """Create a classification instruction-response pair."""

    # Filter out metadata-only entries (e.g., eliminationTest-only)
    flow_clips = [c for c in clips if 'flow' in c]

    clips_str = "\n".join([
        f"  • Clip {i+1}: {c['flow']} {c['fromType']}→{c['toType']} (position={c['posYRatio']:.3f})"
        + (f" [eliminationTest={c.get('eliminationTest')}]" if c.get('eliminationTest') else "")
        for i, c in enumerate(flow_clips)
    ])

    knowledge = CHIVA_CLASSIFICATION_KNOWLEDGE.get(shunt_type, {})

    instruction = f"""Analyze the following ultrasound clips and classify the CHIVA venous shunt type:

Clips:
{clips_str}

Based on the flow patterns and anatomical relationships, determine:
1. The CHIVA shunt type
2. Your confidence level
3. Clinical reasoning"""

    response = f"""CLASSIFICATION: {shunt_type}

CONFIDENCE: 0.95

ANATOMICAL FINDINGS:
{chr(10).join('• ' + finding for finding in knowledge.get('key_findings', []))}

HEMODYNAMIC INTERPRETATION:
{knowledge.get('hemodynamics', 'Complex reflux pattern identified.')}

CLINICAL SIGNIFICANCE:
{knowledge.get('clinical_significance', 'Hemodynamic abnormality requiring intervention.')}

TREATMENT CONSIDERATIONS:
{knowledge.get('treatment', 'Specialist evaluation recommended.')}"""

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "shunt_type": shunt_type,
        "type": "classification",
        "difficulty": "intermediate"
    }


def create_ligation_instruction(shunt_type: str) -> Dict:
    """Create a ligation planning instruction-response pair."""

    treatment_info = LIGATION_PLANNING_KNOWLEDGE.get("Treatment Selection by Type", {}).get(
        shunt_type,
        "Specialist evaluation needed."
    )

    instruction = f"For a {shunt_type} CHIVA venous shunt, outline the ligation strategy, procedure options, and expected outcomes."

    # Build comprehensive response
    response = f"""SHUNT TYPE: {shunt_type}

TREATMENT STRATEGY:
{treatment_info}

PROCEDURE OPTIONS:

1. ENDOVENOUS LASER ABLATION (EVLA)
{chr(10).join('   ' + line for line in [
    f"• Indication: {LIGATION_PLANNING_KNOWLEDGE['EVLA (Endovenous Laser Ablation)']['indication']}",
    f"• Success Rate: {LIGATION_PLANNING_KNOWLEDGE['EVLA (Endovenous Laser Ablation)']['success_rate']}",
    f"• Recovery: {LIGATION_PLANNING_KNOWLEDGE['EVLA (Endovenous Laser Ablation)']['time_to_symptom_relief']}",
    f"• Advantages: {LIGATION_PLANNING_KNOWLEDGE['EVLA (Endovenous Laser Ablation)']['advantages']}"
])}

2. RADIOFREQUENCY ABLATION (RFA)
{chr(10).join('   ' + line for line in [
    f"• Indication: {LIGATION_PLANNING_KNOWLEDGE['RFA (Radiofrequency Ablation)']['indication']}",
    f"• Success Rate: {LIGATION_PLANNING_KNOWLEDGE['RFA (Radiofrequency Ablation)']['success_rate']}",
    f"• Recovery: {LIGATION_PLANNING_KNOWLEDGE['RFA (Radiofrequency Ablation)']['time_to_symptom_relief']}"
])}

3. FOAM SCLEROTHERAPY
{chr(10).join('   ' + line for line in [
    f"• Indication: {LIGATION_PLANNING_KNOWLEDGE['Foam Sclerotherapy']['indication']}",
    f"• Success Rate: {LIGATION_PLANNING_KNOWLEDGE['Foam Sclerotherapy']['success_rate']}",
    f"• Cost-Effective: Yes, office-based procedure"
])}

ADDITIONAL CONSIDERATIONS:
• Compression therapy post-procedure (30-40 mmHg)
• Duplex ultrasound follow-up at 2-4 weeks
• Patient education on activity restrictions during recovery
• Symptom-driven approach for tributary treatment

EXPECTED OUTCOMES:
• Symptom relief: 80-95% of patients
• Recurrence rate: 5-15% at 2 years (modality-dependent)
• Functional improvement: Usually significant within 6 weeks"""

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "shunt_type": shunt_type,
        "type": "ligation",
        "difficulty": "intermediate"
    }


def create_anatomical_instruction() -> Dict:
    """Create anatomical reference instruction."""

    instruction = """Explain the key anatomical structures used in CHIVA classification and their clinical importance."""

    response = """CHIVA ANATOMICAL CLASSIFICATION:

N1 (DEEP VENOUS SYSTEM):
• Femoral vein (common, superficial, deep)
• Popliteal vein
• Role: Primary drainage pathway; reflux here indicates direct hemodynamic compromise

N2 (SAPHENOUS TRUNK):
• Great Saphenous Vein (GSV) - medial leg
• Small Saphenous Vein (SSV) - posterior leg
• Role: Secondary drainage; incompetence can cause direct or secondary reflux

N3 (TRIBUTARIES):
• Accessory GSV, anterior thigh veins
• Calf veins, branch veins
• Role: Superficial drainage; reflux indicates tributary-exclusive or tributary-driven pathology

FLOW PATTERNS:

EP (ANTEGRADE/PHYSIOLOGIC):
• Normal forward flow direction
• N1 → N2 (deep to saphenous, normal at SFJ)
• N2 → N3 (saphenous to tributaries, normal in physiologic state)
• Indicates patent, competent pathway

RP (RETROGRADE/PATHOLOGIC):
• Abnormal backward reflux
• N2 → N1 (saphenous to deep, indicates valvular incompetence)
• N3 → N2 or N3 → N1 (tributary reflux, indicates secondary incompetence)
• Indicates hemodynamic failure at one or more valve stations

JUNCTION ZONES:

SFJ (Saphenofemoral Junction):
• Location: GSV-femoral vein confluence
• posYRatio ≤ 0.098 (proximal 9.8% of limb)
• Clinical importance: Most common reflux entry point in CHIVA Type 1

HUNTERIAN PERFORATOR:
• Location: Mid-thigh, GSV to deep system
• posYRatio: 0.098-0.353 (9.8-35.3% of limb)
• Clinical importance: Entry point for Type 2B/2C shunts

CLINICAL APPLICATION:
Understanding these anatomical zones allows:
1. Identification of reflux entry points
2. Determination of shunt type
3. Planning of targeted intervention (SFJ vs perforator vs tributary)
4. Prediction of treatment success rates"""

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "type": "anatomy",
        "difficulty": "basic"
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC CASE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_clips_for_type(shunt_type: str, variation: int = 0) -> List[Dict]:
    """Generate realistic synthetic clips for each CHIVA type with variations."""

    base_cases = {
        "Type 1": [
            # Base case
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
            ],
            # Variation: Different SFJ position
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.060},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.280},
            ],
            # Variation: Hunterian entry
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.180},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.350},
            ],
        ],
        "Type 2A": [
            [
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.180},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.450},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.220},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.500},
            ],
        ],
        "Type 2B": [
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.250},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.480},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.220},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.520},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.280},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.490},
            ],
        ],
        "Type 2C": [
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.240},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.310},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.490},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.200},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.450},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.290},
            ],
            [
                {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.300},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.350},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.520},
            ],
        ],
        "Type 3": [
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
                {"eliminationTest": "No Reflux"},
            ],
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.070},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.150},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.240},
                {"eliminationTest": "No Reflux"},
            ],
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.090},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.170},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.300},
                {"eliminationTest": "No Reflux"},
            ],
        ],
        "Type 1+2": [
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.075},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.140},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.310},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.220},
                {"eliminationTest": "Reflux"},
            ],
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.085},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.160},
                {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.250},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.330},
                {"eliminationTest": "Reflux"},
            ],
            [
                {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.095},
                {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.180},
                {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.290},
                {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.270},
            ],
        ],
    }

    if shunt_type not in base_cases:
        return []

    cases = base_cases[shunt_type]
    return cases[variation % len(cases)] if variation < len(cases) else random.choice(cases)


def create_full_training_dataset(num_variations_per_type: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Create comprehensive training and validation datasets."""

    all_pairs = []
    shunt_types = ["Type 1", "Type 2A", "Type 2B", "Type 2C", "Type 3", "Type 1+2"]

    # 1. Anatomical foundations
    all_pairs.append(create_anatomical_instruction())

    # 2. Classification pairs with synthetic variations
    for shunt_type in shunt_types:
        for variation in range(num_variations_per_type):
            clips = generate_synthetic_clips_for_type(shunt_type, variation)
            if clips:
                pair = create_classification_instruction(shunt_type, clips)
                all_pairs.append(pair)

    # 3. Ligation planning pairs
    for shunt_type in shunt_types:
        pair = create_ligation_instruction(shunt_type)
        all_pairs.append(pair)

    # 4. Additional ligation details (procedures)
    procedure_pairs = [
        {
            "instruction": "What are the key advantages and complication rates of EVLA for treating venous insufficiency?",
            "input": "",
            "output": f"""ENDOVENOUS LASER ABLATION (EVLA) - ADVANTAGES & COMPLICATIONS

ADVANTAGES:
• Minimally invasive (small catheter, local anesthesia)
• Excellent cosmetic outcomes
• Quick recovery (1-2 days vs weeks for open surgery)
• High success rate (95-98% at 1 year)
• Can treat multiple veins in one session
• Reduced infection risk vs open surgery
• Effective for veins 3-15mm diameter

COMPLICATIONS:
Frequency          Complication
0.1-0.5%          Deep vein thrombosis (DVT)
1-2%              Superficial thrombophlebitis
<0.1%             Pulmonary embolism
0.2-1%            Burns or skin injury
0.5-2%            Nerve irritation (paresthesias)
Rare              Arterial injury
Rare              Thermal perforation

RISK REDUCTION:
• Proper tumescent anesthesia technique
• Correct laser wavelength (1064nm or 1470nm)
• Appropriate power settings
• Adequate compression post-procedure
• Patient selection (exclude acute thrombosis)""",
            "type": "ligation",
            "difficulty": "intermediate"
        },
        {
            "instruction": "When is open surgical ligation preferred over endovenous procedures for venous insufficiency?",
            "input": "",
            "output": """OPEN SURGICAL LIGATION - INDICATIONS

PREFERRED OVER ENDOVENOUS WHEN:

1. ACUTE THROMBOTIC DISEASE
   • Recent DVT with IVC filter
   • Thrombophilic disorders
   • Massive varicosities with thrombosis risk

2. FAILED ENDOVENOUS PROCEDURES
   • Recurrence after EVLA/RFA
   • Endovenous treatment not technically possible
   • Patient intolerance to minimally invasive approach

3. CONCURRENT PROCEDURES
   • Requiring abdominal exploration
   • Complex reconstruction needed
   • High-ligation with stump ligation indicated

4. SPECIFIC ANATOMICAL SITUATIONS
   • Aneurysmal GSV dilation
   • Severe reflux in emergency setting
   • Recurrent incompetence after previous procedures

5. PATIENT FACTORS
   • Pregnancy (defer EVLA, open safer)
   • Severe peripheral arterial disease
   • Inability to comply with compression/follow-up

SURGICAL APPROACH:
• High ligation at SFJ (saphenofemoral ligation)
• Selective stripping or ligation of GSV
• Careful technique to avoid complications
• Can combine with perforator ligation if needed

SUCCESS RATE:
• >98% if properly performed
• Lower recurrence with complete saphenous removal
• Takes 6-8 weeks for full functional recovery""",
            "type": "ligation",
            "difficulty": "intermediate"
        }
    ]
    all_pairs.extend(procedure_pairs)

    # Split into training (80%) and validation (20%)
    random.shuffle(all_pairs)
    split_point = int(0.8 * len(all_pairs))

    training_data = all_pairs[:split_point]
    validation_data = all_pairs[split_point:]

    return training_data, validation_data


# ─────────────────────────────────────────────────────────────────────────────
# FILE SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_datasets(training_data: List[Dict], validation_data: List[Dict]) -> None:
    """Save training and validation data to JSONL files."""

    output_dir = Path("./training_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_file = output_dir / "training_data.jsonl"
    validation_file = output_dir / "validation_data.jsonl"

    # Save training data
    with open(training_file, 'w') as f:
        for pair in training_data:
            f.write(json.dumps(pair) + '\n')

    # Save validation data
    with open(validation_file, 'w') as f:
        for pair in validation_data:
            f.write(json.dumps(pair) + '\n')

    print(f"[OK] Saved {len(training_data)} training pairs to {training_file}")
    print(f"[OK] Saved {len(validation_data)} validation pairs to {validation_file}")

    return training_file, validation_file


def create_summary_report(training_data: List[Dict], validation_data: List[Dict]) -> None:
    """Create detailed summary report."""

    output_dir = Path("./training_datasets")
    report_file = output_dir / "DATASET_SUMMARY.md"

    type_counts_train = {}
    type_counts_val = {}
    difficulty_counts = {}

    for pair in training_data:
        ptype = pair.get("type", "unknown")
        difficulty = pair.get("difficulty", "unknown")
        type_counts_train[ptype] = type_counts_train.get(ptype, 0) + 1
        difficulty_counts[f"{difficulty}"] = difficulty_counts.get(f"{difficulty}", 0) + 1

    for pair in validation_data:
        ptype = pair.get("type", "unknown")
        type_counts_val[ptype] = type_counts_val.get(ptype, 0) + 1

    with open(report_file, 'w') as f:
        f.write("# CHIVA Training Dataset Summary\n\n")

        f.write("## Dataset Statistics\n\n")
        f.write(f"**Total Training Pairs:** {len(training_data)}\n")
        f.write(f"**Total Validation Pairs:** {len(validation_data)}\n")
        f.write(f"**Combined Total:** {len(training_data) + len(validation_data)}\n\n")

        f.write("### Training Data by Type\n")
        for ptype, count in sorted(type_counts_train.items()):
            f.write(f"- {ptype}: {count}\n")

        f.write("\n### Validation Data by Type\n")
        for ptype, count in sorted(type_counts_val.items()):
            f.write(f"- {ptype}: {count}\n")

        f.write("\n### Difficulty Distribution\n")
        for difficulty, count in sorted(difficulty_counts.items()):
            pct = 100 * count / len(training_data)
            f.write(f"- {difficulty}: {count} ({pct:.1f}%)\n")

        f.write("\n## Data Source\n\n")
        f.write("- **Classification Data:** Synthetic cases generated from CHIVA medical literature principles\n")
        f.write("- **Ligation Planning Data:** Based on established clinical practice guidelines\n")
        f.write("- **Anatomical Reference:** CHIVA classification system documentation\n")
        f.write("- **Procedure Details:** Evidence-based treatment modalities (EVLA, RFA, foam sclerotherapy, open ligation)\n\n")

        f.write("## Training Configuration\n\n")
        f.write("**Recommended Fine-tuning Parameters:**\n")
        f.write("- Model: mistralai/Mistral-7B-Instruct-v0.2\n")
        f.write("- Method: LoRA with r=16\n")
        f.write("- Learning Rate: 1e-4\n")
        f.write("- Epochs: 5-10\n")
        f.write("- Batch Size: 4 (per device)\n")
        f.write("- Max Length: 512 tokens\n\n")

        f.write("## Usage Examples\n\n")
        f.write("### Training\n")
        f.write("```bash\n")
        f.write("python training_lora_from_medical_literature.py \\\n")
        f.write("  --train_data ./training_datasets/training_data.jsonl \\\n")
        f.write("  --val_data ./training_datasets/validation_data.jsonl\n")
        f.write("```\n\n")

        f.write("### Validation\n")
        f.write("```bash\n")
        f.write("python validate_fine_tuned_model.py \\\n")
        f.write("  --validation_file ./training_datasets/validation_data.jsonl\n")
        f.write("```\n\n")

        f.write("## Sample Training Pairs\n\n")
        f.write("### Classification Example\n")
        f.write(f"**Type:** {training_data[1].get('type')}\n")
        f.write(f"**Shunt Type:** {training_data[1].get('shunt_type')}\n")
        f.write(f"**Instruction:** {training_data[1].get('instruction')[:100]}...\n\n")

        f.write("### Ligation Planning Example\n")
        f.write(f"**Type:** {training_data[-1].get('type')}\n")
        f.write(f"**Instruction:** {training_data[-1].get('instruction')[:100]}...\n\n")

    print(f"[OK] Saved summary report to {report_file}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("CREATING PROPER CHIVA TRAINING DATASET")
    print("="*80)
    print("\nGenerating training data from CHIVA medical knowledge...")

    # Create datasets
    training_data, validation_data = create_full_training_dataset(num_variations_per_type=5)

    print(f"\n[OK] Generated {len(training_data)} training pairs")
    print(f"[OK] Generated {len(validation_data)} validation pairs")

    # Save datasets
    print("\nSaving datasets...")
    train_file, val_file = save_datasets(training_data, validation_data)
    create_summary_report(training_data, validation_data)

    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)
    print(f"\n[OUTPUT] Training Data: {train_file}")
    print(f"[OUTPUT] Validation Data: {val_file}")
    print(f"[OUTPUT] Summary: ./training_datasets/DATASET_SUMMARY.md")
    print("\n[OK] Ready for fine-tuning!")


if __name__ == "__main__":
    main()
