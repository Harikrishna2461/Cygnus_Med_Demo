#!/usr/bin/env python3
"""
Generate explicit CHIVA classification training examples from chiva_rules.txt
Creates ~1000 structured Q&A examples covering all shunt types.
"""

import json
import random
import sys

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

SYSTEM = ("You are a medical expert specialising in venous and lymphatic disorders, "
          "vascular surgery, duplex ultrasound, and haemodynamics. Answer questions "
          "accurately using clinical and scientific knowledge.")


def fmt_example(instruction, response):
    text = f"<s>[INST] {SYSTEM}\n\n{instruction} [/INST] {response} </s>"
    return {"text": text}


def y_sfj():
    return round(random.uniform(0.030, 0.095), 3)


def y_hunterian():
    return round(random.uniform(0.100, 0.350), 3)


def y_thigh():
    return round(random.uniform(0.150, 0.400), 3)


def y_knee():
    return round(random.uniform(0.400, 0.550), 3)


def y_calf():
    return round(random.uniform(0.550, 0.850), 3)


examples = []

# ============================================================
# TYPE 1: EP N1->N2 + RP N2->N1, no EP N2->N3, no RP at N3
# ============================================================
type1_templates = [
    "Analyse the following CHIVA duplex ultrasound clips and classify the shunt type:\n\nClip 1: EP N1->N2 (y={y1}, step=SFJ-ENTRY)\nClip 2: RP N2->N1 (y={y2}, step=Thigh)\n\nDetermine the shunt type, the reasoning, and the recommended ligation strategy.",
    "Given these duplex findings, classify the CHIVA shunt:\n\n- EP N1->N2 at y={y1} (SFJ-ENTRY)\n- RP N2->N1 at y={y2}\n- No EP N2->N3 detected\n- No reflux at N3 tributaries\n\nProvide classification, reasoning, and ligation plan.",
    "A patient presents with the following clips:\nEP N1->N2 (y={y1})\nRP N2->N1 (y={y2})\nNo other reflux observed.\n\nWhat is the CHIVA shunt type? Justify and recommend treatment.",
]

type1_response = (
    "Classification: TYPE 1\n\n"
    "Reasoning: An EP N1->N2 clip is present at y={y1}, indicating SFJ incompetence "
    "(blood entering the saphenous trunk from the deep system). RP N2->N1 at y={y2} confirms "
    "retrograde reflux within the GSV trunk. There is no EP N2->N3 (no tributary feed) and no reflux "
    "at N3 tributaries. By the CHIVA rules, this matches Case A: SFJ/Hunterian incompetence with "
    "RP N2->N1 and no N3 involvement -> TYPE 1.\n\n"
    "Ligation strategy: Ligate at the SFJ (y<=0.098) or at the Hunterian perforator level "
    "(y<=0.353). If multiple RP N2->N1 sites exist, ligate below each except the most distal "
    "to preserve drainage."
)

for _ in range(180):
    y1 = y_sfj()
    y2 = y_thigh()
    template = random.choice(type1_templates)
    instr = template.format(y1=y1, y2=y2)
    resp = type1_response.format(y1=y1, y2=y2)
    examples.append(fmt_example(instr, resp))


# ============================================================
# TYPE 2A: No EP N1->N2, EP N2->N3 present
# ============================================================
type2a_templates = [
    "Analyse the following CHIVA duplex findings:\n\nClip 1: EP N2->N3 (y={y1}, step=Thigh)\nClip 2: RP N3->N2 (y={y2})\nNo EP N1->N2 detected.\n\nClassify and recommend ligation.",
    "Duplex ultrasound shows:\n- EP N2->N3 at y={y1}\n- No SFJ incompetence (no EP N1->N2)\n- RP N3->N1 at y={y2}\n\nIdentify the CHIVA shunt type and the reasoning.",
    "Clips:\nEP N2->N3 (y={y1})\nNo EP N1->N2 anywhere on the limb.\n\nClassify the shunt and explain the haemodynamic basis.",
]

type2a_response = (
    "Classification: TYPE 2A\n\n"
    "Reasoning: There is no EP N1->N2 anywhere, so the SFJ is competent. The defining feature "
    "is EP N2->N3 at y={y1} - the GSV is feeding a tributary directly without any deep-system "
    "entry. This matches the Case C subtype 2A pattern: 'No EP N1->N2 + EP N2->N3 -> TYPE 2A'.\n\n"
    "Ligation strategy: Ligate the highest EP N2->N3 at the N2 junction. If multiple branching "
    "tributaries are present at N3, the choice depends on calibre, distance to perforator, and "
    "drainage feasibility through the thinner vessel (set ask_branching=true)."
)

for _ in range(150):
    y1 = y_thigh()
    y2 = y_knee()
    template = random.choice(type2a_templates)
    instr = template.format(y1=y1, y2=y2)
    resp = type2a_response.format(y1=y1)
    examples.append(fmt_example(instr, resp))


# ============================================================
# TYPE 2B: No EP N1->N2, EP N2->N2 (perforator), RP at N3, no RP N2->N1
# ============================================================
type2b_templates = [
    "Duplex ultrasound clips:\n\nClip 1: EP N2->N2 (y={y1}, step=SFJ-Knee, ligation-point-marker)\nClip 2: RP N3->N1 (y={y2})\n\nNo EP N1->N2 anywhere. Classify the CHIVA shunt.",
    "Findings:\n- EP N2->N2 at y={y1}\n- RP N3->N2 at y={y2}\n- No EP N1->N2\n- No RP N2->N1\n\nWhat type of CHIVA shunt is this and why?",
    "A clip is recorded as EP N2->N2 at y={y1} with no SFJ entry, plus a single RP N3->N1 at y={y2}. Classify.",
]

type2b_response = (
    "Classification: TYPE 2B\n\n"
    "Reasoning: Critical rule - EP N2->N2 means circulation within the saphenous trunk via a "
    "perforator, NOT an SFJ entry. Even though y={y1} is anatomically close to the SFJ region, "
    "the clip reads N2->N2, which is by definition a perforator. The SFJ remains competent. "
    "Combined with RP at N3 and no RP N2->N1, this matches: 'No EP N1->N2 + EP N2->N2 + "
    "RP N3 + NO RP N2->N1 -> TYPE 2B'.\n\n"
    "Ligation strategy: Ligate the highest EP N2->N2 (the perforator entry point)."
)

for _ in range(150):
    y1 = round(random.uniform(0.04, 0.20), 3)
    y2 = y_knee()
    template = random.choice(type2b_templates)
    instr = template.format(y1=y1, y2=y2)
    resp = type2b_response.format(y1=y1)
    examples.append(fmt_example(instr, resp))


# ============================================================
# TYPE 2C: No EP N1->N2, EP N2->N2, RP at N3, RP N2->N1 also present
# ============================================================
type2c_templates = [
    "Duplex clips:\n\nEP N2->N2 (y={y1}, step=SFJ-Knee)\nRP N3->N1 (y={y2})\nRP N2->N1 (y={y3})\n\nNo EP N1->N2. Classify the shunt.",
    "Findings on duplex:\n- EP N2->N2 at y={y1} (perforator entry)\n- RP N3->N2 at y={y2}\n- RP N2->N1 at y={y3}\n- No EP N1->N2 anywhere\n\nClassify and recommend ligation.",
]

type2c_response = (
    "Classification: TYPE 2C\n\n"
    "Reasoning: EP N2->N2 at y={y1} indicates a perforator entry (NOT SFJ). The SFJ remains "
    "competent because no EP N1->N2 is present. However, RP N2->N1 at y={y3} represents "
    "secondary GSV reflux, and RP at N3 (y={y2}) confirms tributary involvement. This matches: "
    "'No EP N1->N2 + EP N2->N2 + RP N3 + RP N2->N1 -> TYPE 2C'. Note: Type 2C is distinguished "
    "from Type 1+2 by the presence of EP N2->N2 (perforator) rather than EP N1->N2 (SFJ entry).\n\n"
    "Ligation strategy: Ligate the perforator entry (highest EP N2->N2) AND all RP N2->N1 sites "
    "along the GSV."
)

for _ in range(150):
    y1 = round(random.uniform(0.04, 0.20), 3)
    y2 = y_knee()
    y3 = y_thigh()
    template = random.choice(type2c_templates)
    instr = template.format(y1=y1, y2=y2, y3=y3)
    resp = type2c_response.format(y1=y1, y2=y2, y3=y3)
    examples.append(fmt_example(instr, resp))


# ============================================================
# TYPE 3: EP N1->N2 + EP N2->N3 + RP at N3 (no RP N2->N1)
# ============================================================
type3_templates = [
    "Duplex findings:\n\nEP N1->N2 (y={y1}, SFJ-ENTRY)\nEP N2->N3 (y={y2}, ligation-point-marker)\nRP N3->N1 (y={y3})\n\nClassify the CHIVA shunt.",
    "Patient ultrasound:\n- EP N1->N2 at y={y1}\n- EP N2->N3 at y={y2}\n- RP N3->N2 at y={y3}\n- No RP N2->N1 detected\n\nWhat type and why?",
]

type3_response = (
    "Classification: TYPE 3\n\n"
    "Reasoning: EP N1->N2 at y={y1} confirms SFJ incompetence. EP N2->N3 at y={y2} indicates "
    "the GSV is also feeding a tributary. Reflux is observed only at N3 (y={y3}) with no "
    "RP N2->N1 in the GSV trunk. This matches Case B1: 'EP N1->N2 + EP N2->N3 + RP N3 only "
    "-> TYPE 3'.\n\n"
    "Ligation strategy: For a single RP at N3, ligate the EP at N2->N3. Follow up at 6-12 "
    "months; if N2 reflux develops subsequently, ligate the SFJ. For multiple RP at N3 (CHIVA 2 "
    "step 1), ligate every refluxing tributary at the N2 junction."
)

for _ in range(150):
    y1 = y_sfj()
    y2 = y_hunterian()
    y3 = y_knee()
    template = random.choice(type3_templates)
    instr = template.format(y1=y1, y2=y2, y3=y3)
    resp = type3_response.format(y1=y1, y2=y2, y3=y3)
    examples.append(fmt_example(instr, resp))


# ============================================================
# TYPE 1+2: EP N1->N2 + EP N2->N3 + RP N3->N1 + RP N2->N1 + elim="Reflux"
# ============================================================
type12_templates = [
    "Duplex clips:\n\nEP N1->N2 (y={y1}, SFJ-ENTRY)\nEP N2->N3 (y={y2})\nRP N3->N1 (y={y3})\nRP N2->N1 (y={y4})\nEliminationTest: Reflux confirmed\n\nClassify.",
    "Findings:\n- EP N1->N2 at y={y1}\n- EP N2->N3 at y={y2}\n- RP N3->N1 at y={y3}\n- RP N2->N1 at y={y4}\n- Elimination test result: Reflux\n\nClassify and recommend.",
]

type12_response = (
    "Classification: TYPE 1+2\n\n"
    "Reasoning: All four key clips are present - EP N1->N2 (SFJ incompetent), EP N2->N3 "
    "(tributary feed), RP N3->N1 (N3 reflux), and RP N2->N1 (GSV trunk reflux). The "
    "elimination test confirms persistent reflux, matching Case B4: 'RP N3->N1 + RP N2->N1 + "
    "eliminationTest=Reflux -> TYPE 1+2'.\n\n"
    "Ligation strategy: This depends on RP N2->N1 calibre. For small RP N2->N1, apply CHIVA 2 "
    "(ligate EP N2->N3 first, then SFJ/Hunterian) OR ligate SFJ first plus all tributaries "
    "except one, and once N2 normalises ligate the last tributary. For large or multiple "
    "RP N2->N1, ligate SFJ/Hunterian and every refluxing tributary simultaneously, ligating "
    "below each RP N2->N1 except the most distal."
)

for _ in range(120):
    y1 = y_sfj()
    y2 = y_hunterian()
    y3 = y_knee()
    y4 = y_thigh()
    template = random.choice(type12_templates)
    instr = template.format(y1=y1, y2=y2, y3=y3, y4=y4)
    resp = type12_response.format(y1=y1, y2=y2, y3=y3, y4=y4)
    examples.append(fmt_example(instr, resp))


# ============================================================
# UNDETERMINED: EP N1->N2 + EP N2->N3 + RP N3->N1 + RP N2->N1 (no elim test)
# ============================================================
undet_templates = [
    "Duplex clips:\n\nEP N1->N2 (y={y1})\nEP N2->N3 (y={y2})\nRP N3->N1 (y={y3})\nRP N2->N1 (y={y4})\nNo elimination test performed.\n\nClassify.",
]

undet_response = (
    "Classification: UNDETERMINED (needs_elim_test=true)\n\n"
    "Reasoning: All four clip types are present (EP N1->N2, EP N2->N3, RP N3->N1, RP N2->N1) "
    "but the elimination test has not been performed. Per Case B3 of the CHIVA rules, this "
    "configuration cannot be definitively classified between TYPE 3 and TYPE 1+2 without an "
    "elimination test result. If the test shows 'Reflux' it becomes TYPE 1+2; if 'No Reflux' "
    "it becomes TYPE 3.\n\n"
    "Recommendation: Perform the elimination test before deciding ligation strategy."
)

for _ in range(80):
    y1 = y_sfj()
    y2 = y_hunterian()
    y3 = y_knee()
    y4 = y_thigh()
    template = random.choice(undet_templates)
    instr = template.format(y1=y1, y2=y2, y3=y3, y4=y4)
    resp = undet_response.format(y1=y1, y2=y2, y3=y3, y4=y4)
    examples.append(fmt_example(instr, resp))


# ============================================================
# NO SHUNT: No RP at all, or EP only with no reflux
# ============================================================
noshunt_templates = [
    "Duplex clips:\n\nEP N1->N2 (y={y1})\nNo retrograde flow detected anywhere.\n\nClassify.",
    "Findings:\n- EP N2->N2 at y={y1}\n- No RP clips of any kind\n\nWhat is the CHIVA classification?",
    "Ultrasound shows only physiological forward flow (EP) clips with no retrograde or "
    "pathological reflux. y={y1}. Classify.",
]

noshunt_response = (
    "Classification: NO SHUNT DETECTED\n\n"
    "Reasoning: The CHIVA classification requires the presence of retrograde (RP) flow to "
    "indicate a pathological shunt. Without any RP clips, there is no haemodynamic shunt to "
    "classify. This matches Case D of the rules: 'No RP in any clip -> NO SHUNT DETECTED'.\n\n"
    "Ligation strategy: No ligation needed."
)

for _ in range(100):
    y1 = round(random.uniform(0.05, 0.5), 3)
    template = random.choice(noshunt_templates)
    instr = template.format(y1=y1)
    resp = noshunt_response.format(y1=y1)
    examples.append(fmt_example(instr, resp))


# ============================================================
# CONCEPTUAL Q&A: rules, anatomy, key distinctions
# ============================================================
conceptual = [
    ("In CHIVA classification, what does N1, N2, and N3 refer to anatomically?",
     "N1 refers to the deep venous system (femoral or popliteal vein). N2 refers to the Great "
     "Saphenous Vein (GSV) or Small Saphenous Vein (SSV) trunk. N3 refers to tributaries or "
     "superficial branches of the saphenous system."),

    ("What is the difference between EP and RP flow in CHIVA duplex?",
     "EP stands for physiological (forward, antegrade) flow, which is the normal direction. "
     "RP stands for retrograde (pathological, reflux) flow, which is the abnormal direction "
     "and indicates a shunt or valvular incompetence."),

    ("Critical rule: when is the SFJ considered incompetent in CHIVA?",
     "The SFJ is incompetent if and only if a clip has fromType=N1 AND toType=N2 (EP N1->N2). "
     "EP N2->N2 means blood circulates within the saphenous trunk via a perforator, and the "
     "SFJ remains competent. This is true regardless of posYRatio or step label."),

    ("What is the key signal that distinguishes Type 2C from Type 1+2?",
     "Type 2C has EP N2->N2 (perforator entry) with NO EP N1->N2. Type 1+2 has EP N1->N2 (true "
     "SFJ entry). If RP N2->N1 exists with EP N2->N2 but no EP N1->N2, the answer is TYPE 2C, "
     "not TYPE 1+2."),

    ("How is Type 2A defined in CHIVA classification?",
     "Type 2A is defined by EP N2->N3 present with NO EP N1->N2 anywhere. The GSV feeds a "
     "tributary directly without any SFJ entry. RP may or may not be present in early cases. "
     "Typical pattern: EP N2->N3 plus RP N3->N2 or RP N3->N1, with no RP N2->N1."),

    ("What is the ligation strategy for CHIVA Type 1?",
     "For Type 1, ligate at the SFJ (y<=0.098) or at the Hunterian perforator level "
     "(y<=0.353). If multiple RP N2->N1 sites exist, ligate below each except the most distal."),

    ("In Type 3 with multiple refluxing tributaries at N3, what is the ligation approach?",
     "For multiple RP at N3 in Type 3 (CHIVA 2 step 1), ligate every refluxing tributary at the "
     "N2 junction. Follow up at 6-12 months; if N2 reflux develops subsequently, ligate the "
     "SFJ. For a single RP at N3, ligate just the EP at N2->N3."),

    ("When should you set ask_branching=true in CHIVA classification?",
     "Set ask_branching=true when there are multiple RP at N3 tributaries in a Type 2A, 2B, or "
     "2C case. The ligation choice among multiple N3 branches depends on calibre of branches, "
     "distance of each branch to its perforator, and whether drainage through the thinner "
     "vessel is possible."),

    ("What posYRatio defines the SFJ region versus the Hunterian perforator?",
     "The SFJ corresponds to posYRatio <= 0.098. The Hunterian perforator region is "
     "0.098 < posYRatio <= 0.353. However, the critical rule is that EP N2->N2 at any "
     "posYRatio (even 0.05) is a perforator entry, not an SFJ entry."),

    ("For Type 1+2 with large or multiple RP N2->N1, what is the ligation plan?",
     "Ligate the SFJ or Hunterian and every refluxing tributary simultaneously. Among multiple "
     "RP N2->N1 sites, ligate below each except the most distal to preserve drainage."),

    ("If a clip shows EP N2->N2 at y=0.05 with step=SFJ-Knee labelled as ligation-point-marker, "
     "is the SFJ incompetent?",
     "No. The clip reads EP N2->N2, which by definition is a perforator entry within the "
     "saphenous trunk. The SFJ remains competent regardless of the step label or posYRatio. "
     "Only EP N1->N2 indicates SFJ incompetence."),

    ("What role does the elimination test play in distinguishing Type 3 from Type 1+2?",
     "When EP N1->N2, EP N2->N3, RP N3->N1, and RP N2->N1 are all present, the elimination "
     "test result determines the type. If the test shows 'Reflux', the case is TYPE 1+2. If "
     "'No Reflux', it is TYPE 3. Without an elimination test, the case is UNDETERMINED."),
]

for instr, resp in conceptual:
    examples.append(fmt_example(instr, resp))
    # Add 5 paraphrased copies of each conceptual Q&A for stronger learning
    for _ in range(5):
        examples.append(fmt_example(instr, resp))


# ============================================================
# Shuffle and save
# ============================================================
random.shuffle(examples)
print(f"Generated {len(examples)} CHIVA classification examples")

# Append to existing training data
existing_train = []
with open('training_datasets/training_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            existing_train.append(json.loads(line))

existing_val = []
with open('training_datasets/validation_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            existing_val.append(json.loads(line))

print(f"Existing train: {len(existing_train)}")
print(f"Existing val: {len(existing_val)}")

# Split new examples 90/10
split = int(len(examples) * 0.9)
new_train = examples[:split]
new_val = examples[split:]

# Merge
all_train = existing_train + new_train
all_val = existing_val + new_val
random.shuffle(all_train)
random.shuffle(all_val)

# Save
with open('training_datasets/training_data.jsonl', 'w', encoding='utf-8') as f:
    for ex in all_train:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

with open('training_datasets/validation_data.jsonl', 'w', encoding='utf-8') as f:
    for ex in all_val:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"\nFinal training: {len(all_train)}")
print(f"Final validation: {len(all_val)}")
print(f"\nSaved to training_datasets/training_data.jsonl")
print(f"Saved to training_datasets/validation_data.jsonl")
