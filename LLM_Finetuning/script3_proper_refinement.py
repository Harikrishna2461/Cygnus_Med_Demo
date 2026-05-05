"""
Script 3: Proper Fine-Tuning for LLM Reasoning Engine
Trains the model to replicate the reasoning workflow from the unified classifier:
- Parse structured clip data
- Apply CHIVA decision logic
- Generate sound clinical reasoning
"""

import torch
import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
import numpy as np

# ============================================================================
# PART 1: GENERATE TRAINING DATA THAT MIRRORS THE ACTUAL WORKFLOW
# ============================================================================

CHIVA_RULES_SNIPPET = """
ANATOMY: N1=Deep system, N2=GSV/SSV trunk, N3=Tributaries
EP=Forward flow, RP=Reflux flow
SFJ COMPETENCE: EP N1→N2 = INCOMPETENT. EP N2→N2 = COMPETENT (perforator).

QUICK RULES:
- EP N1→N2 + RP N2→N1, no N3 → TYPE 1
- EP N1→N2 + EP N2→N3 + RP N3 only → TYPE 3
- No EP N1→N2 + EP N2→N3 → TYPE 2A
- No EP N1→N2 + EP N2→N2 + RP N3, no RP N2→N1 → TYPE 2B
- No EP N1→N2 + EP N2→N2 + RP N3 + RP N2→N1 → TYPE 2C
- No RP → NO SHUNT
"""

# Training examples that match actual workflow
TRAINING_EXAMPLES = [
    {
        "case_name": "Type 1 - Classic SFJ Incompetence",
        "clips": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.06, "step": "SFJ-ENTRY"},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.25},
        ],
        "expected_type": "Type 1",
        "expected_confidence": 0.92,
        "reasoning": [
            "STEP 1: Found EP N1→N2 with y=0.06 (SFJ region) → SFJ INCOMPETENT",
            "STEP 2: RP N2→N1 present (GSV reflux back to deep system)",
            "STEP 3: NO EP N2→N3 (no tributary entry), NO RP at N3",
            "STEP 4: Matches Case A pattern → TYPE 1",
            "Confidence: 0.92 (clear single pattern, no ambiguity)",
        ],
        "ligation_rationale": "Type 1 has primary SFJ incompetence with circular reflux (N1→N2→N1). The hemodynamic source is the incompetent SFJ. CHIVA principle: target the source. Ligate at SFJ (y ≤ 0.098) to interrupt the entrance of reflux from deep system. This is a single-source shunt, so SFJ ligation alone is curative. No tributary involvement, so no additional ligations needed.",
        "ligation_steps": [
            "Ligate saphenofemoral junction at the femorosaphenous level",
            "Optional: ligate any duplicate RP N2→N1 pathways below the main SFJ ligation (except the most distal)",
        ],
        "chiva_approach": "Hemodynamic CHIVA principle: identify and eliminate the source of reflux (incompetent SFJ). Preserves distal GSV, reducing limb edema and maintaining collateral drainage. Single ligation usually curative for Type 1.",
        "followup": "Follow-up ultrasound at 1 month post-op, then 6 months. If GSV remains patent with normalization of flow, consider surveillance only. If residual reflux develops distally, may need distal GSV ligation.",
    },
    {
        "case_name": "Type 2A - GSV Tributary Entry, Competent SFJ",
        "clips": [
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.20},
            {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.47},
        ],
        "expected_type": "Type 2A",
        "expected_confidence": 0.88,
        "reasoning": [
            "STEP 1: Scan for EP N1→N2... NOT FOUND → SFJ COMPETENT",
            "STEP 2: Found EP N2→N3 (GSV feeding tributary at y=0.20)",
            "STEP 3: RP N3→N2 present (tributary reflux)",
            "STEP 4: No RP N2→N1 (no GSV reflux), matches Case C TYPE 2A pattern",
            "Confidence: 0.88 (clear pattern, characteristic for 2A)",
        ],
        "ligation_rationale": "Type 2A: SFJ is competent; the shunt is driven by a tributary receiving forward flow from the GSV (EP N2→N3) which then drains retrogradely back into the superficial system (RP N3→N2). The source is the EP pathway at N2→N3 junction. CHIVA principle: preserve SFJ (it's working), target the tributary entry. Selective ligation of the GSV at the tributary junction eliminates the forward-flow entry while keeping SFJ and proximal GSV functional.",
        "ligation_steps": [
            "Identify the highest EP at N2→N3 junction (at y=0.20 level, mid-thigh)",
            "Ligate GSV at the tributary entry point (highest EP N2→N3)",
            "Preserve proximal GSV and SFJ for collateral drainage",
        ],
        "chiva_approach": "Hemodynamic reasoning: SFJ is competent, so SFJ ligation is unnecessary and would damage normal function. The problem is selective—one tributary is stealing forward flow from the GSV and draining it back superficially. Surgical principle: remove the abnormal pathway, not the normal junction. Preserves GSV as a collateral route.",
        "followup": "Follow-up ultrasound at 1 month and 6 months post-op. Assess for residual tributary reflux. If successful, GSV reflux may resolve passively as pressure gradients normalize. If multiple tributaries present, may require staged approach or additional ligation.",
    },
    {
        "case_name": "Type 2B - Perforator Entry, No GSV Reflux",
        "clips": [
            {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.050, "step": "SFJ-Knee"},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.132},
        ],
        "expected_type": "Type 2B",
        "expected_confidence": 0.84,
        "reasoning": [
            "STEP 1: Scan for EP N1→N2... NOT FOUND → SFJ COMPETENT",
            "STEP 2: Found EP N2→N2 (NOT N2→N3!) at y=0.050 → PERFORATOR ENTRY, not SFJ",
            "STEP 3: RP N3→N1 present (tributary reflux to deep system)",
            "STEP 4: NO RP N2→N1 (no GSV trunk reflux)",
            "STEP 5: Matches Case C TYPE 2B pattern → TYPE 2B",
            "Confidence: 0.84 (pattern is clear, N2→N2 confirms perforator)",
        ],
        "ligation_rationale": "Type 2B: Perforator-driven shunt. A perforator feeds the GSV (EP N2→N2, bidirectional flow within saphenous system), and then reflux escapes into tributaries (RP N3→N1). The GSV trunk itself has no reflux (no RP N2→N1). The source is the perforator. CHIVA principle: ligate the perforator, spare the GSV. This is distinctly different from Type 2A, which has an upstream (tributary) source. Type 2B has a lateral (perforator) source.",
        "ligation_steps": [
            "Identify and ligate the perforator (highest EP N2→N2 entry point) at the mid-thigh level",
            "Ligate above and below the perforator insertion to prevent recanalization",
            "Preserve GSV trunk—no need to ligate it unless secondary reflux develops",
        ],
        "chiva_approach": "Hemodynamic reasoning: GSV is competent (no RP N2→N1), so it's not a primary reflux source. The abnormal input is the perforator feeding the GSV. Remove the input, and the GSV normalizes passively. This preserves collateral function and reduces operative burden compared to Type 1 or 2A.",
        "followup": "Follow-up at 1 month to confirm perforator ligation. If tributary reflux persists, may require secondary ligation. Monitor for development of secondary GSV reflux at 6-12 months; if it emerges, may need late SFJ ligation.",
    },
    {
        "case_name": "Type 3 - SFJ Incompetent with Tributary Escape, No GSV Reflux",
        "clips": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.05, "step": "SFJ-ENTRY"},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
        ],
        "expected_type": "Type 3",
        "expected_confidence": 0.88,
        "reasoning": [
            "STEP 1: Found EP N1→N2 with y=0.05 → SFJ INCOMPETENT",
            "STEP 2: ALSO found EP N2→N3 (two-stage entry)",
            "STEP 3: RP N3→N1 present (reflux escapes through tributary back to deep)",
            "STEP 4: NO RP N2→N1 (GSV trunk itself does NOT reflux)",
            "STEP 5: Matches Case B1 pattern → TYPE 3",
            "Confidence: 0.88 (clear dual-entry pattern with single-return pathway)",
        ],
        "ligation_rationale": "Type 3: SFJ is incompetent, but the reflux doesn't circulate in the GSV (no RP N2→N1). Instead, antegrade flow from deep → GSV → tributaries, with reflux looping back through tributaries to deep system. The GSV is acting as a transit pathway, not a reflux source. CHIVA principle: ligate the tributary entry (first step), then reassess. This reduces load on the SFJ. If SFJ remains incompetent post-op but without secondary reflux, may not need SFJ ligation—hemodynamic situation may normalize with tributary closure.",
        "ligation_steps": [
            "Stage 1 (immediate): Ligate every refluxing tributary at N2→N3 junction (ligate the EP N2→N3 entry points)",
            "Stage 2 (if needed at 6-12 month follow-up): If SFJ remains incompetent with new GSV reflux development, ligate SFJ",
        ],
        "chiva_approach": "Hemodynamic reasoning: This is a 'staggered' shunt—reflux takes multiple steps to return (N1→N2→N3→N1). CHIVA principle: interrupt the pathway at the weakest point. Tributaries are easier to ligate than SFJ (less risk of saphenous vein sacrifice). First ligation (tributary) may normalize the SFJ hemodynamically by reducing reflux load. Follow-up at 6-12 months determines if SFJ needs ligation. This staged approach reduces operative burden and preserves vein function when possible.",
        "followup": "Critical follow-up at 6-12 months with duplex ultrasound. Assess for SFJ reflux development. Many Type 3 cases do NOT need SFJ ligation if tributary ligation alone normalizes the hemodynamic situation. If residual SFJ reflux with new GSV reflux develops, then SFJ ligation is indicated at that time.",
    },
    {
        "case_name": "Type 1+2 - SFJ Incompetent with Dual Reflux (via elimination test)",
        "clips": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.06, "step": "SFJ-ENTRY"},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132, "eliminationTest": "Reflux"},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.25},
        ],
        "expected_type": "Type 1+2",
        "expected_confidence": 0.80,
        "reasoning": [
            "STEP 1: Found EP N1→N2 with y=0.06 → SFJ INCOMPETENT",
            "STEP 2: ALSO found EP N2→N3 (tributary entry)",
            "STEP 3: Found BOTH RP N3→N1 (tributary reflux) AND RP N2→N1 (GSV reflux)",
            "STEP 4: eliminationTest='Reflux' is PRESENT → confirms Type 1+2 (not Type 3)",
            "STEP 5: Matches Case B4 pattern → TYPE 1+2",
            "Confidence: 0.80 (dual reflux pattern confirmed by elimination test)",
        ],
        "ligation_rationale": "Type 1+2: Complex dual-source shunt. SFJ is incompetent (Type 1 component), AND there's a secondary tributary reflux component (Type 2 component). Both the GSV trunk (RP N2→N1) and tributaries (RP N3→N1) are refluxing. The elimination test confirmed that RP N2→N1 is pathological and not just collateral. Strategy depends on calibre of RP N2→N1: if small, use CHIVA 2 (ligate tributaries first, then SFJ); if large, ligate SFJ + tributaries simultaneously. Here we'll use simultaneous ligation for clarity.",
        "ligation_steps": [
            "Ligate saphenofemoral junction at the femorosaphenous level",
            "Ligate every refluxing tributary at the N2→N3 junction simultaneously",
            "Ligate GSV below each major RP N2→N1 point except the most distal (if multiple)",
        ],
        "chiva_approach": "Hemodynamic reasoning: Two pathological sources must be addressed: (1) SFJ incompetence driving reflux into GSV trunk, and (2) reflux escaping into tributaries. The elimination test confirmed that tributary reflux is not just collateral drainage—it's pathological. Simultaneous ligation targets both sources. Alternative: CHIVA 2 staged approach—ligate tributaries first, reassess SFJ after 1 month, then SFJ ligation if reflux persists. Simultaneous approach used here for clarity and to address both pathologies definitively.",
        "followup": "Follow-up at 1 month to assess healing and confirm ligation effectiveness. At 6-12 months, reassess for recurrent reflux. Type 1+2 has higher recurrence risk than Type 1, so longer surveillance is warranted.",
    },
]

print("=" * 80)
print("GENERATING TRAINING DATA FOR LLM REASONING ENGINE")
print("=" * 80)

def format_clips_for_input(clips):
    """Format clip data as the LLM sees it during inference."""
    lines = []
    for i, c in enumerate(clips):
        flow = c.get("flow", "?")
        ft = c.get("fromType", "?")
        tt = c.get("toType", "?")
        y = c.get("posYRatio", 0.0)
        step = c.get("step", "")
        elim = c.get("eliminationTest", "")

        loc = ""
        if flow == "EP" and ft == "N1" and tt == "N2":
            if y <= 0.098:
                loc = " [SFJ-ENTRY=INCOMPETENT]"
            elif y <= 0.353:
                loc = " [Hunterian-ENTRY=INCOMPETENT]"
        elif flow == "EP" and ft == "N2" and tt == "N2":
            loc = " [PERFORATOR-ENTRY: N2→N2, SFJ=COMPETENT]"
        elif flow == "EP" and ft == "N2" and tt == "N3":
            loc = " [GSV-to-TRIBUTARY-ENTRY: N2→N3]"
        elif flow == "RP" and ft == "N3":
            loc = f" [TRIBUTARY-REFLUX: N3→{tt}]"
        elif flow == "RP" and ft == "N2" and tt == "N1":
            loc = " [GSV-TRUNK-REFLUX: N2→N1]"

        line = f"  Clip {i:02d}: {flow} {ft}→{tt}  y={y:.3f}{loc}"
        if step:
            line += f" step={step}"
        if elim:
            line += f' eliminationTest="{elim}"'
        lines.append(line)
    return "\n".join(lines)

training_pairs = []
for example in TRAINING_EXAMPLES:
    # Create instruction-response pairs that mirror actual workflow
    clips_formatted = format_clips_for_input(example["clips"])

    # Task 1: Classification reasoning
    classification_input = f"""=== CLINICAL ASSESSMENT: {example['case_name']} ===
Number of clips: {len(example['clips'])}
{clips_formatted}

{CHIVA_RULES_SNIPPET}

STEP-BY-STEP DECISION GUIDE:
1. Check for EP N1→N2 (SFJ entry)
2. Check for other EP patterns (N2→N3, N2→N2, N1→N3)
3. Check for RP patterns (N2→N1, N3→N1, N3→N2)
4. Match to CHIVA decision table
5. Assign shunt type and confidence

Classify the shunt type and provide step-by-step reasoning."""

    classification_response = f"""**Shunt Classification: {example['expected_type']}**

**Confidence: {example['expected_confidence']}**

**Decision Reasoning:**
{chr(10).join(f"  {step}" for step in example["reasoning"])}

**Clinical Summary:**
{example['case_name']}: {example['expected_type']} shunt with {len(example['clips'])} hemodynamically significant clips. Clear decision pattern matched to CHIVA classification rules."""

    training_pairs.append({
        "text": f"[INST] {classification_input} [/INST] {classification_response}",
        "type": "classification"
    })

    # Task 2: Ligation planning
    ligation_input = f"""=== LIGATION PLANNING ===
Shunt Type: {example['expected_type']}
Case: {example['case_name']}

{clips_formatted}

Based on the shunt type and clips, generate a detailed ligation plan with:
1. Specific ligation steps
2. Clinical rationale
3. CHIVA hemodynamic principles
4. Follow-up recommendations"""

    ligation_response = f"""**Ligation Plan for {example['expected_type']}**

**Ligation Steps:**
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(example["ligation_steps"]))}

**Clinical Rationale:**
{example['ligation_rationale']}

**CHIVA Hemodynamic Approach:**
{example['chiva_approach']}

**Follow-up Schedule:**
{example['followup']}"""

    training_pairs.append({
        "text": f"[INST] {ligation_input} [/INST] {ligation_response}",
        "type": "ligation"
    })

print(f"Generated {len(training_pairs)} training pairs")
print(f"\nExample training pair (classification):")
print(training_pairs[0]["text"][:500] + "...\n")

# ============================================================================
# PART 2: PREPARE DATASET
# ============================================================================

# Create balanced split
np.random.seed(42)
indices = np.random.permutation(len(training_pairs))
train_size = int(0.8 * len(training_pairs))
train_indices = indices[:train_size]
eval_indices = indices[train_size:]

train_data = [training_pairs[i] for i in train_indices]
eval_data = [training_pairs[i] for i in eval_indices]

train_dataset = Dataset.from_dict({
    "text": [ex["text"] for ex in train_data],
})

eval_dataset = Dataset.from_dict({
    "text": [ex["text"] for ex in eval_data],
})

print(f"Dataset split:")
print(f"  Training: {len(train_dataset)} pairs")
print(f"  Evaluation: {len(eval_dataset)} pairs")

# ============================================================================
# PART 3: LOAD MODEL WITH EXISTING LORA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device_map = "auto"
trained_model_path = "/home/krish/finetuning/venv_pytorch/Include/merged_model"
lora_adapter_path = Path("./mistral_lora_medical_finetuned")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# CHECK FOR FINE-TUNED MODEL FIRST (do NOT download base model if already trained)
if Path(trained_model_path).exists():
    print(f"✓ Found merged trained model at {trained_model_path}")
    print(f"Loading fine-tuned model (no cloud download)...")
    model = AutoModelForCausalLM.from_pretrained(
        trained_model_path,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path, trust_remote_code=True)
    print(f"✓ Loaded your trained model successfully")
elif lora_adapter_path.exists():
    print(f"✓ Found LoRA adapter at {lora_adapter_path}")
    print(f"Loading fine-tuned model (no cloud download)...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(lora_adapter_path),
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(lora_adapter_path), trust_remote_code=True)
    print(f"✓ Loaded your trained model successfully")
else:
    print(f"✗ No trained model found at {lora_adapter_path}")
    print(f"Downloading base model from cloud: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Creating new LoRA configuration...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

tokenizer.pad_token = tokenizer.eos_token
model.print_trainable_parameters()

# ============================================================================
# PART 4: TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FOR REASONING TASK")
print("=" * 80)

output_dir = "./mistral_reasoning_refined"
Path(output_dir).mkdir(exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_grad_norm=1.0,
    seed=42,
    bf16=True,
    optim="paged_adamw_32bit",
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    packing=False,
    dataset_text_field="text",
    max_seq_length=2048,
)

print("Starting training...")
train_result = trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print(f"Final training loss: {train_result.training_loss:.4f}")

# Save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# ============================================================================
# PART 5: EVALUATE ON NEW TEST CASES
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING ON NEW REASONING TEST CASES")
print("=" * 80)

model.eval()

test_cases = [
    {
        "name": "Type 2C Case",
        "clips": [
            {"flow": "EP", "fromType": "N2", "toType": "N2", "posYRatio": 0.050, "step": "SFJ-Knee"},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.132},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.212},
        ]
    },
    {
        "name": "Type 4 Case",
        "clips": [
            {"flow": "EP", "fromType": "N1", "toType": "N3", "posYRatio": 0.60},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.40},
        ]
    },
]

for test_case in test_cases:
    print(f"\n--- {test_case['name']} ---")
    clips_str = format_clips_for_input(test_case['clips'])

    prompt = f"""[INST] === CLASSIFICATION TASK ===
{clips_str}

{CHIVA_RULES_SNIPPET}

Classify this shunt using the CHIVA decision guide. Provide shunt type, confidence, and reasoning steps. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    print(f"Response:\n{response}")

print("\n" + "=" * 80)
print("FINE-TUNING FOR REASONING ENGINE COMPLETE")
print("=" * 80)

# Save final model
final_path = "./mistral_reasoning_final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

metadata = {
    "task": "shunt_classification_and_ligation_reasoning",
    "base_model": model_name,
    "training_pairs": len(train_data),
    "eval_pairs": len(eval_data),
    "epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "max_seq_length": 2048,
    "final_training_loss": float(train_result.training_loss),
    "approach": "Train model to reason through CHIVA decision trees using structured clip data",
}

with open(f"{final_path}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Final model saved to {final_path}")
