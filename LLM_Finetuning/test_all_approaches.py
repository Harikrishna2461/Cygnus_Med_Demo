"""
TEST ALL THREE SHUNT CLASSIFICATION APPROACHES
Compare unified inference, LoRA fine-tuning, and prompt-based methods
"""

import torch
import sys
import os

os.chdir("/mnt/c/Users/Krish/Downloads/LLM_Finetuning")
sys.path.insert(0, "/mnt/c/Users/Krish/Downloads/LLM_Finetuning")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unified_inference_script import classify_shunt_with_lora_model, build_shunt_classification_prompt

# ─────────────────────────────────────────────────────────────────────────────
# TEST DATA
# ─────────────────────────────────────────────────────────────────────────────

test_cases = {
    "Type 1": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080, "legSide": "Left"},
        {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300, "legSide": "Left"},
    ],
    "Type 2A": [
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200, "legSide": "Left"},
        {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470, "legSide": "Left"},
    ],
    "Type 3": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050, "legSide": "Left"},
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132, "legSide": "Left"},
        {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212, "legSide": "Left"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE MODEL & TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("LOADING BASE MODEL AND TOKENIZER")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

print("✓ Base model loaded\n")

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1: UNIFIED INFERENCE WITH EMBEDDED CHIVA RULES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("APPROACH 1: UNIFIED INFERENCE (EMBEDDED CHIVA RULES)")
print("=" * 80)

for expected, clips in test_cases.items():
    print(f"\nExpected: {expected}")

    prompt = build_shunt_classification_prompt(clips, "Left")
    full_prompt = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    # Extract TYPE from response
    import re
    type_match = re.search(r"TYPE:\s*([^\n]+)", response, re.IGNORECASE)
    detected_type = type_match.group(1).strip() if type_match else "Unknown"

    print(f"Detected: {detected_type}")
    print(f"Response preview: {response[:150]}...")
    print(f"✓ Match: {expected.lower() in detected_type.lower()}")

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2: LORA FINE-TUNED MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("APPROACH 2: LORA FINE-TUNED MODEL")
print("=" * 80)

try:
    lora_model = PeftModel.from_pretrained(model, "./lora_chiva_classifier")
    lora_model.gradient_checkpointing_disable()
    lora_model.eval()
    print("✓ LoRA model loaded\n")

    for expected, clips in test_cases.items():
        print(f"\nExpected: {expected}")

        prompt = build_shunt_classification_prompt(clips, "Left")
        full_prompt = f"[INST] {prompt} [/INST]"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(lora_model.device)

        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()

        type_match = re.search(r"TYPE:\s*([^\n]+)", response, re.IGNORECASE)
        detected_type = type_match.group(1).strip() if type_match else "Unknown"

        print(f"Detected: {detected_type}")
        print(f"Response preview: {response[:150]}...")
        print(f"✓ Match: {expected.lower() in detected_type.lower()}")

except Exception as e:
    print(f"✗ LoRA model not available: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 3: PROMPT-BASED FEW-SHOT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("APPROACH 3: PROMPT-BASED FEW-SHOT WITH EXPLICIT FORMATTING")
print("=" * 80)

for expected, clips in test_cases.items():
    print(f"\nExpected: {expected}")

    # Few-shot prompt with EXPLICIT EXAMPLES
    clips_str = "\n".join([
        f"Clip {i}: {c['flow']} {c['fromType']}→{c['toType']} y={c.get('posYRatio', 0):.3f}"
        for i, c in enumerate(clips)
    ])

    prompt = f"""You are a CHIVA venous shunt classifier. Classify shunts using ONLY these rules:

RULES:
- Check for EP N1→N2: YES=SFJ incompetent, NO=SFJ competent
- If EP N1→N2 + no EP N2→N3 + RP N2→N1 → TYPE 1
- If EP N1→N2 + EP N2→N3 + RP N3 only → TYPE 3
- If no EP N1→N2 + EP N2→N3 → TYPE 2A
- If no EP N1→N2 + EP N2→N2 + RP N3 only → TYPE 2B
- If no EP N1→N2 + EP N2→N2 + RP N3 + RP N2→N1 → TYPE 2C

EXAMPLE 1:
Clips: EP N1→N2, RP N2→N1
Steps: Has EP N1→N2? YES. Has EP N2→N3? NO. Has RP N2→N1? YES. Has RP at N3? NO.
Answer: Type 1

EXAMPLE 2:
Clips: EP N2→N3, RP N3→N2
Steps: Has EP N1→N2? NO. Has EP N2→N3? YES.
Answer: Type 2A

NOW CLASSIFY THIS:
{clips_str}

ANSWER: Type"""

    full_prompt = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    type_match = re.search(r"Type\s+(\d+[A-Z]*|\d+\+\d+|2[A-C])", response, re.IGNORECASE)
    detected_type = type_match.group(1) if type_match else "Unknown"

    print(f"Detected: {detected_type}")
    print(f"Response: {response[:100]}...")
    print(f"✓ Match: {expected.lower() in detected_type.lower()}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
print("\nSUMMARY:")
print("- Approach 1: Unified inference with embedded CHIVA rules")
print("- Approach 2: LoRA fine-tuned model on structured training data")
print("- Approach 3: Few-shot prompting with explicit examples")
