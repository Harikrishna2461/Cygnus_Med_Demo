#!/usr/bin/env python3
"""
UBUNTU TEST: Can the base Qwen2.5-7B model follow CHIVA instructions?
Run this BEFORE fine-tuning to verify the approach will work.

Usage:
  ssh user@ubuntu
  cd /home/user/llm_finetuning
  python test_base_model_ubuntu.py
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

print("=" * 100)
print("BASE MODEL INSTRUCTION-FOLLOWING TEST")
print("=" * 100)

BASE_MODEL = "Qwen/Qwen2.5-7B"
CACHE_DIR = "/home/username/llm_finetuning/.cache"  # CHANGE THIS

print(f"\nDevice info:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading {BASE_MODEL}...")
print("(First run will download ~15GB)\n")

try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

model.eval()

test_results = []

# ============================================================
# TEST 1: JSON OUTPUT INSTRUCTION
# ============================================================

print("=" * 100)
print("TEST 1: Can the model output JSON when instructed?")
print("=" * 100)

test1_input = """You are a CHIVA classification expert. Analyze the clips and output ONLY valid JSON.

Clips:
  Clip 00: EP N1→N2 at y=0.050 [SFJ-ENTRY=INCOMPETENT]
  Clip 01: RP N2→N1 at y=0.132 [GSV-TRUNK-REFLUX]

Output format: {"shunt_type": "TYPE X", "confidence": 0.0-1.0, "reasoning": "..."}

JSON output:"""

print("\nInput to model:")
print(test1_input)
print("\n" + "-" * 100)
print("Model output:")
print("-" * 100)

inputs = tokenizer.encode(test1_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response1_only = response1[len(test1_input):].strip()
print(response1_only)

# Check if valid JSON
test1_pass = False
try:
    json_obj = json.loads(response1_only)
    if 'shunt_type' in json_obj and 'confidence' in json_obj:
        print("\n✓ PASS: Valid JSON with required fields")
        test1_pass = True
    else:
        print("\n✗ FAIL: JSON missing required fields")
except json.JSONDecodeError as e:
    print(f"\n✗ FAIL: Not valid JSON - {e}")

test_results.append(("JSON Output Instruction", test1_pass))

# ============================================================
# TEST 2: FOLLOW CHIVA CLASSIFICATION RULES
# ============================================================

print("\n" + "=" * 100)
print("TEST 2: Can the model apply CHIVA rules correctly?")
print("=" * 100)

test2_input = """CHIVA Classification Rules:
- TYPE 1: IF EP N1→N2 present AND RP N2→N1 present AND NO EP N2→N3 → TYPE 1
- TYPE 3: IF EP N1→N2 AND EP N2→N3 AND RP N3 AND NO RP N2→N1 → TYPE 3
- TYPE 1+2: IF EP N1→N2 AND RP N2→N1 AND EP N2→N3 AND RP N3 → TYPE 1+2

Patient has: EP N1→N2, RP N2→N1, EP N2→N3, RP N3

Apply the rules. What type? Answer ONLY with the type name:"""

print("\nInput to model:")
print(test2_input)
print("\n" + "-" * 100)
print("Model output:")
print("-" * 100)

inputs = tokenizer.encode(test2_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response2_only = response2[len(test2_input):].strip().split('\n')[0]
print(response2_only)

test2_pass = False
if "TYPE 1+2" in response2_only:
    print("\n✓ PASS: Correctly identified TYPE 1+2 per CHIVA rules")
    test2_pass = True
else:
    print(f"\n✗ FAIL: Expected 'TYPE 1+2', got '{response2_only}'")

test_results.append(("Apply CHIVA Rules", test2_pass))

# ============================================================
# TEST 3: FOLLOW LIGATION INSTRUCTION
# ============================================================

print("\n" + "=" * 100)
print("TEST 3: Can the model follow ligation strategy instruction?")
print("=" * 100)

test3_input = """You are a CHIVA surgeon.

Classification: TYPE 1

Ligation rule for TYPE 1: "Ligate at the SFJ (y ≤ 0.098) or Hunterian (y ≤ 0.353)"

What should be done? Answer with just the ligation action:"""

print("\nInput to model:")
print(test3_input)
print("\n" + "-" * 100)
print("Model output:")
print("-" * 100)

inputs = tokenizer.encode(test3_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response3 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response3_only = response3[len(test3_input):].strip()
print(response3_only)

test3_pass = False
if "Ligate" in response3_only or "ligation" in response3_only.lower():
    print("\n✓ PASS: Model followed ligation instruction")
    test3_pass = True
else:
    print("\n✗ FAIL: Model did not mention ligation")

test_results.append(("Ligation Strategy Instruction", test3_pass))

# ============================================================
# TEST 4: HANDLE NO SHUNT CASE
# ============================================================

print("\n" + "=" * 100)
print("TEST 4: Can the model recognize NO SHUNT (normal case)?")
print("=" * 100)

test4_input = """Classify this duplex finding:

Duplex ultrasound demonstrates: No significant reflux pattern detected. All veins patent and competent. Normal venous hemodynamics.

Is this: TYPE 1, TYPE 2A, TYPE 3, or NO SHUNT?

Answer with ONLY the classification:"""

print("\nInput to model:")
print(test4_input)
print("\n" + "-" * 100)
print("Model output:")
print("-" * 100)

inputs = tokenizer.encode(test4_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response4 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response4_only = response4[len(test4_input):].strip().split('\n')[0]
print(response4_only)

test4_pass = False
if "NO SHUNT" in response4_only or "normal" in response4_only.lower():
    print("\n✓ PASS: Model correctly identified normal case")
    test4_pass = True
else:
    print(f"\n✗ FAIL: Expected 'NO SHUNT', got '{response4_only}'")

test_results.append(("NO SHUNT Recognition", test4_pass))

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("TEST RESULTS SUMMARY")
print("=" * 100)

passed = sum(1 for _, result in test_results if result)
total = len(test_results)

print(f"\nPassed: {passed}/{total}\n")

for test_name, result in test_results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {test_name}")

print("\n" + "=" * 100)
if passed == total:
    print("""
    ✓ ALL TESTS PASSED

    The base model CAN follow CHIVA instructions.
    Fine-tuning will ENHANCE these capabilities.

    PROCEED WITH PHASE 2 TRAINING:
    1. Copy training_data_comprehensive.jsonl to /home/user/llm_finetuning/latest_data/
    2. Copy Phase2_V1_V2_Training.ipynb to /home/user/llm_finetuning/
    3. Open Jupyter and run cells 1→2→3→4
    """)
elif passed >= 3:
    print("""
    ⚠ MOST TESTS PASSED (but not all)

    The model has instruction-following capability, but not perfect.
    Fine-tuning should still help, but results may be variable.

    PROCEED WITH CAUTION.
    """)
else:
    print("""
    ✗ TESTS FAILED - MODEL CANNOT FOLLOW INSTRUCTIONS

    The base model doesn't have strong instruction-following ability.
    Fine-tuning alone won't fix this.

    DO NOT PROCEED WITH PHASE 2.

    Consider:
    - Using a more instruction-tuned base model
    - Using Mistral-7B-Instruct instead
    - Adding instruction-following training to Phase 1
    """)

print("=" * 100)
