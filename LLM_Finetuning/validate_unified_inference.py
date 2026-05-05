"""
VALIDATE UNIFIED INFERENCE - Quick sanity check
Tests the unified_inference_script on basic shunt cases
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 80)
print("VALIDATING UNIFIED INFERENCE")
print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

# Import unified inference module
import sys
import os
os.chdir("/mnt/c/Users/Krish/Downloads/LLM_Finetuning")
sys.path.insert(0, "/mnt/c/Users/Krish/Downloads/LLM_Finetuning")

from unified_inference_script import (
    build_shunt_classification_prompt,
    _repair_and_parse,
)

# Test cases
cases = {
    "Type 1": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
        {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
    ],
    "Type 2A": [
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200},
        {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470},
    ],
    "Type 3": [
        {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050},
        {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
        {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
    ],
}

model.eval()
results = []

for expected, clips in cases.items():
    print(f"\n{'─' * 80}")
    print(f"Test case: {expected}")
    print(f"{'─' * 80}")

    # Build prompt
    prompt = build_shunt_classification_prompt(clips, "Left")
    full_prompt = f"[INST] {prompt} [/INST]"

    # Generate
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

    # Parse
    parsed = _repair_and_parse(response)
    detected = parsed.get("shunt_type", "Unknown") if parsed else "Parse failed"
    confidence = parsed.get("confidence", 0.0) if parsed else 0.0

    print(f"Expected:  {expected}")
    print(f"Detected:  {detected}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw: {response[:200]}...")

    # Check if match
    match = expected.lower() in detected.lower()
    results.append((expected, detected, match))
    print(f"Status: {'✓ PASS' if match else '✗ FAIL'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

passed = sum(1 for _, _, m in results if m)
total = len(results)
print(f"Passed: {passed}/{total}")

for expected, detected, match in results:
    status = "✓" if match else "✗"
    print(f"{status} {expected:12} → {detected}")

print("\n" + "=" * 80)
if passed == total:
    print("✓ ALL TESTS PASSED - Unified inference is working correctly")
else:
    print(f"✗ {total - passed} test(s) failed - Review model outputs")
print("=" * 80)
