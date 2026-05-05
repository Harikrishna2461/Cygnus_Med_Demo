"""
QUICK INFERENCE TEST (no training)
Run this if training causes memory issues
"""

import sys
import os
import torch

torch.cuda.empty_cache()

wsl_path = "/mnt/c/Users/Krish/Downloads/LLM_Finetuning"
sys.path.insert(0, wsl_path)
os.chdir(wsl_path)

from training_data_comprehensive import generate_comprehensive_training_pairs

print("=" * 80)
print("TESTING COMPREHENSIVE TRAINING DATA")
print("=" * 80)

# Generate and display sample training pairs
training_pairs = generate_comprehensive_training_pairs()
print(f"\n✓ Generated {len(training_pairs)} comprehensive training pairs\n")

for i, pair in enumerate(training_pairs[:3], 1):
    print(f"--- Training Pair {i} ---")
    print(f"Input (first 200 chars):\n{pair['text'][:200]}...\n")
    print()

# Test inference on your loaded model
print("=" * 80)
print("INFERENCE TEST ON LOADED MODEL")
print("=" * 80)

test_cases = [
    "A patient has reflux at the saphenofemoral junction with GSV reflux to mid-thigh. What shunt type and ligation strategy?",
    "SFJ is competent. There is an entry from GSV into a tributary at mid-thigh. No DV involvement. Classify and recommend ligation.",
    "A perforator at Hunterian level feeds the GSV with reflux. No SFJ involvement. What type and where to ligate?",
]

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Query: {test}\n")

    prompt = f"[INST] {test} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    print(f"Response:\n{response}\n")

print("=" * 80)
print("Inference test complete!")
print("=" * 80)
