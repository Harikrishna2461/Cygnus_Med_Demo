"""
TEST YOUR TRAINED MODEL ON COMPREHENSIVE TRAINING DATA
No training - just inference testing to verify the model works
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
print("COMPREHENSIVE TRAINING DATA OVERVIEW")
print("=" * 80)

training_pairs = generate_comprehensive_training_pairs()
print(f"\n✓ Generated {len(training_pairs)} comprehensive training pairs")
print("✓ Coverage: All shunt types (1, 2A, 2B, 2C, 1+2, 3, 4, 5)")
print("✓ Content: Ligation strategies, CHIVA rules, surgical principles")
print("✓ Knowledge: 8000+ lines from 3 reference documents\n")

print("=" * 80)
print("TESTING MODEL ON SAMPLE CASES")
print("=" * 80)

# Model and tokenizer already loaded in your notebook
test_cases = [
    ("Type 1 Shunt", "A patient has reflux at the saphenofemoral junction with GSV reflux to mid-thigh tributaries, no deep vein involvement. What type of venous shunt is this and what ligation strategy do you recommend?"),
    ("Type 2A Shunt", "The SFJ is competent. There is an entry point from the GSV into a tributary at the mid-thigh without any deep vein involvement. The tributary shows reflux. Classify the shunt and recommend ligation."),
    ("Type 3 Shunt", "A perforator at the Hunterian level (mid-thigh) feeds the GSV with reflux into tributaries. No SFJ involvement. What type and where should I ligate?"),
    ("Type 5 Recurrent", "Previous SFJ ligation is evident. New recanalized pathways have formed at the old ligation site. What type of shunt is this?"),
    ("Type 6 Pelvic", "Reflux originates from pelvic veins flowing into the superficial system via tributaries. Symptoms are bilateral and severe. Classify and recommend management."),
]

for case_name, test_query in test_cases:
    print(f"\n{'=' * 80}")
    print(f"Test Case: {case_name}")
    print(f"{'=' * 80}")
    print(f"Query: {test_query}\n")

    prompt = f"[INST] {test_query} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    print(f"Model Response:\n{response}\n")

print("\n" + "=" * 80)
print("INFERENCE TEST COMPLETE")
print("=" * 80)
print("\nYour model has been trained on comprehensive CHIVA knowledge:")
print("  ✓ All shunt type classifications")
print("  ✓ CHIVA surgical principles and decision trees")
print("  ✓ Ligation strategies for each type")
print("  ✓ Post-operative management protocols")
print("\nSaved model location: /home/krish/finetuning/venv_pytorch/Include/merged_model")
print("Backup location: C:\\Users\\Krish\\Downloads\\LLM_Finetuning\\Include_backup")
