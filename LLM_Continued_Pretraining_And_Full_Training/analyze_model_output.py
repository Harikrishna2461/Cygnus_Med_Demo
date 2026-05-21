"""
Diagnostic script to analyze trained model output patterns.
Helps understand if the model is learning medical knowledge or just copying.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

TRAINED_MODEL_PATH = Path("medical_qwen_cpt")

print("="*80)
print("MEDICAL LLM OUTPUT ANALYSIS")
print("="*80)

# Load model
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(str(TRAINED_MODEL_PATH))
model = AutoModelForCausalLM.from_pretrained(
    str(TRAINED_MODEL_PATH),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

# Test prompts to understand model behavior
diagnostic_tests = [
    {
        "name": "Medical Knowledge Test",
        "prompt": "What is a Type 1 CHIVA shunt? Describe the hemodynamics."
    },
    {
        "name": "Instruction Following Test",
        "prompt": "Classify this shunt: EP at N1->N2, RP at N2->N1. Output JSON format only."
    },
    {
        "name": "Classification Test 1",
        "prompt": "Given clip data: EP at N1->N2, RP at N2->N1\nShunt type? (Type 1/Type 2A/Type 2B/Type 2C/Type 1+2/No Shunt)"
    },
    {
        "name": "Classification Test 2",
        "prompt": "Given clip data: EP at N2->N2, RP at N3->N1\nShunt type? (Type 1/Type 2A/Type 2B/Type 2C/Type 1+2/No Shunt)"
    },
    {
        "name": "Direct Classification",
        "prompt": "Analyze: EP at N1->N2, RP at N2->N1\nClassify as JSON: {\"shunt\": \"?\", \"logic\": \"?\"}"
    }
]

results = []

for test in diagnostic_tests:
    print(f"\n{'─'*80}")
    print(f"Test: {test['name']}")
    print(f"{'─'*80}")
    print(f"Prompt: {test['prompt']}")

    inputs = tokenizer(test['prompt'], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(test['prompt']):].strip()

    print(f"\nModel Output:")
    print(f"{response}")

    results.append({
        "test": test['name'],
        "prompt": test['prompt'],
        "output": response
    })

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\n✓ Model is responding")
print("✓ Check outputs above for:")
print("  - Medical knowledge (Type 1 definitions, CHIVA rules)")
print("  - JSON format compliance")
print("  - Actual classification vs. template copying")
print("  - Instruction following")

# Save for review
with open("logs/model_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed analysis saved to: logs/model_analysis.json")
print("="*80)
