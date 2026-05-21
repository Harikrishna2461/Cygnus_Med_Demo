"""
Detailed diagnostic to understand model's shunt type classifications.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

TRAINED_MODEL_PATH = Path("medical_qwen_cpt")

print("="*80)
print("DIAGNOSTIC: What has the model learned about shunt classification?")
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

# Test each type with direct prompts
shunt_definitions = {
    "Type 1": "What is a Type 1 shunt? Define it.",
    "Type 2A": "What is a Type 2A shunt? Define it.",
    "Type 2B": "What is a Type 2B shunt? Define it.",
    "Type 2C": "What is a Type 2C shunt? Define it.",
    "Type 1+2": "What is a Type 1+2 combined shunt? Define it.",
    "No Shunt": "What does 'No Shunt' mean in CHIVA classification?"
}

print("\n" + "="*80)
print("TEST 1: Model's understanding of each shunt type")
print("="*80)

for shunt_type, question in shunt_definitions.items():
    print(f"\n{shunt_type}:")
    print(f"Q: {question}")

    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(question):].strip()
    print(f"A: {response[:150]}...")

# Test if model can distinguish types by their characteristics
print("\n" + "="*80)
print("TEST 2: Can model distinguish based on flow patterns?")
print("="*80)

flow_patterns = [
    ("EP N1->N2, RP N2->N1", "Type 1 expected"),
    ("EP N2->N2, RP N3->N1", "Type 2A/2B expected"),
    ("EP N1->N2, EP N2->N3, RP N3->N2, RP N2->N1", "Type 1+2 expected"),
]

for pattern, expected in flow_patterns:
    print(f"\n{pattern}")
    print(f"(Expected: {expected})")

    prompt = f"Shunt flow: {pattern}. Type?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    print(f"Model says: {response[:100]}")

# Test JSON output reliability
print("\n" + "="*80)
print("TEST 3: JSON format compliance")
print("="*80)

json_test_prompts = [
    'Respond as JSON: Type 1 shunt with ligate strategy. Keys: "classification" and "strategy".',
    'Output JSON only: {"classification": "Type 2B", "strategy": "perforator ligation"}',
    'Classify EP N1->N2, RP N2->N1 as JSON with classification and strategy keys.',
]

for prompt in json_test_prompts:
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    # Check if valid JSON
    try:
        parsed = json.loads(response)
        print(f"✓ Valid JSON: {parsed}")
    except:
        print(f"✗ Not JSON: {response[:100]}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Check the above outputs to understand:
1. Does model know definitions of each shunt type?
2. Can model distinguish between types based on flow patterns?
3. Can model output consistent JSON?

If answers are mostly 'no', the training data may not have included
enough examples of shunt classification or medical terminology.
""")
print("="*80)
