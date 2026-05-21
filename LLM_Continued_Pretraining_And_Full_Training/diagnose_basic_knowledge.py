"""
Test what the model actually understands about CHIVA shunt classification.
Diagnostic to check basic medical knowledge before debugging classification errors.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")  # Latest checkpoint

print("="*80)
print("DIAGNOSTIC: Basic CHIVA Knowledge Test")
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
print("✓ Model loaded")

def ask_model(question):
    """Ask model a question about CHIVA"""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(question):].strip()

# Test 1: Definition questions
print("\n" + "="*80)
print("TEST 1: Does the model know definitions?")
print("="*80)

definitions = {
    "What is a Type 1 CHIVA shunt?": "Type 1",
    "What is a Type 2A shunt?": "Type 2A",
    "What is a Type 2B shunt?": "Type 2B",
    "What is a Type 2C shunt?": "Type 2C",
    "What does Type 1+2 mean?": "Type 1+2",
    "What is Type 3 venous insufficiency?": "Type 3",
    "What does 'No Shunt' mean?": "No Shunt",
}

for question, expected_type in definitions.items():
    print(f"\nQ: {question}")
    answer = ask_model(question)
    print(f"A: {answer[:150]}")
    if expected_type.lower() in answer.lower():
        print(f"✓ Mentions {expected_type}")
    else:
        print(f"✗ Does NOT mention {expected_type}")

# Test 2: Flow pattern interpretation
print("\n" + "="*80)
print("TEST 2: Can it interpret flow patterns?")
print("="*80)

flow_tests = {
    "What does 'EP at N1->N2' mean?": "entry point",
    "What does 'RP at N2->N1' mean?": "re-entry point",
    "If there is only EP and no RP, is there a shunt?": "no shunt",
    "What is the difference between EP entering the superficial system vs a perforator?": "superficial vs perforator",
}

for question, expected_concept in flow_tests.items():
    print(f"\nQ: {question}")
    answer = ask_model(question)
    print(f"A: {answer[:150]}")

# Test 3: Classification logic
print("\n" + "="*80)
print("TEST 3: Can it apply decision logic?")
print("="*80)

logic_tests = {
    "If EP enters the superficial system and RP exits to deep system, what type is it?": "Type 1",
    "If only EP is present with no RP, is there a shunt?": "No Shunt",
    "If EP is via perforator and RP is via superficial veins, what type?": "Type 2A",
    "If EP is via perforator and RP goes back to deep system, what type?": "Type 2B",
    "If EP is via perforator with RP directly to deep (no superficial), what type?": "Type 2C",
    "If there are 2 distinct EP-RP pairs, what type?": "Type 1+2",
}

for question, expected_answer in logic_tests.items():
    print(f"\nQ: {question}")
    answer = ask_model(question)
    print(f"A: {answer[:150]}")
    if expected_answer.lower() in answer.lower():
        print(f"✓ Correct: mentions {expected_answer}")
    else:
        print(f"✗ Wrong: expected {expected_answer}")

# Test 4: Specific clip data
print("\n" + "="*80)
print("TEST 4: Classify specific clip patterns")
print("="*80)

clip_tests = {
    "Clips: EP at N1->N2, RP at N2->N1. Type?": "Type 1",
    "Clips: EP only at N1->N2, no RP. Type?": "No Shunt",
    "Clips: EP at N2->N2 (perforator), RP at N3->N1. Type?": "Type 2B",
    "Clips: EP at N3->N4 (perforator), RP at N4->N2 (superficial). Type?": "Type 2A",
    "Clips: EP at N1->N2 AND EP at N2->N3, RP at N3->N1 AND RP at N2->N1. Type?": "Type 1+2",
}

for question, expected_type in clip_tests.items():
    print(f"\nQ: {question}")
    answer = ask_model(question)
    print(f"A: {answer[:150]}")
    if expected_type.lower() in answer.lower():
        print(f"✓ Correct: {expected_type}")
    else:
        print(f"✗ Wrong: expected {expected_type}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
This diagnostic shows what the model actually learned about CHIVA.
If it fails most of these basic questions, then the training data
probably didn't include enough medical knowledge about shunt types.

If it passes these but fails the 34-case test, then it's a problem
with clip data interpretation or prompt engineering.
""")
print("="*80)
