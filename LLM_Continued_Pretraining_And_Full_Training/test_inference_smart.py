"""
Medical LLM inference with comprehensive CHIVA classification rules.
Uses detailed decision guide to prevent default Type 2B output.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")  # Latest checkpoint
OUTPUT_FILE = Path("logs") / "smart_inference_results.json"

# CHIVA classification system with decision rules
CHIVA_SYSTEM_PROMPT = """You are an expert in CHIVA venous hemodynamics classification. Use these rules EXACTLY:

CLASSIFICATION DECISION GUIDE:
================================

1. CHECK FOR NO SHUNT:
   - If clip data shows normal venous return or no bidirectional flow → NO SHUNT
   - If only antegrade flow (no reflux) → NO SHUNT

2. IF SHUNT EXISTS, DETERMINE TYPE:

TYPE 1 (Superficial Truncal Incompetence):
   - EP: enters directly into superficial system (truncal vein incompetence)
   - RP: re-enters from superficial to deep system
   - KEY: Direct EP-RP loop in superficial system
   - Example: EP N1->N2 (saphenous trunk), RP N2->N1 (deep vein)

TYPE 2A (Perforator + Superficial Incompetence):
   - EP: enters via perforator(s) from deep system
   - RP: re-enters via superficial veins
   - KEY: Perforator drainage with superficial involvement
   - Superficial system is incompetent

TYPE 2B (Perforator + Superficial Reflux to Deep):
   - EP: enters via perforator from deep system
   - RP: re-enters via superficial reflux back into deep system
   - KEY: Perforator involvement but RP back to deep
   - Superficial acts as reflux channel

TYPE 2C (Perforator Only):
   - EP: enters via perforator from deep system
   - RP: re-enters directly into deep system (no superficial involvement)
   - KEY: Pure perforator-to-deep pathway
   - No significant superficial reflux

TYPE 1+2 (Combined):
   - BOTH Type 1 AND Type 2 shunts present simultaneously
   - Multiple distinct shunt pathways
   - KEY: Look for multiple independent EP-RP pairs

DECISION TREE:
==============
1. Does data show normal flow? → NO SHUNT
2. Are there multiple independent shunt pathways? → TYPE 1+2
3. Is EP in superficial (truncal)?
   Yes → TYPE 1
   No → Check RP location:
        Superficial? → TYPE 2A or 2B (check if RP goes to deep = 2B, to superficial = 2A)
        Deep only? → TYPE 2C

IMPORTANT: Base classification ONLY on the flow patterns provided. Do not assume or guess."""

# Test cases with correct answers based on CHIVA rules
test_cases = [
    {
        "name": "Simple Type 1",
        "clips": "EP at N1->N2 (saphenous), RP at N2->N1 (deep)",
        "answer": "Type 1",
        "reason": "Direct superficial truncal incompetence with deep re-entry"
    },
    {
        "name": "Type 2B - Perforator to Deep",
        "clips": "EP at N2->N2 (perforator), RP at N3->N1 (reflux into deep)",
        "answer": "Type 2B",
        "reason": "Perforator entry with superficial reflux back to deep system"
    },
    {
        "name": "Type 1+2 - Combined",
        "clips": "EP at N1->N2 (truncal) RP at N2->N1 (deep); EP at N2->N3 (perforator) RP at N3->N1 (deep)",
        "answer": "Type 1+2",
        "reason": "Two distinct shunt pathways present"
    },
    {
        "name": "Type 2A - Perforator to Superficial",
        "clips": "EP at N3->N4 (deep perforator), RP at N4->N2 (superficial)",
        "answer": "Type 2A",
        "reason": "Perforator entry with superficial re-entry"
    },
    {
        "name": "No Shunt",
        "clips": "Normal venous return, no bidirectional flow",
        "answer": "No Shunt",
        "reason": "Normal hemodynamics without pathological reflux"
    }
]

GEN_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.2,  # Even lower for rule-following
    "do_sample": False,
    "repetition_penalty": 1.0,
}

# Load model
print("="*80)
print("LOADING TRAINED MEDICAL LLM WITH CHIVA CLASSIFICATION")
print("="*80)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TRAINED_MODEL_PATH))
print(f"✓ Tokenizer loaded")

print("Loading trained model...")
model = AutoModelForCausalLM.from_pretrained(
    str(TRAINED_MODEL_PATH),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()
print(f"✓ Model loaded")

def classify_with_rules(clip_data):
    """Classify using CHIVA rules as system prompt"""
    prompt = f"""{CHIVA_SYSTEM_PROMPT}

TASK: Classify shunt for clip data: {clip_data}

Output ONLY valid JSON, no text before or after JSON:
{{"shunt_type": "Type 1 or Type 2A or Type 2B or Type 2C or Type 1+2 or No Shunt", "reasoning": "one line"}}"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GEN_CONFIG,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Extract JSON with proper brace matching
        if '{' in response:
            json_start = response.index('{')
            brace_count = 0
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        response = response[json_start:i+1]
                        break

        return response
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

# Run tests
print("\n" + "="*80)
print("TESTING WITH CHIVA RULES (5 Cases)")
print("="*80)

results = []
correct = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─'*80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─'*80}")
    print(f"Clips: {test['clips']}")
    print(f"Expected: {test['answer']} - {test['reason']}")

    print("\nClassifying...")
    response = classify_with_rules(test['clips'])

    print(f"\nModel output: {response}")

    try:
        parsed = json.loads(response)
        shunt_type = parsed.get('shunt_type', 'ERROR')
        reasoning = parsed.get('reasoning', '')

        print(f"✓ Parsed: {shunt_type}")
        if reasoning:
            print(f"  Reasoning: {reasoning}")

        # Normalize for comparison
        normalized_pred = shunt_type.replace('_', ' ').lower()
        normalized_exp = test['answer'].replace('_', ' ').lower()
        match = normalized_pred == normalized_exp
        if match:
            print(f"✓ CORRECT!")
            correct += 1
        else:
            print(f"✗ WRONG (expected {test['answer']})")

        results.append({
            "test": i,
            "name": test['name'],
            "clips": test['clips'],
            "expected": test['answer'],
            "predicted": shunt_type,
            "reasoning": reasoning,
            "correct": match
        })
    except Exception as e:
        print(f"✗ Parse error: {e}")
        results.append({
            "test": i,
            "name": test['name'],
            "clips": test['clips'],
            "expected": test['answer'],
            "error": str(e),
            "raw_response": response
        })

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct//len(test_cases)}%)")
print(f"\nDetailed results:")
for r in results:
    if 'correct' in r:
        status = "✓" if r['correct'] else "✗"
        pred = r['predicted'].replace('_', ' ')
        exp = r['expected'].replace('_', ' ')
        print(f"{status} Test {r['test']}: {pred} (expected {exp})")
    else:
        print(f"✗ Test {r['test']}: ERROR - {r['error']}")

# Save
OUTPUT_FILE.parent.mkdir(exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "model": str(TRAINED_MODEL_PATH),
        "accuracy": f"{correct}/{len(test_cases)}",
        "results": results
    }, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")
print("="*80)
