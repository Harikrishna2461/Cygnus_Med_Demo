"""
Comprehensive test on all 34 medical shunt classification cases.
Uses full CHIVA rules + decision guide + optimized prompt.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
from collections import defaultdict

TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")  # Latest checkpoint
TEST_CASES_FILE = Path("test_cases_for_evaluation.json")
OUTPUT_FILE = Path("logs") / "comprehensive_34_case_results.json"

# Full CHIVA classification system with complete decision guide
CHIVA_SYSTEM = """CHIVA VENOUS SHUNT CLASSIFICATION SYSTEM:

TYPE 1 - Superficial Truncal Incompetence:
  Definition: Direct deep-to-superficial reflux via incompetent truncal vein
  EP Location: Superficial system (e.g., saphenous trunk)
  RP Location: Deep system (e.g., femoral vein)
  Key Indicator: Direct EP-to-RP loop in truncal system
  Example: EP N1->N2 (saphenous), RP N2->N1 (deep)
  Ligation: Truncal vein ligation

TYPE 2A - Perforator + Superficial Incompetence:
  Definition: Perforator incompetence with superficial venous drainage
  EP Location: Via perforator from deep system
  RP Location: Via superficial veins
  Key Indicator: Perforator entry + superficial exit
  Example: EP N3->N4 (perforator), RP N4->N2 (superficial)
  Ligation: Perforator + superficial vein ligation

TYPE 2B - Perforator with Deep Reflux:
  Definition: Perforator incompetence with superficial reflux back to deep system
  EP Location: Via perforator from deep system
  RP Location: Via superficial reflux back to deep
  Key Indicator: Perforator involvement, RP goes back to deep
  Example: EP N2->N2 (perforator), RP N3->N1 (reflux to deep)
  Ligation: Perforator ligation + superficial reflux management

TYPE 2C - Perforator Only:
  Definition: Pure perforator incompetence without superficial involvement
  EP Location: Via perforator from deep system
  RP Location: Directly to deep system (no superficial)
  Key Indicator: Perforator-to-deep pathway only
  Example: EP N3->N4 (perforator), RP N4->N3 (deep only)
  Ligation: Perforator ligation

TYPE 1+2 - Combined Shunts:
  Definition: Multiple independent shunt pathways present
  Indicators: Both Type 1 AND Type 2 components present
  Key: Look for 2+ distinct EP-RP pairs with different characteristics
  Ligation: Combined approach for both pathways

TYPE 3 - Perforator or Branch Incompetence:
  Definition: Mid-calf perforator incompetence without major truncal involvement
  Key Indicator: Branch veins or mid-calf perforators
  Ligation: Selective perforator ligation

NO SHUNT:
  Definition: Normal venous hemodynamics without pathological reflux
  Key Indicator: Antegrade flow only, no bidirectional flow
  Example: Normal venous return

DECISION ALGORITHM:
1. Check if any bidirectional flow (reflux) exists
   NO → Classification: NO SHUNT
   YES → Continue

2. Count distinct shunt pathways (EP-RP pairs)
   Multiple (2+) → Classification: TYPE 1+2
   Single → Continue

3. Analyze single EP-RP pair
   EP in superficial?
   YES → Classification: TYPE 1
   NO (perforator) → Continue

4. For perforator shunt, check RP location
   RP in superficial veins?
   YES → Classification: TYPE 2A
   NO → Check if RP goes to deep system
        YES → Classification: TYPE 2B
        NO → Classification: TYPE 2C"""

GEN_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.2,
    "do_sample": False,
    "repetition_penalty": 1.0,
}

# Load model
print("="*80)
print("LOADING MEDICAL LLM FOR 34-CASE COMPREHENSIVE TEST")
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
print(f"✓ Model loaded to {next(model.parameters()).device}")

# Load test cases
print(f"\nLoading test cases from {TEST_CASES_FILE}...")
if not TEST_CASES_FILE.exists():
    print(f"ERROR: {TEST_CASES_FILE} not found!")
    exit(1)

with open(TEST_CASES_FILE) as f:
    test_data = json.load(f)

test_cases = test_data if isinstance(test_data, list) else test_data.get('cases', [])
print(f"✓ Loaded {len(test_cases)} test cases")

def classify_shunt(clip_data, ground_truth=None):
    """Classify shunt using CHIVA rules"""
    prompt = f"""{CHIVA_SYSTEM}

ANALYZE THIS SHUNT:
Clip data: {clip_data}

Follow the decision algorithm. Output ONLY JSON (no text before/after):
{{"classification": "Type 1 or Type 2A or Type 2B or Type 2C or Type 1+2 or Type 3 or No Shunt", "reasoning": "brief"}}"""

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

# Run comprehensive test
print("\n" + "="*80)
print(f"TESTING ON ALL {len(test_cases)} CASES")
print("="*80)

results = []
accuracy_by_type = defaultdict(lambda: {"correct": 0, "total": 0})

for idx, test in enumerate(test_cases, 1):
    # Extract test info from actual file structure
    name = test.get('id', f'Case {idx}')
    clips = test.get('clips_str', 'N/A')
    expected_raw = test.get('ground_truth_shunt', 'Unknown')
    shunt_label = test.get('shunt_type_label', '')

    # Convert ground truth to standard format
    if expected_raw == 'Unknown':
        if 'No_Shunt' in shunt_label:
            expected = 'No Shunt'
        else:
            expected = 'Unknown'
    else:
        # Convert "1" -> "Type 1", "2A" -> "Type 2A", etc.
        if expected_raw == '1+2':
            expected = 'Type 1+2'
        else:
            expected = f'Type {expected_raw}'

    # Normalize expected for comparison
    expected_norm = expected.replace('_', ' ').lower() if expected != 'Unknown' else 'unknown'

    # Print progress for every case
    print(f"[{idx:2}/34] {name[:45]:<45} ", end='', flush=True)

    response = classify_shunt(clips)

    try:
        parsed = json.loads(response)
        classification = parsed.get('classification', 'ERROR')
        reasoning = parsed.get('reasoning', '')

        # Normalize for comparison
        pred_norm = classification.replace('_', ' ').lower()
        match = pred_norm == expected_norm and expected != 'Unknown'

        if expected != 'Unknown':
            accuracy_by_type[expected_norm]["total"] += 1
            if match:
                accuracy_by_type[expected_norm]["correct"] += 1

        status = "✓" if match else "✗"
        pred_display = classification.replace('_', ' ')
        exp_display = expected.replace('_', ' ')
        print(f"{status} {pred_display:12} (exp: {exp_display})")

        results.append({
            "index": idx,
            "name": name,
            "clips": clips,
            "expected": expected,
            "predicted": classification,
            "reasoning": reasoning,
            "correct": match,
            "raw_response": response
        })

    except Exception as e:
        print(f"✗ ERROR: {str(e)[:40]}")
        results.append({
            "index": idx,
            "name": name,
            "clips": clips,
            "expected": expected,
            "error": str(e),
            "raw_response": response
        })

# Summary
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

correct_count = sum(1 for r in results if r.get('correct', False))
total_count = len([r for r in results if 'correct' in r])
overall_accuracy = 100 * correct_count / total_count if total_count > 0 else 0

print(f"\nOVERALL ACCURACY: {correct_count}/{total_count} ({overall_accuracy:.1f}%)")

print(f"\nAccuracy by Shunt Type:")
for shunt_type in sorted(accuracy_by_type.keys()):
    stats = accuracy_by_type[shunt_type]
    acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    bar = "█" * int(acc/5) + "░" * (20 - int(acc/5))
    print(f"  {shunt_type.upper():15} {stats['correct']:2}/{stats['total']:2} ({acc:5.1f}%) {bar}")

print(f"\n" + "="*100)
print(f"DETAILED RESULTS WITH FULL MODEL OUTPUT (ALL {len(results)} cases)")
print("="*100)

error_count = 0
for r in results:
    status = "✓" if r.get('correct') else "✗"
    print(f"\n[{r['index']:2}] {status} {r['name']}")
    print(f"    Clips: {r['clips']}")

    if 'error' in r:
        print(f"    ERROR: {r['error']}")
    else:
        print(f"    Raw Model Output: {r['raw_response']}")
        pred = r['predicted'].replace('_', ' ')
        exp = r['expected'].replace('_', ' ')
        print(f"    Parsed: {pred}")
        print(f"    Expected: {exp}")
        if r.get('reasoning'):
            print(f"    Reasoning: {r['reasoning']}")
        error_count += 1 if 'error' in r else 0

# Save comprehensive results
OUTPUT_FILE.parent.mkdir(exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "model": str(TRAINED_MODEL_PATH),
        "total_cases": len(test_cases),
        "total_evaluated": total_count,
        "overall_accuracy": f"{correct_count}/{total_count} ({overall_accuracy:.1f}%)",
        "accuracy_by_type": dict(accuracy_by_type),
        "results": results
    }, f, indent=2)

print(f"\nFull results saved to: {OUTPUT_FILE}")
print("="*80)
