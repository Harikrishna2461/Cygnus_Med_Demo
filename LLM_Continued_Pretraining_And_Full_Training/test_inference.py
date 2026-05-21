"""
Test the trained medical LLM on shunt classification and ligation planning tasks.
Compares trained model output with base model for quality assessment.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")  # Latest checkpoint
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_FILE = Path("logs") / "inference_results.json"

# Test prompts - improved to avoid template copying
test_prompts = [
    {
        "name": "Test 1: Simple Shunt with EP->RP",
        "clips": "EP at N2->N2,RP at N3->N1",
        "prompt": """Analyze the following venous shunt hemodynamics using CHIVA classification rules.

Clip data (flow patterns): EP at N2->N2, RP at N3->N1

Based on the entry point (EP) and re-entry point (RP) flow patterns, determine:
1. The shunt type: Choose from [Type 1, Type 2A, Type 2B, Type 2C, Type 1+2, No Shunt]
2. The appropriate ligation strategy to eliminate the shunt

Output only a JSON response with exactly two keys:
- "shunt_classification": your determined shunt type
- "ligation": the ligation strategy

Do not include any other text or explanation."""
    },
    {
        "name": "Test 2: Complex Shunt with Multiple Nodes",
        "clips": "EP at N1->N2,EP at N2->N3,RP at N3->N2,RP at N2->N1",
        "prompt": """Analyze the following venous shunt hemodynamics using CHIVA classification rules.

Clip data (flow patterns): EP at N1->N2, EP at N2->N3, RP at N3->N2, RP at N2->N1

Based on the entry point (EP) and re-entry point (RP) flow patterns, determine:
1. The shunt type: Choose from [Type 1, Type 2A, Type 2B, Type 2C, Type 1+2, No Shunt]
2. The appropriate ligation strategy to eliminate the shunt

Output only a JSON response with exactly two keys:
- "shunt_classification": your determined shunt type
- "ligation": the ligation strategy

Do not include any other text or explanation."""
    },
    {
        "name": "Test 3: Simple Bidirectional Shunt",
        "clips": "EP at N1->N2,RP at N2->N1",
        "prompt": """Analyze the following venous shunt hemodynamics using CHIVA classification rules.

Clip data (flow patterns): EP at N1->N2, RP at N2->N1

Based on the entry point (EP) and re-entry point (RP) flow patterns, determine:
1. The shunt type: Choose from [Type 1, Type 2A, Type 2B, Type 2C, Type 1+2, No Shunt]
2. The appropriate ligation strategy to eliminate the shunt

Output only a JSON response with exactly two keys:
- "shunt_classification": your determined shunt type
- "ligation": the ligation strategy

Do not include any other text or explanation."""
    }
]

# Generation config - optimized for medical classification JSON output
GEN_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.3,  # Lower temp for more deterministic JSON output
    "top_p": 0.95,
    "do_sample": False,  # Greedy decoding for more consistent output
    "repetition_penalty": 1.05,
}

# ── Load Models ────────────────────────────────────────────────────────────────
print("="*80)
print("LOADING MODELS")
print("="*80)

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TRAINED_MODEL_PATH))
print(f"   ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

print("\n2. Loading trained medical LLM...")
trained_model = AutoModelForCausalLM.from_pretrained(
    str(TRAINED_MODEL_PATH),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
trained_model.eval()
print(f"   ✓ Trained model loaded")

print("\n3. Loading base model for comparison...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    print(f"   ✓ Base model loaded (from HuggingFace)")
    compare_with_base = True
except Exception as e:
    print(f"   ⚠ Could not load base model for comparison: {e}")
    compare_with_base = False

# ── Inference Function ─────────────────────────────────────────────────────────
def generate_response(model, prompt, model_name="model"):
    """Generate response from model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GEN_CONFIG,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()

        # Extract JSON if present
        if '{' in response and '}' in response:
            try:
                json_start = response.index('{')
                json_end = response.rindex('}') + 1
                response = response[json_start:json_end]
            except (ValueError, IndexError):
                pass

        return response
    except Exception as e:
        return f"ERROR: {str(e)}"

# ── Run Tests ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("RUNNING INFERENCE TESTS")
print("="*80)

results = []

for i, test in enumerate(test_prompts, 1):
    print(f"\n{'─'*80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─'*80}")

    prompt = test['prompt']
    clips = test.get('clips', 'N/A')
    print(f"\nClip Data: {clips}")
    print(f"\nPrompt:\n{prompt}\n")

    # Trained model
    print("Generating response from TRAINED model...")
    trained_response = generate_response(trained_model, prompt, "trained")

    print(f"\n📋 Trained Model Response:")
    print(f"{trained_response}")

    # Try to parse as JSON
    try:
        trained_json = json.loads(trained_response)
        print(f"\n✓ Valid JSON: {trained_json}")
        if 'shunt_classification' in trained_json:
            print(f"  → Classification: {trained_json['shunt_classification']}")
        if 'ligation' in trained_json:
            print(f"  → Ligation: {trained_json['ligation']}")
    except:
        print(f"\n⚠ Not valid JSON format")

    # Base model (if available)
    base_response = None
    if compare_with_base:
        print("\n" + "-"*40)
        print("Generating response from BASE model...")
        base_response = generate_response(base_model, prompt, "base")

        print(f"\n📋 Base Model Response:")
        print(f"{base_response}")

        try:
            base_json = json.loads(base_response)
            print(f"\n✓ Valid JSON: {base_json}")
        except:
            print(f"\n⚠ Not valid JSON format")

    # Extract classifications from JSON if available
    trained_class = None
    trained_ligation = None
    base_class = None

    try:
        trained_json = json.loads(trained_response)
        trained_class = trained_json.get('shunt_classification')
        trained_ligation = trained_json.get('ligation')
    except:
        pass

    if compare_with_base and base_response:
        try:
            base_json = json.loads(base_response)
            base_class = base_json.get('shunt_classification')
        except:
            pass

    # Store results
    result_entry = {
        "test_number": i,
        "test_name": test['name'],
        "clips": clips,
        "trained_response": trained_response,
        "trained_classification": trained_class,
        "trained_ligation": trained_ligation,
        "base_response": base_response,
        "base_classification": base_class
    }
    results.append(result_entry)

# ── Save Results ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_data = {
    "timestamp": datetime.now().isoformat(),
    "trained_model": str(TRAINED_MODEL_PATH),
    "base_model": BASE_MODEL_NAME,
    "generation_config": GEN_CONFIG,
    "results": results
}

OUTPUT_FILE.parent.mkdir(exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_FILE}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTests completed: {len(results)}/3")
print(f"Results file: {OUTPUT_FILE}")
print(f"\nTo analyze results:")
print(f"  import json")
print(f"  with open('{OUTPUT_FILE}') as f:")
print(f"      results = json.load(f)")
print(f"      for r in results['results']:")
print(f"          print(r['trained_response'])")

print("\n" + "="*80)
