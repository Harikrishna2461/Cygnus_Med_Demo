#!/usr/bin/env python3
"""
Test: Can the BASE model follow CHIVA instructions in the prompt?
This determines if fine-tuning will work at all.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

print("=" * 80)
print("TESTING BASE MODEL INSTRUCTION-FOLLOWING CAPABILITY")
print("=" * 80)

BASE_MODEL = "Qwen/Qwen2.5-7B"
CACHE_DIR = "./hf_cache"

print(f"\nLoading {BASE_MODEL}...")
print("(This will download ~15GB on first run)")

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

print("Model loaded.\n")

# ============================================================
# TEST 1: Can it follow JSON output instruction?
# ============================================================

print("=" * 80)
print("TEST 1: JSON Output Instruction")
print("=" * 80)

test1_prompt = """You are a CHIVA classification expert. Follow these rules EXACTLY:
1. Analyze the clips provided
2. Classify as TYPE 1, TYPE 2A, TYPE 2B, TYPE 2C, TYPE 1+2, TYPE 3, or NO SHUNT
3. Output ONLY valid JSON (no other text)
4. JSON fields: shunt_type, confidence (0-1), reasoning

Clips:
  Clip 00: EP N1→N2 at y=0.050 [SFJ-ENTRY=INCOMPETENT]
  Clip 01: RP N2→N1 at y=0.132 [GSV-TRUNK-REFLUX]

Respond with JSON only:"""

print("\nPrompt:")
print(test1_prompt)
print("\n" + "-" * 80)
print("Model Response:")
print("-" * 80)

inputs = tokenizer.encode(test1_prompt, return_tensors="pt").to(model.device)
model.eval()

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response1_only = response1[len(test1_prompt):].strip()
print(response1_only)

# Try to parse as JSON
try:
    json_obj = json.loads(response1_only)
    print("\n✓ Valid JSON output")
    print(f"  Shunt type: {json_obj.get('shunt_type')}")
    print(f"  Confidence: {json_obj.get('confidence')}")
except:
    print("\n✗ NOT valid JSON - Model did not follow instruction")

# ============================================================
# TEST 2: Can it follow CHIVA rule constraints?
# ============================================================

print("\n" + "=" * 80)
print("TEST 2: CHIVA Rule Constraint Following")
print("=" * 80)

test2_prompt = """You are a CHIVA expert. Apply these CHIVA rules:

RULE 1 (TYPE 1): IF EP N1→N2 present AND RP N2→N1 present AND NO EP N2→N3 → TYPE 1
RULE 2 (TYPE 3): IF EP N1→N2 present AND EP N2→N3 present AND RP N3 AND NO RP N2→N1 → TYPE 3
RULE 3 (TYPE 1+2): IF EP N1→N2 AND RP N2→N1 AND EP N2→N3 AND RP N3 → TYPE 1+2

Now classify:
Clips present: EP N1→N2, RP N2→N1, EP N2→N3, RP N3

What CHIVA type? Answer with ONLY the type name (TYPE 1, TYPE 3, or TYPE 1+2):"""

print("\nPrompt:")
print(test2_prompt)
print("\n" + "-" * 80)
print("Model Response:")
print("-" * 80)

inputs = tokenizer.encode(test2_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=20,
        do_sample=False,  # Greedy decoding for determinism
        pad_token_id=tokenizer.eos_token_id,
    )

response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response2_only = response2[len(test2_prompt):].strip().split('\n')[0]
print(response2_only)

# Check if it's correct
correct_answer = "TYPE 1+2"
if correct_answer in response2_only:
    print(f"\n✓ CORRECT - Model identified {correct_answer} per CHIVA rules")
else:
    print(f"\n✗ INCORRECT - Expected '{correct_answer}', got '{response2_only}'")

# ============================================================
# TEST 3: Can it follow management instructions?
# ============================================================

print("\n" + "=" * 80)
print("TEST 3: Management Instruction Following")
print("=" * 80)

test3_prompt = """You are a CHIVA management expert.

Given this classification and CHIVA rule, what should be done?

Classification: TYPE 1 (SFJ incompetence with isolated GSV reflux)
CHIVA Rule for TYPE 1: "Ligate at the SFJ (y ≤ 0.098) or Hunterian perforator (y ≤ 0.353)"

Respond with ONLY the ligation strategy from the rule (start with "Ligate"):"""

print("\nPrompt:")
print(test3_prompt)
print("\n" + "-" * 80)
print("Model Response:")
print("-" * 80)

inputs = tokenizer.encode(test3_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response3 = tokenizer.decode(outputs[0], skip_special_tokens=True)
response3_only = response3[len(test3_prompt):].strip()
print(response3_only)

if "Ligate" in response3_only:
    print("\n✓ Model followed management instruction")
else:
    print("\n✗ Model did not follow management instruction")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The base model's ability to follow in-prompt instructions is CRITICAL.

If the base model:
✓ Outputs valid JSON when asked
✓ Applies CHIVA rules correctly
✓ Follows management instructions
✓ Respects constraints in the prompt

THEN fine-tuning will ENHANCE these capabilities (model will be even better).

If the base model:
✗ Outputs garbage
✗ Ignores instructions
✗ Doesn't follow constraints

THEN fine-tuning won't fix this - the model lacks instruction-following ability.

Your Phase 1 (medical pretraining) helps with medical knowledge, but the base
model MUST already have instruction-following capability for this to work.
""")
print("=" * 80)
