#!/usr/bin/env python3
"""
PHASE 2 EVALUATION: TEST FINE-TUNED MODEL
Evaluate the fine-tuned Qwen2.5-7B LoRA on CHIVA classification tasks.
Tests on: classification, ligation, reasoning, and generalization.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("=" * 80)
print("PHASE 2: EVALUATE FINE-TUNED MODEL")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80 + "\n")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-7B"
PHASE2_LORA_PATH = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_chiva_tasks_lora"
CACHE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\.cache"

# Verify model exists
if not os.path.exists(PHASE2_LORA_PATH):
    print(f"ERROR: Phase 2 LoRA not found: {PHASE2_LORA_PATH}")
    print("Run phase2_direct_training.py first")
    exit(1)

print("[1/3] Loading tokenizer and base model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
print(f"[OK] Base model loaded")

print("[2/3] Loading Phase 2 LoRA adapter...")
model = PeftModel.from_pretrained(model, PHASE2_LORA_PATH)
model.eval()
print(f"[OK] Phase 2 LoRA loaded\n")

# ============================================================
# TEST CASES
# ============================================================

print("[3/3] Running evaluation tests...")
print()

test_cases = [
    # CLASSIFICATION TESTS
    {
        "category": "Classification - TYPE 1",
        "prompt": "Classify: EP N1->N2 at y=0.06 (SFJ-ENTRY) with RP N2->N1 at y=0.25. No N3 involvement.",
        "expected_keywords": ["TYPE 1", "SFJ", "saphenous trunk"]
    },
    {
        "category": "Classification - TYPE 2A",
        "prompt": "Classify: EP N2->N3 at y=0.20. No EP N1->N2 anywhere.",
        "expected_keywords": ["TYPE 2A", "GSV", "tributary", "SFJ competent"]
    },
    {
        "category": "Classification - TYPE 2B",
        "prompt": "Classify: EP N2->N2 at y=0.050 (perforator), RP N3->N1 at y=0.132. No N1->N2, no N2->N1 reflux.",
        "expected_keywords": ["TYPE 2B", "perforator"]
    },
    {
        "category": "Classification - TYPE 2C",
        "prompt": "Classify: EP N2->N2 at y=0.050, RP N3->N1 at y=0.132, RP N2->N1 at y=0.212. No EP N1->N2.",
        "expected_keywords": ["TYPE 2C", "perforator", "secondary"]
    },
    {
        "category": "Classification - TYPE 3",
        "prompt": "Classify: EP N1->N2 at y=0.05, EP N2->N3 at y=0.132, RP N3->N1 at y=0.212. No RP N2->N1.",
        "expected_keywords": ["TYPE 3", "tributary", "SFJ incompetence"]
    },
    {
        "category": "Classification - TYPE 1+2",
        "prompt": "Classify: EP N1->N2, EP N2->N3, RP N3->N1, RP N2->N1, eliminationTest=Reflux.",
        "expected_keywords": ["TYPE 1+2", "combined"]
    },
    {
        "category": "Classification - NO SHUNT",
        "prompt": "Classify: EP N1->N2 only, no RP clips at all.",
        "expected_keywords": ["NO SHUNT", "normal", "forward flow"]
    },

    # LIGATION STRATEGY TESTS
    {
        "category": "Ligation - TYPE 1",
        "prompt": "For TYPE 1 shunt, what is the ligation strategy?",
        "expected_keywords": ["SFJ", "ligate", "RP N2->N1"]
    },
    {
        "category": "Ligation - TYPE 2A",
        "prompt": "For TYPE 2A with multiple tributaries, how to approach ligation?",
        "expected_keywords": ["tributary", "calibre", "drainage", "N2->N3"]
    },
    {
        "category": "Ligation - TYPE 2B",
        "prompt": "For TYPE 2B perforator entry, what is the primary ligation target?",
        "expected_keywords": ["perforator", "N2->N2"]
    },
    {
        "category": "Ligation - TYPE 3",
        "prompt": "For TYPE 3 shunt with tributary ligation, what is the follow-up plan?",
        "expected_keywords": ["follow-up", "6-12 months", "SFJ", "reflux"]
    },
    {
        "category": "Ligation - TYPE 1+2 small RP",
        "prompt": "For TYPE 1+2 with small RP N2->N1, what is the CHIVA 2 approach?",
        "expected_keywords": ["CHIVA 2", "decompress", "tributary first"]
    },

    # RULE-BASED REASONING TESTS
    {
        "category": "Rules - SFJ Competence",
        "prompt": "Does EP N2->N2 at y=0.05 indicate SFJ incompetence?",
        "expected_keywords": ["perforator", "SFJ competent", "no", "N1->N2"]
    },
    {
        "category": "Rules - Decision Tree",
        "prompt": "What is the first step in CHIVA classification?",
        "expected_keywords": ["EP N1->N2", "check", "scan"]
    },
    {
        "category": "Rules - Anatomy",
        "prompt": "Define N1, N2, and N3 in CHIVA terminology.",
        "expected_keywords": ["deep venous", "saphenous", "tributary"]
    },

    # EDGE CASE / GENERALIZATION TESTS
    {
        "category": "Generalization - Variant position",
        "prompt": "Classify: EP N1->N2 at y=0.10 (slightly below SFJ) with RP N2->N1. No N3.",
        "expected_keywords": ["TYPE 1", "SFJ", "Hunterian"]
    },
    {
        "category": "Generalization - Multiple RP N2->N1",
        "prompt": "For TYPE 1 with TWO RP N2->N1 clips at y=0.15 and y=0.25, ligation strategy?",
        "expected_keywords": ["ligate below each", "distal", "RP N2->N1"]
    },
    {
        "category": "Generalization - Undetermined case",
        "prompt": "What happens when EP N1->N2, EP N2->N3, RP N3->N1, RP N2->N1 all present but no eliminationTest?",
        "expected_keywords": ["UNDETERMINED", "needs", "eliminationTest"]
    },
]

# ============================================================
# RUN TESTS
# ============================================================

results = []

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['category']}")
    print(f"  Prompt: {test['prompt'][:70]}...")

    # Generate response
    inputs = tokenizer.encode(test["prompt"], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(test["prompt"]):].strip()

    # Check for expected keywords
    matched_keywords = []
    for keyword in test["expected_keywords"]:
        if keyword.lower() in response.lower():
            matched_keywords.append(keyword)

    match_rate = len(matched_keywords) / len(test["expected_keywords"])

    print(f"  Response: {response[:100]}...")
    print(f"  Keywords: {matched_keywords} ({len(matched_keywords)}/{len(test['expected_keywords'])})")
    print(f"  Score: {match_rate * 100:.0f}%")
    print()

    results.append({
        "test": test["category"],
        "match_rate": match_rate,
        "matched": len(matched_keywords),
        "total": len(test["expected_keywords"])
    })

# ============================================================
# SUMMARY
# ============================================================

print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

total_matches = sum(r["matched"] for r in results)
total_expected = sum(r["total"] for r in results)
overall_score = total_matches / total_expected if total_expected > 0 else 0

print(f"Overall keyword match rate: {overall_score * 100:.1f}%")
print(f"Tests passed (>75%): {sum(1 for r in results if r['match_rate'] > 0.75)}/{len(results)}")
print()

print("Results by category:")
for result in results:
    status = "[PASS]" if result["match_rate"] > 0.75 else "[WARN]"
    print(f"  {status} {result['test']}: {result['matched']}/{result['total']}")

print()
print("=" * 80)
if overall_score > 0.85:
    print("MODEL STATUS: Strong performance on CHIVA tasks")
elif overall_score > 0.70:
    print("MODEL STATUS: Good performance, some edge cases need improvement")
else:
    print("MODEL STATUS: Moderate performance, consider additional training")
print("=" * 80)
