#!/usr/bin/env python3
"""
CHIVA Model Inference
Test the fine-tuned model interactively on classification and ligation tasks.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_PATH = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_chiva_tasks_lora"
CACHE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\.cache"

print("=" * 80)
print("CHIVA Model - Interactive Inference")
print("=" * 80)
print(f"Loading model from: {LORA_PATH}")
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

# Load LoRA
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("[OK] Model loaded\n")

# Example prompts
examples = [
    "Classify: EP N1->N2 at y=0.06 (SFJ-ENTRY) with RP N2->N1 at y=0.25. No N3 involvement.",
    "For TYPE 1 shunt with EP N1->N2 at the SFJ, what is the ligation strategy?",
    "Classify: EP N2->N3 at y=0.20 with no EP N1->N2 anywhere.",
    "Define N1, N2, N3 in CHIVA classification.",
    "Classify: EP N2->N2 at y=0.050 (perforator), RP N3->N1 at y=0.132. No N1->N2, no N2->N1 reflux.",
]

print("=" * 80)
print("EXAMPLE QUERIES:")
print("=" * 80)
for i, prompt in enumerate(examples, 1):
    print(f"\n[{i}] QUERY: {prompt}")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    print(f"    RESPONSE: {response}\n")

print()
print("=" * 80)
print("Interactive Mode - Type 'quit' to exit")
print("=" * 80)

while True:
    print()
    user_prompt = input("Enter your CHIVA question: ").strip()

    if user_prompt.lower() == 'quit':
        break

    if not user_prompt:
        continue

    inputs = tokenizer.encode(user_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(user_prompt):].strip()

    print(f"\nResponse:\n{response}\n")
