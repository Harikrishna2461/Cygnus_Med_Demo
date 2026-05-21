#!/usr/bin/env python3
"""
Diagnose why training hangs/freezes the system.
Run this BEFORE attempting full training.
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import tracemalloc

print("=" * 80)
print("TRAINING HANG DIAGNOSTIC")
print("=" * 80)

# 1. Check GPU Memory
print("\n1. GPU Memory Check")
print("-" * 80)
if torch.cuda.is_available():
    print(f"GPU Available: Yes")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU Memory: {total_mem:.1f} GB")
    print(f"Currently Reserved: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    print(f"Currently Allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    if total_mem < 24:
        print(f"⚠️  WARNING: GPU has only {total_mem:.1f}GB (need ≥24GB)")
else:
    print("✗ CUDA not available!")
    exit(1)

# 2. Load and check data
print("\n2. Data Loading Check")
print("-" * 80)
train_file = Path("augmented_output/train.jsonl")
if not train_file.exists():
    print(f"✗ {train_file} not found!")
    exit(1)

print("Loading training data...")
train_data = []
with open(train_file) as f:
    for i, line in enumerate(f):
        if i >= 100:  # Just load first 100 for quick check
            break
        train_data.append(json.loads(line))

print(f"✓ Loaded {len(train_data)} chunks (sample)")

# Check token counts
token_counts = [c.get('token_count', 0) for c in train_data]
print(f"Token count range: {min(token_counts)} - {max(token_counts)}")

# 3. Tokenizer check
print("\n3. Tokenizer Loading Check")
print("-" * 80)
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded successfully")
    print(f"  Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    exit(1)

# 4. Quick tokenization test
print("\n4. Tokenization Speed Test")
print("-" * 80)
import time
sample_text = train_data[0]["text"][:1000]  # First 1000 chars
print(f"Sample text length: {len(sample_text)} chars")

start = time.time()
tokens = tokenizer(sample_text, truncation=True, max_length=2048, padding="max_length")
elapsed = time.time() - start
print(f"✓ Tokenization time: {elapsed:.3f}s")
print(f"  Output shape: {len(tokens['input_ids'])}")

# 5. Model loading check
print("\n5. Model Loading Check")
print("-" * 80)
print("Loading model (this may take a minute)...")
tracemalloc.start()

try:
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU first
        trust_remote_code=True,
    )
    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"✓ Model loaded (CPU)")
    print(f"  Load time: {elapsed:.1f}s")
    print(f"  Peak memory: {peak / 1e9:.1f} GB")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# 6. Move to GPU and test forward pass
print("\n6. GPU Forward Pass Check")
print("-" * 80)
print("Moving model to GPU...")
try:
    model = model.to("cuda:0")
    print("✓ Model moved to GPU")
    print(f"  GPU memory after load: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        inputs = tokenizer(
            "The diagnosis",
            return_tensors="pt",
            padding="max_length",
            max_length=2048,
        ).to("cuda:0")

        start = time.time()
        outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    print(f"✓ Forward pass successful ({elapsed:.2f}s)")
    print(f"  Output shape: {outputs.logits.shape}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    traceback.print_exc()
    exit(1)

# 7. Full batch test
print("\n7. Batch Processing Test")
print("-" * 80)
print("Testing batch loading and processing...")
try:
    dataset = Dataset.from_dict({"text": [c["text"] for c in train_data[:10]]})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    print("Tokenizing batch...")
    start = time.time()
    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=5, remove_columns=["text"])
    elapsed = time.time() - start

    print(f"✓ Batch tokenized ({elapsed:.2f}s)")
    print(f"  Dataset size: {len(tokenized)}")
    print(f"  Sample keys: {list(tokenized[0].keys())}")

except Exception as e:
    print(f"✗ Batch processing failed: {e}")
    traceback.print_exc()
    exit(1)

# 8. Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("""
✓ All checks passed!

If training still hangs, try these fixes:

1. REDUCE BATCH SIZE (memory issue)
   --per_device_train_batch_size 2
   --gradient_accumulation_steps 8

2. REDUCE SEQUENCE LENGTH (memory issue)
   --max_seq_length 1024

3. DISABLE GRADIENT CHECKPOINTING (if using)
   model.gradient_checkpointing_disable()

4. USE SIMPLER TRAINING SCRIPT
   python3 train_medical_cpt_simple.py

5. MONITOR DURING TRAINING
   Watch system with: watch -n 1 nvidia-smi

6. SET TIMEOUT (prevent freeze)
   timeout 3600 python3 train_medical_cpt.py
""")

print("=" * 80)
