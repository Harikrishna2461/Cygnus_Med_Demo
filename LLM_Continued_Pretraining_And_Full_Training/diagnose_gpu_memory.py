#!/usr/bin/env python3
"""
Comprehensive GPU Memory Diagnostic
- Shows what's using GPU memory
- Tests why training fails
- Identifies memory leaks
"""

import os
import sys
import torch
import psutil
import subprocess
from datetime import datetime

print("\n" + "="*80)
print("GPU MEMORY DIAGNOSTIC")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============ SECTION 1: NVIDIA-SMI OUTPUT ============
print("1. NVIDIA-SMI OUTPUT")
print("-"*80)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
except Exception as e:
    print(f"❌ Failed to run nvidia-smi: {e}")

# ============ SECTION 2: PYTORCH GPU STATE ============
print("\n2. PYTORCH GPU STATE")
print("-"*80)

if not torch.cuda.is_available():
    print("❌ CUDA NOT AVAILABLE")
    sys.exit(1)

print(f"✅ CUDA Available: True")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ Device Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    device_name = torch.cuda.get_device_name(i)
    device_cap = torch.cuda.get_device_capability(i)
    print(f"\nGPU {i}: {device_name}")
    print(f"  Compute Capability: {device_cap}")

    # Reset GPU memory tracking
    torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.empty_cache()

    # Get memory info
    total_mem = torch.cuda.get_device_properties(i).total_memory
    allocated = torch.cuda.memory_allocated(i)
    reserved = torch.cuda.memory_reserved(i)
    free = total_mem - reserved

    print(f"  Total Memory: {total_mem / 1e9:.2f} GB")
    print(f"  Allocated: {allocated / 1e9:.2f} GB")
    print(f"  Reserved: {reserved / 1e9:.2f} GB")
    print(f"  Free: {free / 1e9:.2f} GB")
    print(f"  Fragmentation: {((reserved - allocated) / reserved * 100):.1f}% if reserved > 0")

# ============ SECTION 3: SYSTEM MEMORY ============
print("\n3. SYSTEM MEMORY")
print("-"*80)

mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1e9:.2f} GB")
print(f"Used RAM: {mem.used / 1e9:.2f} GB")
print(f"Free RAM: {mem.available / 1e9:.2f} GB")
print(f"Memory %: {mem.percent}%")

if mem.percent > 90:
    print("⚠️  WARNING: RAM is >90% full!")

# ============ SECTION 4: PYTHON PROCESSES ============
print("\n4. PYTHON PROCESSES")
print("-"*80)

python_procs = [p for p in psutil.process_iter(['pid', 'name', 'memory_info'])
                if 'python' in p.info['name'].lower()]

if python_procs:
    print(f"Found {len(python_procs)} Python process(es):")
    for p in python_procs:
        try:
            mem_mb = p.info['memory_info'].rss / 1e6
            print(f"  PID {p.info['pid']}: {mem_mb:.1f} MB")
        except:
            pass
else:
    print("✅ No Python processes using significant memory")

# ============ SECTION 5: TORCH MEMORY TEST ============
print("\n5. TORCH MEMORY ALLOCATION TEST")
print("-"*80)

try:
    # Try allocating test tensor
    print("Testing memory allocation...")

    # Start with small allocation
    test_size = 100 * 1024 * 1024  # 100 MB
    print(f"Allocating {test_size / 1e6:.0f} MB...", end=" ")
    x = torch.randn(test_size // 4, dtype=torch.float32).cuda()
    print("✅ Success")

    allocated = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU Memory allocated: {allocated:.2f} GB")

    del x
    torch.cuda.empty_cache()
    print("✅ Memory freed")

except RuntimeError as e:
    print(f"❌ Failed: {e}")
    print("\nThis indicates GPU memory allocation is failing!")

# ============ SECTION 6: TOKENIZER/MODEL TEST ============
print("\n6. TOKENIZER & MODEL LOADING TEST")
print("-"*80)

try:
    from transformers import AutoTokenizer

    print("Loading tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B",
        trust_remote_code=True,
        use_fast=True,
    )
    print("✅ Success")
    print(f"Vocab size: {len(tokenizer):,}")

except Exception as e:
    print(f"❌ Failed: {e}")

# ============ SECTION 7: MODEL LOADING TEST ============
print("\n7. MODEL LOADING TEST")
print("-"*80)

try:
    from transformers import AutoModelForCausalLM

    print("Loading model in bfloat16...", end=" ")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Success")

    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
    current_mem = torch.cuda.memory_allocated(0) / 1e9

    print(f"Model loaded successfully")
    print(f"Peak memory during load: {peak_mem:.2f} GB")
    print(f"Current memory allocated: {current_mem:.2f} GB")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e9:.2f}B")

    del model
    torch.cuda.empty_cache()
    print("✅ Model unloaded and memory freed")

except RuntimeError as e:
    print(f"❌ Failed: {e}")
    print("\nModel loading failed - this is likely why training crashes!")

# ============ SECTION 8: DATA LOADING TEST ============
print("\n8. DATA LOADING TEST")
print("-"*80)

try:
    from datasets import Dataset

    print("Creating dummy dataset...", end=" ")
    dummy_texts = ["The medical diagnosis is " + str(i) for i in range(100)]
    dataset = Dataset.from_dict({"text": dummy_texts})
    print("✅ Success")

    print(f"Dataset samples: {len(dataset):,}")

except Exception as e:
    print(f"❌ Failed: {e}")

# ============ SECTION 9: SUMMARY & RECOMMENDATIONS ============
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
allocated = torch.cuda.memory_allocated(0) / 1e9
free = total_gpu - allocated

print(f"\nGPU Status:")
print(f"  Total: {total_gpu:.2f} GB")
print(f"  Used: {allocated:.2f} GB")
print(f"  Free: {free:.2f} GB")

if allocated > total_gpu * 0.9:
    print(f"\n❌ CRITICAL: GPU is {allocated/total_gpu*100:.0f}% full!")
    print("   Kill all processes and restart Python")
elif free < 10:
    print(f"\n⚠️  WARNING: Less than 10GB free ({free:.2f} GB)")
    print("   Training may fail. Free up space or reduce batch size.")
else:
    print(f"\n✅ GPU has sufficient free memory ({free:.2f} GB)")

print("\n" + "="*80)
