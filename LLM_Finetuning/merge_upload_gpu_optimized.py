#!/usr/bin/env python3
"""
MERGE LORA + BASE MODEL ON GPU - OPTIMIZED
LoRA: HariKrishna1824/qwen25_chiva_v2
Base: Qwen/Qwen2.5-7B
Output: HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

print("="*80)
print("MERGE LORA + BASE MODEL (GPU OPTIMIZED)")
print("="*80)

# Configuration
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
HF_USERNAME = "HariKrishna1824"
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_REPO = "HariKrishna1824/qwen25_chiva_v2"
OUTPUT_REPO = "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
OUTPUT_DIR = Path("./merged_model")

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Install dependencies
print("\n[1/5] Installing dependencies...")
packages = ['transformers', 'peft', 'torch', 'huggingface-hub', 'peft']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"  {pkg}...", end=" ", flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
        print("OK")

# Import
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

print("\n[2/5] Loading base model and LoRA...")

try:
    # Login
    login(token=HF_TOKEN)
    api = HfApi()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Load base model - keep on GPU
    print(f"  Loading base: {BASE_MODEL}...", end=" ", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    print(f"OK ({next(base_model.parameters()).device})")

    # Load LoRA - keep on GPU
    print(f"  Loading LoRA: {LORA_REPO}...", end=" ", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        LORA_REPO,
        token=HF_TOKEN,
        is_trainable=False
    )
    print(f"OK ({next(model.parameters()).device})")

    # Load tokenizer
    print(f"  Loading tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_REPO,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    print("OK")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

# Merge on GPU
print("\n[3/5] Merging LoRA with base on GPU...")

try:
    print("  Merging...", end=" ", flush=True)
    merged_model = model.merge_and_unload()
    print("OK")

    device = next(merged_model.parameters()).device
    dtype = merged_model.dtype
    print(f"  Model dtype: {dtype}")
    print(f"  Model device: {device}")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

# Save with streaming
print("\n[4/5] Saving merged model (streaming to disk)...")

try:
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Saving to: {OUTPUT_DIR}")
    print("  Using sharded save (10GB chunks)...", end=" ", flush=True)

    # Save with max_shard_size to handle large model
    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="10GB",  # 10GB shards
        push_to_hub=False
    )

    print("OK")

    print("  Saving tokenizer...", end=" ", flush=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("OK")

    # Get folder size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file())
    print(f"  Total size: {total_size / (1024**3):.2f}GB")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

# Create README
print("\n  Creating README.md...", end=" ", flush=True)
readme_content = """# Qwen2.5-7B CHIVA Varicose Veins Treatment - Merged Model

Fully merged model: Qwen2.5-7B base + CHIVA LoRA fine-tuning.

## Model Details

- **Base**: Qwen/Qwen2.5-7B (7B parameters)
- **Fine-tuning**: LoRA merged and integrated
- **Specialization**: CHIVA classification + varicose veins treatment
- **Status**: Production-ready (no PEFT required)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(repo)

prompt = "Classify the CHIVA shunt type: EP N1->N2, RP N2->N1"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

## Capabilities

1. **CHIVA Shunt Classification**: TYPE 1, 2A, 2B, 2C, 3, 1+2
2. **Ligation Planning**: Treatment strategies based on shunt type
3. **Medical Reasoning**: Explains clinical rationale
4. **Query Formats**: Raw clips or natural language medical descriptions

## Training Data

- CHIVA classification rules and examples
- Venous insufficiency clinical cases
- Duplex ultrasound interpretation
- Surgical treatment planning guidelines
"""

with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write(readme_content)
print("OK")

# Upload to HF
print("\n[5/5] Uploading to Hugging Face...")

try:
    # Create repo
    print(f"  Creating repo: {OUTPUT_REPO}...", end=" ", flush=True)
    try:
        api.create_repo(
            repo_id=OUTPUT_REPO,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("OK")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("(exists)")
        else:
            raise

    # Upload folder
    print(f"  Uploading model files...", end=" ", flush=True)
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message="Upload: Qwen2.5-7B + CHIVA LoRA merged model",
        ignore_patterns=["*.git*"]
    )
    print("OK")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)
print("SUCCESS")
print("="*80)

print(f"""
Model Successfully Merged and Uploaded!

Repository: https://huggingface.co/{OUTPUT_REPO}

Components:
  Base: {BASE_MODEL}
  LoRA: {LORA_REPO}
  Merged: {OUTPUT_REPO}

Status: Ready for production use
No PEFT library required - fully merged model

Usage:
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("{OUTPUT_REPO}")
""")
