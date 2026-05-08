#!/usr/bin/env python3
"""
MERGE LORA ADAPTER WITH BASE MODEL AND UPLOAD TO HF
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
print("MERGE LORA + BASE MODEL AND UPLOAD")
print("="*80)

# Configuration
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
HF_USERNAME = "HariKrishna1824"
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_REPO = "HariKrishna1824/qwen25_chiva_v2"
OUTPUT_REPO = "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
OUTPUT_DIR = Path("./merged_model")

# Install dependencies
print("\n[1/5] Installing dependencies...")
packages = ['transformers', 'peft', 'torch', 'huggingface-hub']
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

    # Load base model
    print(f"  Loading base model: {BASE_MODEL}...", end=" ", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    print("OK")

    # Load LoRA
    print(f"  Loading LoRA adapter: {LORA_REPO}...", end=" ", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        LORA_REPO,
        token=HF_TOKEN,
        is_trainable=False
    )
    print("OK")

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

# Merge
print("\n[3/5] Merging LoRA with base model...")

try:
    print("  Merging...", end=" ", flush=True)
    merged_model = model.merge_and_unload()
    print("OK")

    print("  Model dtype:", merged_model.dtype)
    print("  Model device:", next(merged_model.parameters()).device)

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

# Save locally
print("\n[4/5] Saving merged model locally...")

try:
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep on GPU and use max_shard_size to avoid memory issues
    print(f"  Saving to: {OUTPUT_DIR}...", end=" ", flush=True)
    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="10GB"  # Split into 10GB shards
    )
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("OK")

    # Create README
    readme_content = f"""# Qwen2.5-7B - CHIVA Varicose Veins Treatment Fine-tuned

This is a merged model combining:
- **Base Model**: Qwen/Qwen2.5-7B
- **LoRA Adapter**: HariKrishna1824/qwen25_chiva_v2

## Model Details

- **Model Size**: ~7 Billion parameters
- **Base Model**: Qwen/Qwen2.5-7B
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Specialization**: CHIVA classification and varicose veins treatment planning
- **Training Data**: Medical literature, CHIVA clinical examples, venous insufficiency cases

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
)

# Generate
prompt = "Classify the CHIVA shunt type based on: EP N1->N2, RP N2->N1"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

## Tasks

This model is fine-tuned for:
1. **CHIVA Shunt Classification** - Classify venous shunt types based on duplex findings
2. **Ligation Planning** - Provide treatment strategies for varicose veins

## CHIVA Classification Rules

- **TYPE 1**: SFJ incompetent, GSV reflux only
- **TYPE 2A**: SFJ competent, perforator-driven reflux
- **TYPE 2B**: SFJ competent, saphenous trunk reflux without proximal backflow
- **TYPE 2C**: SFJ competent, saphenous trunk reflux with proximal backflow
- **TYPE 3**: SFJ incompetent with tributary reflux
- **TYPE 1+2**: Combined SFJ and perforator incompetence

## Citation

If you use this model, please cite:
- Qwen2.5-7B: Qwen Team
- CHIVA Classification: Cappelli et al., Eur J Vasc Endovasc Surg
"""

    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print("  README.md created")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

# Upload to HF
print("\n[5/5] Uploading merged model to Hugging Face...")

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

    # Upload
    print(f"  Uploading model...", end=" ", flush=True)
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message="Merged LoRA model: Qwen2.5-7B + CHIVA fine-tuning",
        ignore_patterns=["*.git*"]
    )
    print("OK")

except Exception as e:
    print(f"\n  ERROR: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)
print("COMPLETE")
print("="*80)

print(f"""
Merged Model Successfully Uploaded!

Repository: {OUTPUT_REPO}
URL: https://huggingface.co/{OUTPUT_REPO}

Base Model: {BASE_MODEL}
LoRA Adapter: {LORA_REPO}
Merged Size: ~28GB (full model)

Usage:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{OUTPUT_REPO}",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_REPO}")
""")
